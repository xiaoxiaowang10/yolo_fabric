"""
增量训练模块 —— 扩展分类数量而不需要从头训练

原理:
  1. 解析旧模型，确定 backbone 和分类头的结构
  2. 修改模型配置 YAML 中的 nc (类别数)
  3. 构建新模型，迁移 backbone 权重（100% 复用）
  4. 扩展分类头权重：旧类别直接拷贝，新类别用均值热启动
  5. 两阶段训练：冻结 backbone 训头 → 解冻全模型微调

"""

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from ultralytics import YOLO

from src.utils import CFG, CLASSES, log, SEED, set_seed

ROOT = Path(__file__).parent.parent


# ──── 工具函数 ────────────────────────────────────────────


def _infer_yaml(old_model_path: str) -> str:
    """从旧模型路径推断模型配置文件"""
    parent = Path(old_model_path).parent.parent.name  # e.g. "d3_yolo26n"
    match = re.search(r"(yolo\d+[nslmx])", parent)
    if not match:
        raise ValueError(f"无法从模型路径推断模型变体: {old_model_path}")

    variant = match.group(1)  # e.g. "yolo26n"
    candidates = [
        ROOT / "config" / f"{variant}-cls.yaml",
        ROOT / f"{variant}-cls.yaml",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)

    # 尝试匹配基类 YAML (yolo26n → yolo26)
    base_match = re.match(r"(yolo\d+)", variant)
    if base_match:
        base = base_match.group(1)  # e.g. "yolo26"
        # 检查是否有 yoloXXn-cls.yaml 自身
        for suffix in ["-cls.yaml", ".yaml"]:
            for loc in [ROOT / "config", ROOT]:
                cand = loc / f"{variant}{suffix}"
                if cand.exists():
                    return str(cand)

    raise FileNotFoundError(f"未找到 {variant} 的模型配置文件")


def _modify_yaml_nc(yaml_path: str, new_nc: int) -> str:
    """基于原 YAML 创建临时副本，仅修改 nc"""
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    old_nc = cfg.get("nc", 0)
    cfg["nc"] = new_nc

    tmp = tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False, encoding="utf-8"
    )
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    log.info("YAML nc: %d → %d  (临时: %s)", old_nc, new_nc, tmp.name)
    return tmp.name


def _find_classifier_keys(state_dict: Dict[str, torch.Tensor]):
    """在 state_dict 中定位分类头 linear 层"""
    linear_w = sorted([k for k in state_dict if "linear.weight" in k])
    linear_b = sorted([k for k in state_dict if "linear.bias" in k])

    if not linear_w:
        raise ValueError("未找到 linear.weight 层，请确认模型为分类模型")

    # 取最后一个 linear 层（即分类头）
    wk = linear_w[-1]
    bk = linear_b[-1] if linear_b else None
    log.debug("分类头: %s", wk)
    return wk, bk


def _transfer_weights(
    old_state: Dict[str, torch.Tensor],
    new_state: Dict[str, torch.Tensor],
    old_nc: int,
    new_nc: int,
):
    """迁移权重 + 分类头智能扩展"""
    w_key, b_key = _find_classifier_keys(old_state)
    old_w = old_state[w_key]  # [old_nc, feat_dim]
    old_b = old_state[b_key] if b_key else None  # [old_nc]

    # 找到新模型中名称匹配的对应层
    new_w_key = next((k for k in new_state if k.endswith("linear.weight")), None)
    new_b_key = next((k for k in new_state if k.endswith("linear.bias")), None)

    if not new_w_key:
        raise ValueError("新模型未找到 linear.weight 层")

    matched = 0
    skipped = 0

    for k, v in old_state.items():
        if k not in new_state:
            skipped += 1
            continue

        if new_state[k].shape == v.shape:
            # 同形状 → 直接复制（backbone / neck 权重）
            new_state[k] = v.clone()
            matched += 1
        elif k == w_key:
            # === 分类头 weight: [old_nc, D] → [new_nc, D] ===
            new_w = new_state[new_w_key]
            # 拷贝旧类别权重
            new_w[:old_nc, :] = v.clone()
            # 新类别用已有类别权重的均值热启动
            new_w[old_nc:, :] = v.mean(dim=0, keepdim=True)
            new_state[new_w_key] = new_w
            log.info(
                "分类头 weight: [%d,%d] → [%d,%d]  (新类别均值初始化)",
                old_w.shape[0],
                old_w.shape[1],
                new_w.shape[0],
                new_w.shape[1],
            )
        elif k == b_key and b_key and new_b_key:
            # === 分类头 bias: [old_nc] → [new_nc] ===
            new_b = new_state[new_b_key]
            new_b[:old_nc] = v.clone()
            new_b[old_nc:] = v.mean()
            new_state[new_b_key] = new_b
            log.info(
                "分类头 bias: [%d] → [%d]  (新类别均值初始化)",
                v.shape[0],
                new_b.shape[0],
            )
        else:
            skipped += 1

    log.info("权重迁移: 匹配 %d, 跳过 %d, 分类头已扩展", matched, skipped)
    return new_state


def _freeze_count(model: YOLO) -> int:
    """计算需要冻结的层数（冻结除分类头外的所有层）"""
    num_layers = len(model.model.model)  # nn.Sequential 的长度
    # 分类头是最后一层，所以冻结前 num_layers-1 层
    freeze_n = num_layers - 1
    log.info("冻结前 %d / %d 层，仅训练分类头", freeze_n, num_layers)
    return freeze_n


# ──── 主入口 ──────────────────────────────────────────────


def incremental_train(
    old_model: str,
    data: str,
    epochs_head: int = 30,
    epochs_fine: int = 10,
    device: Optional[str] = None,
    batch: Optional[int] = None,
    lr_head: float = 0.01,
    lr_fine: float = 0.0005,
    imgsz: Optional[int] = None,
    **kwargs,
) -> None:
    """
    增量训练 — 添加新分类而非从头开始

    工作流程:
      1. 修改 config/default.yaml 和 deploy/config.yaml，添加新类别
      2. 准备新类别数据到 data_ori/ 对应目录
      3. python main.py prepare <dataset>  (重建数据目录)
      4. python main.py incremental --old-model <旧模型.pt> --data <数据目录>

    Parameters
    ----------
    old_model : str
        已训练的 .pt 路径（如 runs/classify/d3_yolo26n/weights/best.pt）
    data : str
        扩类后的数据目录（须已运行 prepare，含新类别文件夹）
    epochs_head : int
        阶段 1: 冻结 backbone，仅训练分类头 (默认 30)
    epochs_fine : int
        阶段 2: 全模型微调 (默认 10, 设 0 跳过)
    lr_head : float
        阶段 1 学习率 (默认 0.01)
    lr_fine : float
        阶段 2 学习率 (默认 0.0005)
    """

    # ── 0. 参数处理 ──
    train_cfg = CFG.get("train", {})
    device = device or train_cfg["device"]
    batch = batch or train_cfg.get("batch", 32)
    imgsz = imgsz or train_cfg.get("imgsz", 384)

    new_nc = len(CLASSES)
    if new_nc < 2:
        raise ValueError(f"类别数过少: {new_nc}")

    # ── 1. 加载旧模型 ──
    old_path = Path(old_model)
    if not old_path.exists():
        # 尝试按训练名查找
        old_path = Path(CFG["train"]["save_dir"]) / old_model / "weights" / "best.pt"
    if not old_path.exists():
        raise FileNotFoundError(f"旧模型不存在: {old_model}")
    if not old_path.suffix:
        old_path = old_path / "weights" / "best.pt"

    log.info("加载旧模型: %s", old_path)
    old_yolo = YOLO(str(old_path))
    if not hasattr(old_yolo, "model") or old_yolo.model is None:
        raise RuntimeError("无法加载旧模型权重")

    old_state = old_yolo.model.state_dict()
    w_key, _ = _find_classifier_keys(old_state)
    old_nc = old_state[w_key].shape[0]
    log.info("旧模型类别数: %d", old_nc)

    if new_nc <= old_nc:
        log.warning("新类别数(%d) ≤ 旧类别数(%d)，不需要增量训练", new_nc, old_nc)
        return

    if new_nc > old_nc:
        log.info("扩展 %d → %d 类  (+%d)", old_nc, new_nc, new_nc - old_nc)

    # ── 2. 构建新模型 ──
    yaml_path = _infer_yaml(str(old_path))
    log.info("配置文件: %s", yaml_path)
    tmp_yaml = _modify_yaml_nc(yaml_path, new_nc)

    try:
        log.info("构建新模型 (nc=%d)...", new_nc)
        new_yolo = YOLO(tmp_yaml, task="classify")
    finally:
        try:
            os.unlink(tmp_yaml)
        except OSError:
            pass

    # ── 3. 迁移权重 ──
    new_state = new_yolo.model.state_dict()
    new_state = _transfer_weights(old_state, new_state, old_nc, new_nc)
    new_yolo.model.load_state_dict(new_state)

    # ── 4. 验证数据 ──
    data_path = Path(data)
    if not (data_path / "train").is_dir():
        raise FileNotFoundError(f"训练集不存在: {data_path / 'train'}")

    set_seed()

    # ── 5. 阶段 1: 训练分类头 ──
    log.info("=" * 50)
    log.info("阶段 1: 冻结 backbone，仅训练分类头 (%d epochs)", epochs_head)
    log.info("=" * 50)

    freeze_n = _freeze_count(new_yolo)
    new_yolo.train(
        data=data,
        epochs=epochs_head,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr_head,
        freeze=freeze_n,
        name=f"{Path(data).name}_incremental_s1",
        exist_ok=True,
        seed=SEED,
        pretrained=False,
        **{
            k: v
            for k, v in train_cfg.items()
            if k
            in (
                "patience",
                "workers",
                "cos_lr",
                "amp",
                "weight_decay",
                "dropout",
                "label_smoothing",
                "optimizer",
                "save_period",
            )
        },
    )

    if epochs_fine <= 0:
        log.info("跳过阶段 2 (epochs_fine=%d)", epochs_fine)
        return

    # ── 6. 阶段 2: 全模型微调 ──
    log.info("=" * 50)
    log.info("阶段 2: 全模型微调 (%d epochs)", epochs_fine)
    log.info("=" * 50)

    new_yolo.train(
        data=data,
        epochs=epochs_fine,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr_fine,
        freeze=None,  # 解冻所有层
        name=f"{Path(data).name}_incremental_s2",
        exist_ok=True,
        seed=SEED,
        pretrained=False,
        **{
            k: v
            for k, v in train_cfg.items()
            if k
            in (
                "patience",
                "workers",
                "cos_lr",
                "amp",
                "weight_decay",
                "dropout",
                "label_smoothing",
                "optimizer",
                "save_period",
            )
        },
    )

    log.info("增量训练完成!")
    log.info(
        "导出: python main.py export --model runs/classify/%s_incremental_s2/weights/best.pt",
        Path(data).name,
    )
