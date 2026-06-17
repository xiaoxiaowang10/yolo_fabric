import re
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from src.constants import TRAIN_KEYS
from src.utils import CFG, CLASSES, log, resolve_model, SEED, set_seed

def _resolve_spec(spec: Optional[str], default: str) -> str:
    """解析模型名为完整的 .pt 文件名"""
    if spec is None:
        return default
    spec = spec.strip()
    if spec.endswith(".pt"):
        return spec
    if "-cls" in spec:
        return f"{spec}.pt"
    if re.match(r"^yolov?\d+", spec):
        return f"{spec}-cls.pt"
    return f"{spec}.pt"


def _detect_yolo_version(name_short: str) -> Optional[str]:
    """从模型名识别 YOLO 版本 (yolo26n → yolo26, yolov8n → yolov8)"""
    m = re.match(r"(yolov?\d+)", name_short)
    return m.group(1) if m else None


def _build_kwargs(
    train_cfg: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """构建训练参数"""
    kwargs = {k: train_cfg[k] for k in TRAIN_KEYS & train_cfg.keys()}
    kwargs["lr0"] = overrides.get("lr") or train_cfg["lr"]
    kwargs["project"] = train_cfg["project"]
    for k in ("epochs", "imgsz", "batch", "device", "workers", "patience"):
        if overrides.get(k) is not None:
            kwargs[k] = overrides[k]
    return kwargs


def _save_best(train_cfg: Dict[str, Any], train_name: str, dataset: str) -> None:
    """保存并验证最佳模型"""
    root = Path(train_cfg["save_dir"]) / train_cfg["project"] / train_name
    best_pt = root / "weights" / "best.pt"

    if best_pt.exists():
        model = YOLO(str(best_pt))
        metrics = model.val(data=dataset, split=CFG["prepare"]["val_subdir"])
        log.info("验证 Top-1: %.2f%%  Top-5: %.2f%%", metrics.top1, metrics.top5)
    log.info("导出: python main.py export --model %s", train_name)


def train(
    data: str,
    model: Optional[str] = None,
    epochs: Optional[int] = None,
    imgsz: Optional[int] = None,
    batch: Optional[int] = None,
    lr: Optional[float] = None,
    device: Optional[str] = None,
    workers: Optional[int] = None,
    patience: Optional[int] = None,
    resume: bool = False,
    train_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """训练新模型"""
    if not train_cfg:
        raise ValueError("train_cfg required")

    model_name = _resolve_spec(model, train_cfg["pretrained"])
    model_path = resolve_model(model_name)
    train_dir = Path(data) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"训练集不存在: {train_dir}")

    (Path(data) / CFG["prepare"]["val_subdir"]).mkdir(parents=True, exist_ok=True)
    set_seed()

    dataset = Path(data).name
    name_short = Path(model_path).stem.replace("-cls", "")
    train_name = f"{dataset}_{name_short}"
    log.info("类别: %d | 模型: %s | 保存: %s", len(CLASSES), model_path, train_name)

    # 识别 YOLO 版本并解析 YAML 配置
    import shutil

    yaml_scaled = Path(f"config/{name_short}-cls.yaml")
    if not yaml_scaled.exists():
        version = _detect_yolo_version(name_short)
        if version:
            yaml_base = Path(f"config/{version}-cls.yaml")
            if yaml_base.exists():
                log.info("使用 %s 配置模板 → %s", yaml_base.name, yaml_scaled.name)
                shutil.copy2(yaml_base, yaml_scaled)

    kwargs = _build_kwargs(train_cfg, locals())

    if yaml_scaled.exists():
        log.info("加载 YAML 配置: %s + 权重: %s", yaml_scaled, model_path.name)
        model = YOLO(str(yaml_scaled)).load(model_path)
    else:
        log.info("无本地 YAML 配置，从权重加载: %s (版本: %s)", model_path.name, _detect_yolo_version(name_short) or "?")
        model = YOLO(str(model_path))

    model.train(
        data=data,
        name=train_name,
        seed=SEED,
        exist_ok=True,
        save=True,
        resume=resume,
        split=CFG["prepare"]["val_subdir"],
        **kwargs,
    )

    _save_best(train_cfg, train_name, data)



