from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from src.constants import TRAIN_KEYS
from src.utils import CFG, CLASSES, log, resolve_model, SEED, set_seed


def _build_kwargs(train_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
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
        metrics = YOLO(str(best_pt))(val=True, data=dataset, split=CFG["prepare"]["val_subdir"])
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
    
    model_path = resolve_model(model or train_cfg["pretrained"])
    train_dir = Path(data) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"训练集不存在: {train_dir}")

    (Path(data) / CFG["prepare"]["val_subdir"]).mkdir(parents=True, exist_ok=True)
    set_seed()

    dataset = Path(data).name
    name_short = Path(model_path).stem.replace("-cls", "")
    train_name = f"{dataset}_{name_short}"
    log.info("类别: %d | 模型: %s | 保存: %s", len(CLASSES), model_path, train_name)

    # 准备配置
    import re, shutil
    yaml_scaled = Path(f"config/{name_short}-cls.yaml")
    if not yaml_scaled.exists():
        match = re.match(r"(yolo\d+)", name_short)
        yaml_base = Path(f"config/{match.group(1) if match else 'yolo26'}-cls.yaml")
        if not yaml_base.exists():
            raise FileNotFoundError(f"YAML 模板不存在: {yaml_base}")
        shutil.copy2(yaml_base, yaml_scaled)

    kwargs = _build_kwargs(train_cfg, locals())

    YOLO(str(yaml_scaled)).load(model_path).train(
        data=data, name=train_name, seed=SEED, exist_ok=True,
        pretrained=True, save=True, resume=resume,
        split=CFG["prepare"]["val_subdir"], **kwargs
    )

    _save_best(train_cfg, train_name, data)


def continue_train(
    model: str,
    data: str,
    epochs: Optional[int] = None,
    imgsz: Optional[int] = None,
    batch: Optional[int] = None,
    lr: Optional[float] = None,
    device: Optional[str] = None,
    workers: Optional[int] = None,
    patience: Optional[int] = None,
    train_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """继续训练已有模型"""
    if not train_cfg:
        raise ValueError("train_cfg required")
    
    model_path = Path(model)
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model}")

    train_dir = Path(data) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"训练集不存在: {train_dir}")

    (Path(data) / CFG["prepare"]["val_subdir"]).mkdir(parents=True, exist_ok=True)
    set_seed()

    dataset = Path(data).name
    train_name = f"{dataset}_{model_path.stem}"
    log.info("加载: %s | 类别: %d | 保存: %s", model_path.absolute(), len(CLASSES), train_name)

    kwargs = _build_kwargs(train_cfg, locals())

    YOLO(str(model)).train(
        data=data, name=train_name, seed=SEED, exist_ok=True,
        save=True, split=CFG["prepare"]["val_subdir"], **kwargs
    )

    _save_best(train_cfg, train_name, data)
