import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent

# 日志配置
log = logging.getLogger("fabric")
log.setLevel(logging.DEBUG)
_log_ready = False


def _setup_log() -> None:
    """延迟初始化日志系统"""
    global _log_ready
    if _log_ready:
        return
    _log_ready = True
    
    log_dir = ROOT / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 文件日志
    fh = logging.FileHandler(log_dir / "fabric.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    # 控制台日志
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    
    log.addHandler(fh)
    log.addHandler(ch)


# 包装日志方法实现延迟初始化
_orig = {k: getattr(log, k) for k in ["debug", "info", "warning", "error"]}
for level in _orig:
    setattr(log, level, lambda msg, *a, lv=level, **kw: (_setup_log(), _orig[lv](msg, *a, **kw)))


# 加载配置
config_path = Path(os.environ.get("FABRIC_CONFIG", str(ROOT / "config" / "default.yaml")))
with open(config_path, encoding="utf-8") as f:
    CFG: Dict[str, Any] = yaml.safe_load(f)

CLASSES: List[str] = sorted(CFG["classes"])
CLASS_NAMES: Dict[str, str] = CFG["class_names"]
SEED: int = CFG["seed"]

if CLASSES != CFG["classes"]:
    log.warning("classes 已按字母序重排")


class ModelNotFoundError(FileNotFoundError):
    """模型文件未找到"""


class DatasetNotFoundError(FileNotFoundError):
    """数据集配置未找到"""


def set_seed(seed: Optional[int] = None) -> None:
    """设置全局随机种子"""
    seed_value = seed or SEED
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import torch
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    log.debug("随机种子: %d", seed_value)


def load_dataset(name: str) -> Dict[str, Any]:
    """加载数据集配置"""
    path = ROOT / "config" / "datasets" / f"{name}.yaml"
    if not path.exists():
        raise DatasetNotFoundError(f"数据集配置不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model(path: str, fatal: bool = True) -> str:
    """解析模型路径，支持自动下载"""
    pretrained_dir = Path(CFG["train"]["pretrained_dir"])
    model_path = Path(path)

    # 尝试在预训练目录查找
    if os.path.sep not in path and not model_path.exists():
        candidate = pretrained_dir / path
        if candidate.exists():
            model_path = candidate

    # 自动下载
    if not model_path.exists():
        log.info("自动下载: %s", model_path.name)
        try:
            from ultralytics import YOLO
            YOLO(str(model_path))
            # 移动到缓存
            if model_path.exists():
                pretrained_dir.mkdir(parents=True, exist_ok=True)
                dst = pretrained_dir / model_path.name
                if not dst.exists():
                    import shutil
                    shutil.move(str(model_path), str(dst))
                model_path = dst
                log.info("已缓存: %s", model_path)
        except Exception as exc:
            if fatal:
                raise ModelNotFoundError(f"下载失败: {path}") from exc
            log.error("下载失败: %s", path)
            return str(model_path)

    if not model_path.exists() and fatal:
        raise ModelNotFoundError(f"模型不存在: {path}")
    
    return str(model_path)


def filter_probs(
    indices: List[int], 
    confs: List[float], 
    top: Optional[int] = None, 
    threshold: float = 0.0
) -> List[Tuple[str, float]]:
    """过滤和格式化预测概率"""
    results = [
        (CLASS_NAMES.get(CLASSES[int(i)], CLASSES[int(i)]) if int(i) < len(CLASSES) else f"class_{int(i)}", 
         float(c))
        for i, c in zip(indices, confs)
    ]
    
    results = results[:top] if top else results
    
    if threshold > 0 and results and results[0][1] < threshold:
        return [(CFG["predict"]["unknown_label"], results[0][1])]
    
    return results
