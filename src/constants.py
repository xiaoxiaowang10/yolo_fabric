"""
常量定义模块
"""
from typing import Set

# 支持的图片格式
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# 支持的视频格式
VIDEO_EXTS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# 训练参数键集合
TRAIN_KEYS: Set[str] = {
    "epochs",
    "imgsz",
    "batch",
    "patience",
    "device",
    "workers",
    "save_period",
    "optimizer",
    "cos_lr",
    "warmup_epochs",
    "weight_decay",
    "dropout",
    "label_smoothing",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "flipud",
    "fliplr",
    "erasing",
    "amp",
    "rect",
}
