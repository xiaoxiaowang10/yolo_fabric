"""数据准备模块"""
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from PIL import Image
from tqdm import tqdm

from src.utils import CFG, CLASSES, log, load_dataset, SEED, set_seed


@dataclass
class ImageInfo:
    category: str
    filename: str
    source_path: Path


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _collect(targets: List[str], src_dir: Path, image_exts: Set[str]) -> List[ImageInfo]:
    images = []
    for cat in targets:
        cat_dir = src_dir / cat
        if not cat_dir.is_dir():
            continue
        for fp in cat_dir.iterdir():
            if fp.suffix.lower() in image_exts:
                images.append(ImageInfo(cat, fp.name, fp))
    return images


def _copy_split(images: List[ImageInfo], dst_dir: Path, split: str, cat: str) -> None:
    class_dir = dst_dir / split / cat
    class_dir.mkdir(parents=True, exist_ok=True)
    for img in tqdm(images, desc=f"  {split}/{cat}", unit="张", leave=False):
        shutil.copy2(img.source_path, class_dir / img.filename)


def run(dataset: str = "yueli") -> None:
    dataset_cfg = load_dataset(dataset)
    prepare_cfg = CFG["prepare"]

    src_dir = Path(dataset_cfg["raw_data"])
    dst_dir = Path(dataset_cfg["output_dir"])
    val_split = prepare_cfg["val_split"]
    train_sub, val_sub = prepare_cfg["train_subdir"], prepare_cfg["val_subdir"]
    image_exts = set(prepare_cfg["image_exts"])
    ds_classes = dataset_cfg["classes"]

    unknown = set(ds_classes) - set(CLASSES)
    if unknown:
        log.error("数据集 %s 含未知类别: %s", dataset, unknown)
        return

    log.info("数据集 %s -> %d 类: %s", dataset, len(ds_classes), ", ".join(ds_classes))
    set_seed()

    if not src_dir.is_dir():
        log.error("原始目录不存在: %s", src_dir)
        return

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    (dst_dir / train_sub).mkdir(parents=True)
    (dst_dir / val_sub).mkdir(parents=True)
    log.info("训练 %.0f%% / 验证 %.0f%%  (seed=%d)", (1 - val_split) * 100, val_split * 100, SEED)

    total_train = total_val = 0
    for cat in tqdm(ds_classes, desc="数据准备", unit="类"):
        images = _collect([cat], src_dir, image_exts)
        if not images:
            continue

        valid = [img for img in tqdm(images, desc=f"校验 {cat}", unit="张", leave=False) if _valid_image(img.source_path)]
        if not valid:
            continue

        random.shuffle(valid)
        split_point = max(1, int(len(valid) * (1 - val_split)))
        train_imgs, val_imgs = valid[:split_point], valid[split_point:]

        _copy_split(train_imgs, dst_dir, train_sub, cat)
        _copy_split(val_imgs, dst_dir, val_sub, cat)

        n_train, n_val = len(train_imgs), len(val_imgs)
        log.info("  %-22s  train: %4d  val: %3d", cat, n_train, n_val)
        total_train += n_train
        total_val += n_val

    for cls in CLASSES:
        if cls not in ds_classes:
            (dst_dir / train_sub / cls).mkdir(parents=True, exist_ok=True)
            (dst_dir / val_sub / cls).mkdir(parents=True, exist_ok=True)

    log.info("完成: 训练 %d / 验证 %d / 总计 %d", total_train, total_val, total_train + total_val)
    log.info("目录: %s", dst_dir.absolute())
