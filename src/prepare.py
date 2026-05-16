"""数据准备模块"""
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

from src.constants import IMAGE_EXTS
from src.utils import CFG, CLASSES, log, load_dataset, SEED, set_seed


@dataclass
class ImageInfo:
    """图片信息"""
    category: str
    subfolder: str
    filename: str
    source_path: Path


def _valid_image(path: Path) -> bool:
    """验证图片是否有效"""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _collect_folder(targets: List[str], src_dir: Path, image_exts: Set[str]) -> List[ImageInfo]:
    """从文件夹结构收集图片"""
    images = []
    for cat in targets:
        cat_dir = src_dir / cat
        if not cat_dir.is_dir():
            continue
        for sub_dir in sorted(cat_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            for file_path in sub_dir.iterdir():
                if file_path.suffix.lower() in image_exts:
                    images.append(ImageInfo(cat, sub_dir.name, file_path.name, file_path))
    return images


def _load_csv_mapping(csv_path: Path) -> Dict[str, str]:
    """加载 Roboflow CSV 标签映射"""
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        classes = next(reader)[1:]
        for row in reader:
            if row:
                for col_idx, value in enumerate(row[1:]):
                    if value == "1":
                        mapping[row[0]] = classes[col_idx]
                        break
    return mapping


def _collect_roboflow(targets: List[str], src_dir: Path, image_exts: Set[str]) -> List[ImageInfo]:
    """从 Roboflow 格式收集图片"""
    target_set = set(targets)
    images = []
    for split in ["train", "valid"]:
        split_dir = src_dir / split
        csv_path = split_dir / "_classes.csv"
        if not split_dir.is_dir() or not csv_path.exists():
            continue
        label_map = _load_csv_mapping(csv_path)
        for file_path in split_dir.iterdir():
            if file_path.suffix.lower() in image_exts and file_path.name in label_map:
                cat = label_map[file_path.name]
                if cat in target_set:
                    images.append(ImageInfo(cat, "", file_path.name, file_path))
    return images


def _collect_yolo(targets: List[str], src_dir: Path, image_exts: Set[str]) -> List[ImageInfo]:
    """从 YOLO 检测格式收集图片"""
    yaml_path = src_dir / "data.yaml"
    if not yaml_path.exists():
        return []
    with open(yaml_path, encoding="utf-8") as f:
        class_names = yaml.safe_load(f).get("names", [])
    if not class_names:
        return []
    
    target_set = set(targets)
    images = []
    for split in ["train", "valid", "test"]:
        image_dir = src_dir / split / "images"
        label_dir = src_dir / split / "labels"
        if not image_dir.is_dir():
            continue
        for file_path in image_dir.iterdir():
            if file_path.suffix.lower() not in image_exts:
                continue
            label_path = label_dir / f"{file_path.stem}.txt"
            class_id = None
            if label_path.exists():
                with open(label_path) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            break
            if class_id is not None and class_id < len(class_names):
                cat = class_names[class_id]
                if cat in target_set:
                    images.append(ImageInfo(cat, "", file_path.name, file_path))
    return images


def _copy_split(images: List[ImageInfo], dst_dir: Path, split: str, cat: str, is_merged: bool) -> None:
    """复制图片到目标目录"""
    class_dir = dst_dir / split / cat
    class_dir.mkdir(parents=True, exist_ok=True)
    for img in tqdm(images, desc=f"  {split}/{cat}", unit="张", leave=False):
        if is_merged:
            dest_name = f"{img.category}_{img.subfolder}_{img.filename}"
        elif img.subfolder:
            dest_name = f"{img.subfolder}_{img.filename}"
        else:
            dest_name = img.filename
        shutil.copy2(img.source_path, class_dir / dest_name)


def _detect_categories(src_dir: Path, merge_map: Dict[str, str], fmt: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """检测数据集中的类别"""
    # 自动检测格式
    if fmt != "yolo_detection" and (src_dir / "train" / "_classes.csv").exists():
        fmt = "roboflow"
    
    # 获取所有源类别
    if fmt == "yolo_detection":
        yaml_path = src_dir / "data.yaml"
        all_sources = sorted(yaml.safe_load(open(yaml_path)).get("names", []) if yaml_path.exists() else [])
    elif fmt == "roboflow":
        labels = set()
        for split in ["train", "valid"]:
            csv_path = src_dir / split / "_classes.csv"
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as f:
                    labels.update(next(csv.reader(f))[1:])
        all_sources = sorted(labels)
    else:
        all_sources = sorted(e.name for e in src_dir.iterdir() if e.is_dir())

    # 分组
    groups, order = {}, []
    for src_cat in all_sources:
        tgt_cat = merge_map.get(src_cat, src_cat)
        if tgt_cat not in groups:
            groups[tgt_cat] = []
            order.append(tgt_cat)
        groups[tgt_cat].append(src_cat)
    return groups, order


def run(dataset: str = "ibug") -> None:
    """执行数据准备流程"""
    dataset_cfg = load_dataset(dataset)
    prepare_cfg = CFG["prepare"]

    # 配置
    src_dir = Path(dataset_cfg["raw_data"])
    dst_dir = Path(dataset_cfg["output_dir"])
    val_split = prepare_cfg["val_split"]
    train_sub, val_sub = prepare_cfg["train_subdir"], prepare_cfg["val_subdir"]
    image_exts = set(prepare_cfg["image_exts"])
    merge_rules = dataset_cfg["merge"]
    excluded = set(dataset_cfg["exclude"])
    limits = dataset_cfg["limits"]
    ds_classes = dataset_cfg["classes"]
    source_format = dataset_cfg["source_format"]
    balance = dataset_cfg["balance"]

    # 验证
    unknown = set(ds_classes) - set(CLASSES)
    if unknown:
        log.error("数据集 %s 含未知类别: %s", dataset, unknown)
        return

    log.info("数据集 %s -> %d 类: %s", dataset, len(ds_classes), ", ".join(ds_classes))
    set_seed()

    if not src_dir.is_dir():
        log.error("原始目录不存在: %s", src_dir)
        return

    # 清理并创建目标目录
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    (dst_dir / train_sub).mkdir(parents=True)
    (dst_dir / val_sub).mkdir(parents=True)
    log.info("训练 %.0f%% / 验证 %.0f%%  (seed=%d)", (1 - val_split) * 100, val_split * 100, SEED)

    # 构建合并映射
    merge_map = {src: tgt for tgt, srcs in merge_rules.items() for src in srcs}
    groups, order = _detect_categories(src_dir, merge_map, source_format)

    # 收集函数
    collectors = {"yolo_detection": _collect_yolo, "roboflow": _collect_roboflow}
    collect_fn = collectors.get(source_format, _collect_folder)

    # 处理每个类别
    total_train = total_val = total_skipped = 0
    stats = []

    for target in tqdm(order, desc="数据准备", unit="类"):
        if target not in ds_classes or target in excluded:
            continue
        
        sources = groups[target]
        is_merged = target in merge_rules

        # 收集图片
        images = collect_fn(sources, src_dir, image_exts)
        if not images:
            continue

        # 验证
        valid = [img for img in tqdm(images, desc=f"校验 {target}", unit="张", leave=False) if _valid_image(img.source_path)]
        skipped = len(images) - len(valid)
        total_skipped += skipped
        if not valid:
            continue

        # 限制数量
        if target in limits and len(valid) > limits[target]:
            random.shuffle(valid)
            valid = valid[:limits[target]]

        # 分割
        random.shuffle(valid)
        split_point = max(1, int(len(valid) * (1 - val_split)))
        train_imgs, val_imgs = valid[:split_point], valid[split_point:]

        # 复制
        _copy_split(train_imgs, dst_dir, train_sub, target, is_merged)
        _copy_split(val_imgs, dst_dir, val_sub, target, is_merged)

        n_train, n_val = len(train_imgs), len(val_imgs)
        merge_info = " <- " + " + ".join(sources) if is_merged else ""
        log.info("  %-22s  train: %4d  val: %3d  %s", target, n_train, n_val, merge_info)
        
        total_train += n_train
        total_val += n_val
        stats.append((target, n_train, n_val))

    # 均衡化
    if balance > 0:
        cap_train = max(1, int(balance * (1 - val_split)))
        cap_val = max(1, int(balance * val_split))
        log.info("均衡化: 每类 %d (train=%d / val=%d)", balance, cap_train, cap_val)
        total_train = total_val = 0
        new_stats = []
        for target, _, _ in stats:
            tr, va = cap_train, cap_val
            for sub, cap in [(train_sub, tr), (val_sub, va)]:
                class_dir = dst_dir / sub / target
                if not class_dir.is_dir():
                    continue
                files = sorted(f.name for f in class_dir.iterdir())
                if len(files) <= cap:
                    if sub == train_sub:
                        tr = len(files)
                    else:
                        va = len(files)
                    continue
                for fn in random.sample(files, len(files) - cap):
                    (class_dir / fn).unlink()
            new_stats.append((target, tr, va))
            total_train += tr
            total_val += va
        stats = new_stats

    # 创建空类别目录
    for cls in CLASSES:
        if cls not in ds_classes:
            (dst_dir / train_sub / cls).mkdir(parents=True, exist_ok=True)
            (dst_dir / val_sub / cls).mkdir(parents=True, exist_ok=True)

    # 统计
    if total_skipped:
        log.warning("跳过 %d 张损坏图片", total_skipped)
    log.info("完成: 训练 %d / 验证 %d / 总计 %d", total_train, total_val, total_train + total_val)
    log.info("目录: %s", dst_dir.absolute())
