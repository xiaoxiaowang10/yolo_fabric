from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO

from src.constants import VIDEO_EXTS, IMAGE_EXTS
from src.utils import filter_probs, log, resolve_model, set_seed


def _classify_image(
    model: YOLO, image_path: str, imgsz: int, device: str, top: int, threshold: float
) -> Optional[List[Tuple[str, float]]]:
    """对单张图片分类"""
    for result in model.predict(image_path, imgsz=imgsz, device=device, verbose=False):
        if result.probs is not None:
            return filter_probs(result.probs.top5, result.probs.top5conf, top, threshold)
    return None


def predict(
    input_path: str,
    model: Optional[str] = None,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    top: int = 3,
    threshold: float = 0.0,
    show: bool = False,
    save: bool = False,
    predict_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """执行模型推理"""
    if not predict_cfg:
        raise ValueError("predict_cfg required")
    
    set_seed()
    yolo_model = YOLO(resolve_model(model or predict_cfg["model"]))
    imgsz = imgsz or predict_cfg["imgsz"]
    device = device or predict_cfg["device"]
    input_p = Path(input_path)
    
    if input_p.is_file():
        ext = input_p.suffix.lower()
        
        # 视频
        if ext in VIDEO_EXTS:
            results = yolo_model.predict(
                input_path, imgsz=imgsz, device=device, stream=True,
                show=show, save=save, project=predict_cfg["video_save_dir"], verbose=False
            )
            for idx, result in enumerate(results, 1):
                if result.probs:
                    probs = filter_probs(result.probs.top5, result.probs.top5conf, 1, threshold)
                    name = probs[0][0] if probs else "—"
                    log.info("  #%05d  %-20s  %.3f", idx, name, float(result.probs.top1conf))
        # 图片
        else:
            result = _classify_image(yolo_model, input_path, imgsz, device, top, threshold)
            if result:
                log.info("\n图片: %s", input_p.name)
                for name, conf in result:
                    log.info("  %-20s  %.4f", name, conf)
    
    elif input_p.is_dir():
        # 目录
        images = sorted(str(p) for p in input_p.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            log.error("目录中无图片: %s", input_path)
            return
        
        log.info("共 %d 张", len(images))
        for path in images:
            result = _classify_image(yolo_model, path, imgsz, device, top, threshold)
            if result:
                log.info("\n图片: %s", Path(path).name)
                for name, conf in result:
                    log.info("  %-20s  %.4f", name, conf)
    else:
        log.error("路径不存在: %s", input_path)
