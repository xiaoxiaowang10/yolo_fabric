import shutil
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from src.utils import CFG, log


def export_onnx(name: str, imgsz: Optional[int] = None, half: bool = False) -> Optional[str]:
    """导出模型为 ONNX 格式"""
    imgsz = imgsz or CFG["export"]["imgsz"]
    model_path = Path("runs") / "classify" / name / "weights" / "best.pt"
    
    if not model_path.exists():
        log.error("权重不存在: %s", model_path)
        return None

    precision = "fp16" if half else "fp32"
    log.info("导出 %s  imgsz=%d  %s", name, imgsz, precision)
    YOLO(str(model_path)).export(format="onnx", imgsz=imgsz, half=half)

    src = model_path.with_suffix(".onnx")
    if not src.exists():
        log.warning("导出文件未生成: %s", src)
        return None

    out_dir = Path(CFG["export"]["onnx_output_dir"])
    dst = out_dir / f"{name}_imgsz{imgsz}_{precision}.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    log.info("  -> %s", dst.absolute())
    return str(dst)
