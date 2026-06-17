import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.utils import log


def export_onnx_arm(name: str, imgsz: int = None) -> list:
    """导出 NCNN 格式模型 → deploy_arm_v2/models/"""
    if imgsz is None:
        cfg_path = Path("deploy_arm_v2/config.yaml")
        imgsz = yaml.safe_load(cfg_path.open(encoding="utf-8")).get("imgsz", 384)

    model_path = Path("runs/classify") / name / "weights" / "best.pt"
    if not model_path.exists():
        log.error("权重不存在: %s", model_path)
        return []

    dst = Path("deploy_arm_v2/models") / f"{name}_imgsz{imgsz}_ncnn_model"
    if dst.exists():
        shutil.rmtree(dst)

    log.info("导出 %s  →  %s", name, dst.name)
    out = YOLO(str(model_path)).export(format="ncnn", imgsz=imgsz)
    shutil.copytree(out, dst)
    shutil.rmtree(out)

    from src.utils import CLASSES
    names = {str(i): c for i, c in enumerate(sorted(CLASSES))}
    (dst / "metadata.yaml").write_text(
        yaml.safe_dump({"names": names, "task": "classify",
                        "imgsz": imgsz, "model": name},
                       allow_unicode=True, sort_keys=False))

    mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1_048_576
    log.info("  完成  %.1f MB", mb)
    return [str(dst)]
