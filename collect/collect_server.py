"""数据采集服务 — 前端摄像头拍照 + 分类保存 (独立运行)"""

import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

# 本目录
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def _load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {CONFIG_PATH}\n"
            "请复制 config.yaml 并修改类别信息"
        )
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = _load_config()
CLASSES = CFG["classes"]
CLASS_NAMES = CFG["class_names"]
DATASET = CFG.get("dataset", "yueli")
DATA_DIR = BASE_DIR / CFG.get("data_dir", "data_ori")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html",
                           classes=CLASSES,
                           class_names=CLASS_NAMES,
                           dataset=DATASET)


@app.route("/api/stats")
def stats():
    """各类别已采集图片数量"""
    dataset_dir = DATA_DIR / DATASET
    result = {}
    total = 0
    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        count = len([f for f in cls_dir.iterdir() if f.is_file()]) if cls_dir.is_dir() else 0
        result[cls] = count
        total += count
    return jsonify({"classes": result, "total": total, "dataset": DATASET})


@app.route("/api/capture", methods=["POST"])
def capture():
    """保存拍照图片到对应类别目录"""
    cls = request.form.get("class", "")
    if cls not in CLASSES:
        return jsonify({"error": f"无效类别: {cls}"}), 400

    if "image" not in request.files:
        return jsonify({"error": "未收到图片"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "空文件"}), 400

    cls_dir = DATA_DIR / DATASET / cls
    cls_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"COLLECT_{timestamp}.jpg"
    filepath = cls_dir / filename

    img = Image.open(file.stream)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(filepath, "JPEG", quality=95)

    return jsonify({
        "success": True,
        "filename": filename,
        "class": cls,
        "class_name": CLASS_NAMES.get(cls, cls),
    })


@app.route("/api/delete", methods=["POST"])
def delete_image():
    """删除指定图片"""
    cls = request.form.get("class", "")
    filename = request.form.get("filename", "")
    if cls not in CLASSES or not filename:
        return jsonify({"error": "参数错误"}), 400

    if not filename.startswith("COLLECT_"):
        return jsonify({"error": "只能删除采集的图片"}), 400

    filepath = DATA_DIR / DATASET / cls / filename
    if filepath.exists():
        filepath.unlink()
        return jsonify({"success": True})
    return jsonify({"error": "文件不存在"}), 404


@app.route("/api/recent")
def recent_images():
    """获取各类别最近的图片"""
    cls = request.args.get("class", "")
    limit = int(request.args.get("limit", 20))
    dataset_dir = DATA_DIR / DATASET

    if cls and cls in CLASSES:
        classes_to_check = [cls]
    else:
        classes_to_check = CLASSES

    result = {}
    for c in classes_to_check:
        cls_dir = dataset_dir / c
        if not cls_dir.is_dir():
            result[c] = []
            continue
        files = sorted(
            [f for f in cls_dir.iterdir() if f.is_file()],
            key=lambda f: f.stat().st_mtime, reverse=True,
        )
        result[c] = [
            {"filename": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
            for f in files[:limit]
        ]
    return jsonify(result)


@app.route("/api/image/<cls>/<filename>")
def serve_image(cls, filename):
    """提供图片访问"""
    if cls not in CLASSES:
        return jsonify({"error": "无效类别"}), 400
    cls_dir = DATA_DIR / DATASET / cls
    if not cls_dir.is_dir():
        return jsonify({"error": "目录不存在"}), 404
    return send_from_directory(cls_dir, filename)


@app.route("/api/list")
def list_images():
    """列出指定类别所有图片, 支持分页"""
    cls = request.args.get("class", "")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 30))
    if cls not in CLASSES:
        return jsonify({"error": "无效类别"}), 400

    cls_dir = DATA_DIR / DATASET / cls
    if not cls_dir.is_dir():
        return jsonify({"files": [], "total": 0, "pages": 0})

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    all_files = sorted(
        [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in exts],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    total = len(all_files)
    pages = (total + per_page - 1) // per_page
    start = (page - 1) * per_page
    files = all_files[start:start + per_page]

    return jsonify({
        "files": [
            {"filename": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
            for f in files
        ],
        "total": total,
        "pages": pages,
        "page": page,
    })


@app.route("/api/move", methods=["POST"])
def move_image():
    """将图片移动到另一个类别目录 (归类)"""
    src_cls = request.form.get("src_class", "")
    dst_cls = request.form.get("dst_class", "")
    filename = request.form.get("filename", "")
    if src_cls not in CLASSES or dst_cls not in CLASSES or not filename:
        return jsonify({"error": "参数错误"}), 400
    if src_cls == dst_cls:
        return jsonify({"error": "源类别与目标类别相同"}), 400

    src_path = DATA_DIR / DATASET / src_cls / filename
    if not src_path.exists():
        return jsonify({"error": "文件不存在"}), 404

    dst_dir = DATA_DIR / DATASET / dst_cls
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / filename

    if dst_path.exists():
        stem = dst_path.stem
        suffix = dst_path.suffix
        i = 1
        while dst_path.exists():
            dst_path = dst_dir / f"{stem}_{i}{suffix}"
            i += 1

    shutil.move(str(src_path), str(dst_path))

    return jsonify({
        "success": True,
        "src_class": src_cls,
        "dst_class": dst_cls,
        "src_filename": filename,
        "dst_filename": dst_path.name,
        "dst_class_name": CLASS_NAMES.get(dst_cls, dst_cls),
    })


if __name__ == "__main__":
    port = int(os.environ.get("COLLECT_PORT", CFG.get("port", 5000)))
    print(f"数据采集服务: http://localhost:{port}")
    print(f"数据集: {DATASET}")
    print(f"类别: {', '.join(CLASS_NAMES[c] for c in CLASSES)}")
    print(f"保存路径: {DATA_DIR / DATASET}")
    app.run(host="0.0.0.0", port=port, debug=False)
