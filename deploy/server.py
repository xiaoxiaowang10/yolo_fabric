"""Flask Web 服务 —— 图片识别 / 模型对比 API"""

import io
from flask import Flask, request, jsonify, render_template
from PIL import Image
from inference import CFG, registry, CLASS_NAMES, CLASSES


# ═══════════════════ 初始化 ═══════════════════

_PORT = CFG["port"]

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

models_list = registry.list_models()
default_model = models_list[0] if models_list else None
print(f"Models ({len(models_list)}): {models_list or 'none'}")
print(f"Default: {default_model or 'none'}")


# ═══════════════════ 工具函数 ═══════════════════


def _read_image(file_storage):
    try:
        return Image.open(io.BytesIO(file_storage.read()))
    except Exception as e:
        raise ValueError(
            f"图片解析失败 ({e}). 请使用 JPG/PNG 格式, iPhone 请关闭 HEIC 拍照"
        )


def _parse_models(param):
    """ "a,b,c" → ["a", "b", "c"]"""
    return [m.strip() for m in param.split(",") if m.strip()]


def _to_dict(predictions, precision=4):
    """[(name, conf), ...] → [{"name":..., "confidence":...}, ...]"""
    return [{"name": p[0], "confidence": round(p[1], precision)} for p in predictions]


# ═══════════════════ 页面 ═══════════════════


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=models_list,
        default=default_model,
        class_count=len(CLASSES),
        webcam_interval_ms=CFG.get("webcam_interval_ms", 1500),
    )


# ═══════════════════ API ═══════════════════


@app.route("/api/classify", methods=["POST"])
def classify():
    """POST /api/classify?model=xxx + image 文件 → 单模型识别"""
    name = request.args.get("model", default_model)
    if not name:
        return jsonify({"error": "no model available"}), 400

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image"}), 400

    try:
        result = registry.classify(name, _read_image(file))
        if result is None:
            return jsonify({"error": f"model not found: {name}"}), 404
        return jsonify(
            {
                "results": _to_dict(result["predictions"]),
                "time_ms": result["time_ms"],
                "model": name,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    """POST /api/compare?models=a,b + image 文件 → 多模型并行对比"""
    raw = request.args.get("models", "")
    names = _parse_models(raw) if raw else registry.list_models()
    if not names:
        return jsonify({"error": "no models available"}), 400

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image"}), 400

    try:
        results = registry.compare(_read_image(file), names)
        return jsonify(
            {
                "models": {
                    name: {
                        "predictions": _to_dict(r["predictions"]),
                        "time_ms": r["time_ms"],
                    }
                    for name, r in results.items()
                }
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models")
def api_models():
    return jsonify(registry.list_models())


@app.route("/api/model-info")
def model_info():
    return jsonify(registry.list_models_with_meta())


@app.route("/api/classes")
def api_classes():
    return jsonify(
        [{"index": i, "name": CLASS_NAMES[c]} for i, c in enumerate(CLASSES)]
    )


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "models": registry.list_models(),
            "model_count": len(registry.list_models()),
        }
    )


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=_PORT, threaded=False)
    except KeyboardInterrupt:
        print("\n[INFO] 服务已停止")
