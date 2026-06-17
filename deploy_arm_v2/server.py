"""日志必须最先配置，再导入 inference"""
import os, sys, logging

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG_PATH = os.path.join(_HERE, "server.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
_log = logging.getLogger("server")
_log.info("=== 启动 ===")
_log.info("Python: %s", sys.version)
_log.info("工作目录: %s", _HERE)

# ── 之后再导入 (Registry 构造含 ncnn 加载, 若 segfault 日志已就绪) ──
try:
    from inference import CFG, Registry, CLASSES, CLASS_NAMES
except Exception as e:
    _log.critical("导入 inference 失败: %s", e, exc_info=True)
    sys.exit(1)

_log.info("inference 导入成功")

import io, traceback, uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template

_PORT = CFG["port"]
_DEBUG_DIR = os.path.join(_HERE, "_debug_uploads")

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

_log.info("创建 Registry ...")
try:
    registry = Registry()
except Exception as e:
    _log.critical("Registry 创建失败: %s", e, exc_info=True)
    sys.exit(1)

models_list = registry.list_models()
default_model = models_list[0] if models_list else None
_log.info("默认模型: %s", default_model)
_log.info("模型列表: %s", models_list)


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=models_list,
        default=default_model,
        class_count=len(CLASSES),
        webcam_interval_ms=CFG.get("webcam_interval_ms", 1500),
    )


@app.route("/api/classify", methods=["POST"])
def classify():
    name = request.args.get("model", default_model)
    if not name:
        return jsonify({"error": "no model available"}), 400

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image"}), 400

    try:
        os.makedirs(_DEBUG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        save_path = os.path.join(_DEBUG_DIR, f"{ts}_{uid}.jpg")
        file.save(save_path)
        file.stream.seek(0)
        _log.info("保存上传: %s (%s B)", save_path, os.path.getsize(save_path))
    except Exception as e:
        _log.warning("保存失败: %s", e)

    data = file.read()
    _log.info("classify %s: %d bytes", name, len(data))

    if not data or len(data) < 100:
        return jsonify({"error": "图片数据为空或过小"}), 400

    try:
        result = registry.classify(name, data)
        if result is None:
            return jsonify({"error": f"model not found: {name}"}), 404
        _log.info("classify %s: %s ms", name, result["time_ms"])
        return jsonify({
            "results": result["predictions"],
            "time_ms": result["time_ms"],
            "model": name,
        })
    except Exception:
        _log.error("classify 失败", exc_info=True)
        return jsonify({"error": "图片识别失败"}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    raw = request.args.get("models", "")
    names = [m.strip() for m in raw.split(",") if m.strip()]
    if not names:
        names = registry.list_models()
    if not names:
        return jsonify({"error": "no models available"}), 400

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image"}), 400

    try:
        os.makedirs(_DEBUG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        save_path = os.path.join(_DEBUG_DIR, f"cmp_{ts}_{uid}.jpg")
        file.save(save_path)
        file.stream.seek(0)
    except Exception:
        pass

    data = file.read()
    if not data or len(data) < 100:
        return jsonify({"error": "图片数据为空或过小"}), 400

    try:
        results = registry.compare(data, names)
        return jsonify({
            "models": {
                name: {
                    "predictions": r["predictions"],
                    "time_ms": r["time_ms"],
                }
                for name, r in results.items()
            }
        })
    except Exception:
        _log.error("compare 失败", exc_info=True)
        return jsonify({"error": "图片识别失败"}), 500


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
    return jsonify({
        "status": "ok",
        "models": registry.list_models(),
        "model_count": len(registry.list_models()),
    })


if __name__ == "__main__":
    _log.info("启动 Flask: 0.0.0.0:%d", _PORT)
    try:
        app.run(host="0.0.0.0", port=_PORT, threaded=True)
    except KeyboardInterrupt:
        _log.info("服务已停止")
