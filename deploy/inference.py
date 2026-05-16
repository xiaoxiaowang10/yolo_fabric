"""ONNX 推理引擎 —— 模型加载 / 预处理 / 推理 / 注册管理"""

import os, time
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image


# ═══════════════════ 配置 ═══════════════════


def _load_config():
    path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml 不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = _load_config()

# 强制字母序，与 YOLO ImageFolder 输出索引一致
CLASSES = sorted(CFG["classes"])
CLASS_NAMES = CFG["class_names"]
UNKNOWN_THRESHOLD = CFG["unknown_threshold"]
UNKNOWN_LABEL = CFG["unknown_label"]

_MODELS_DIR = os.path.join(os.path.dirname(__file__), CFG["models_dir"])
_IMGSZ = CFG["imgsz"]
_NUM_THREADS = CFG["num_threads"]


# ═══════════════════ ONNX 推理器 ═══════════════════


class FabricClassifier:
    """单个 ONNX 模型 —— 封装 .onnx 加载与推理

    Usage:
        clf = FabricClassifier("deploy/models/d3_yolo26n.onnx")
        results = clf.predict(image)
        # → [("棉布 / Cotton", 0.92), ...]
    """

    def __init__(self, model_path, imgsz=_IMGSZ, num_threads=_NUM_THREADS):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._imgsz = imgsz
        self._model_name = os.path.splitext(os.path.basename(model_path))[0]

    @property
    def model_name(self):
        return self._model_name

    @staticmethod
    def preprocess(img, imgsz):
        """PIL Image → ONNX 输入张量 [1, 3, H, W]
        步骤：等比缩放到最小边=imgsz → 中心裁切正方形 → [0,1] → BCHW"""
        w, h = img.size
        scale = imgsz / min(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        left = (nw - imgsz) // 2
        top = (nh - imgsz) // 2
        img = img.crop((left, top, left + imgsz, top + imgsz))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr.transpose(2, 0, 1), axis=0)

    def predict(self, image, top_k=5, threshold=None):
        """PIL Image / numpy 数组 → [(类别名, 置信度), ...]
        top-1 低于阈值则返回 [(unknown_label, conf)]"""
        if threshold is None:
            threshold = UNKNOWN_THRESHOLD
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        arr = self.preprocess(image.convert("RGB"), self._imgsz)
        probs = self._session.run(None, {self._session.get_inputs()[0].name: arr})[0][0]

        results = sorted(
            (
                (CLASS_NAMES.get(CLASSES[i], CLASSES[i]), float(p))
                for i, p in enumerate(probs)
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        top = results[:top_k]
        if threshold > 0 and top and top[0][1] < threshold:
            return [(UNKNOWN_LABEL, top[0][1])]
        return top


# ═══════════════════ 模型管理 ═══════════════════


class ModelRegistry:
    """管理 deploy/models/ 下所有 ONNX 模型 — 扫描 / 惰性加载 / 推理"""

    def __init__(self, models_dir=_MODELS_DIR, imgsz=_IMGSZ, num_threads=_NUM_THREADS):
        self._dir = models_dir
        self._imgsz = imgsz
        self._threads = num_threads
        self._clfs = {}  # {name: FabricClassifier} 惰性缓存
        self._meta = {}  # {name: {path, size_mb, mtime}}

    def scan(self):
        """扫描 models_dir 下所有 .onnx 文件"""
        self._meta.clear()
        if not os.path.isdir(self._dir):
            return
        for f in sorted(os.listdir(self._dir)):
            if not f.endswith(".onnx"):
                continue
            name = os.path.splitext(f)[0]
            path = os.path.join(self._dir, f)
            self._meta[name] = {
                "path": path,
                "size_mb": round(os.path.getsize(path) / 1e6, 1),
                "mtime": os.path.getmtime(path),
            }

    def list_models(self):
        return sorted(self._meta.keys())

    def list_models_with_meta(self):
        return [
            {"name": n, "size_mb": m["size_mb"], "loaded": n in self._clfs}
            for n, m in sorted(self._meta.items())
        ]

    def get(self, name):
        """惰性加载分类器"""
        if name not in self._meta:
            return None
        if name not in self._clfs:
            self._clfs[name] = FabricClassifier(
                self._meta[name]["path"],
                self._imgsz,
                self._threads,
            )
        return self._clfs[name]

    def classify(self, model_name, image, top_k=5):
        """单模型推理，返回 {model, predictions, time_ms}"""
        clf = self.get(model_name)
        if clf is None:
            return None
        t0 = time.perf_counter()
        preds = clf.predict(image, top_k=top_k)
        elapsed = (time.perf_counter() - t0) * 1000
        return {"model": model_name, "predictions": preds, "time_ms": round(elapsed, 1)}

    def compare(self, image, model_names=None, top_k=5):
        """多模型串行推理，返回 {模型名: 结果}"""
        names = model_names or self.list_models()
        return {
            name: r for name in names if (r := self.classify(name, image, top_k=top_k))
        }


# ═══════════════════ 全局单例 ═══════════════════

registry = ModelRegistry()
registry.scan()
print(f"Models: {registry.list_models() or 'none'}")
