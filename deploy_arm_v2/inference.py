import os, sys, time, io, logging
import numpy as np
import yaml
from PIL import Image
import ncnn as _ncnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_log = logging.getLogger("inference")

CFG = yaml.safe_load(open(os.path.join(_HERE, "config.yaml"), encoding="utf-8"))
CLASSES = CFG["classes"]
CLASS_NAMES = CFG["class_names"]
_IMGSZ = CFG["imgsz"]


def _parse_blobs(param_path):
    in_blob, out_blob = "in0", "out0"
    for line in open(param_path, encoding="utf-8").read().splitlines()[2:]:
        parts = line.split()
        if len(parts) < 4: continue
        try: inc = int(parts[2])
        except: continue
        os_ = 3 + inc
        if len(parts) <= os_: continue
        try: outc = int(parts[os_])
        except: continue
        if parts[0] == "Input" and outc > 0: in_blob = parts[os_ + 1]
        elif parts[0] == "Softmax" and outc > 0: out_blob = parts[os_ + 1]
    return in_blob, out_blob


class FabricClassifier:
    def __init__(self, model_dir, imgsz=_IMGSZ):
        self.imgsz = imgsz
        self.name = os.path.basename(str(model_dir)).replace("_ncnn_model", "")

        param = bin_ = None
        for f in os.listdir(str(model_dir)):
            if f.endswith(".param"): param = os.path.join(str(model_dir), f)
            elif f.endswith(".bin"): bin_ = os.path.join(str(model_dir), f)

        _log.info("加载 %s: param=%s bin=%s", self.name, param, bin_)

        self.net = _ncnn.Net()
        self.net.load_param(param)
        self.net.load_model(bin_)
        self.net.opt.num_threads = CFG.get("num_threads", 4)

        try:
            self.net.opt.use_fp16_packed = True
            self.net.opt.use_fp16_storage = True
            self.net.opt.use_fp16_arithmetic = True
        except Exception:
            _log.warning("FP16 选项不可用，跳过")

        self.in_blob, self.out_blob = _parse_blobs(param)

        meta = yaml.safe_load(open(os.path.join(str(model_dir), "metadata.yaml"), encoding="utf-8"))
        names = meta.get("names", {})
        self.classes = [names[i] for i in sorted(names.keys(), key=int)]

        _log.info("  %s: %d cls, in=%s out=%s", self.name, len(self.classes), self.in_blob, self.out_blob)

        try:
            self._warmup()
        except Exception as e:
            _log.error("Warmup 失败: %s", e)

    def _warmup(self):
        d = np.zeros((3, self.imgsz, self.imgsz), dtype=np.float32)
        for _ in range(3):
            ex = self.net.create_extractor()
            ex.input(self.in_blob, _ncnn.Mat(d).clone())
            ex.extract(self.out_blob)

    def preprocess(self, img):
        w, h = img.size
        s = self.imgsz / min(w, h)
        nw, nh = int(w * s), int(h * s)
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        l, t = (nw - self.imgsz) // 2, (nh - self.imgsz) // 2
        img = img.crop((l, t, l + self.imgsz, t + self.imgsz))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    def predict_bytes(self, data, top_k=3):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = self.preprocess(img)
        ex = self.net.create_extractor()
        ex.input(self.in_blob, _ncnn.Mat(arr).clone())
        _, out_mat = ex.extract(self.out_blob)
        out = np.array(out_mat).flatten()
        results = [(self.classes[i], float(p)) for i, p in enumerate(out)]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"name": CLASS_NAMES.get(n, n), "confidence": round(c, 4)} for n, c in results[:top_k]]


class Registry:
    def __init__(self):
        models_dir = os.path.join(_HERE, CFG.get("models_dir", "models"))
        self.models = {}
        self.models_meta = []
        self._model_names = []

        _log.info("扫描模型目录: %s", models_dir)
        if not os.path.isdir(models_dir):
            _log.warning("模型目录不存在: %s", models_dir)
            return

        for name in sorted(os.listdir(models_dir)):
            path = os.path.join(models_dir, name)
            if not os.path.isdir(path):
                continue
            try:
                clf = FabricClassifier(path)
                self.models[clf.name] = clf
                size_mb = round(sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fn in os.walk(path)
                    for f in fn
                ) / (1024 * 1024), 1)
                self.models_meta.append({
                    "name": clf.name,
                    "size_mb": size_mb,
                    "loaded": True,
                })
                _log.info("模型 %s 加载完成 (%.1f MB)", clf.name, size_mb)
            except Exception as e:
                _log.error("加载模型 %s 失败: %s", name, e, exc_info=True)

        self._model_names = [m["name"] for m in self.models_meta]
        _log.info("共 %d 个模型: %s", len(self._model_names), self._model_names)

    def list_models(self):
        return list(self._model_names)

    def list_models_with_meta(self):
        return list(self.models_meta)

    def classify(self, name, image_bytes):
        clf = self.models.get(name)
        if clf is None:
            return None
        t0 = time.perf_counter()
        predictions = clf.predict_bytes(image_bytes)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        return {"predictions": predictions, "time_ms": elapsed}

    def compare(self, image_bytes, model_names=None):
        if model_names is None:
            model_names = self._model_names
        results = {}
        for name in model_names:
            r = self.classify(name, image_bytes)
            if r:
                results[name] = r
        return results
