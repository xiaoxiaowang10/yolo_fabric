# 织物分类 — 部署 (树莓派 v2, <100ms 目标)

## 架构

```
PC 开发机                         树莓派 (ARM Linux)
┌─────────────────┐              ┌──────────────────────┐
│ YOLO .pt        │              │ deploy_arm_v2/       │
│   ↓ export      │   rsync      │   ├─ server.py       │
│ NCNN            │ ──────────→  │   ├─ inference.py    │
│   ↓             │              │   ├─ models/*_ncnn_  │
│ deploy_arm_v2/  │              │   │   ├─ *.param     │
│   models/       │              │   │   └─ *.bin       │
└─────────────────┘              │   ├─ config.yaml     │
                                 │   └─ start.sh        │
                                 └──────────────────────┘
```

## 推理后端

| 后端 | 安装方式 | 性能 (估算, 384×384) | 备注 |
|------|---------|---------------------|------|
| **NCNN** 🏆 | `pip install ncnn` | RPi5: ~15ms, RPi4: ~40ms | 最快, 树莓派首选 |

基准数据来源: Ultralytics 官方 (RPi 5, YOLO26n detection 640×640)
- NCNN: 67ms → 384×384 classification ≈ 15ms

> 与 deploy/ (PC, ONNX Runtime) 预处理/后处理完全对齐, 保证识别结果一致.

## 使用

### 1. PC 端导出

```bash
# 导出 NCNN 模型 (imgsz 默认从 config.yaml 读取 = 384)
python main.py export-arm --model yueli_yolo26n
```

导出后模型位于 `deploy_arm_v2/models/yueli_yolo26n_imgsz384_ncnn_model/`.

### 2. 复制到树莓派

```bash
rsync -avz deploy_arm_v2/ pi@树莓派IP:/home/pi/deploy_arm_v2/
```

### 3. 安装依赖并启动

```bash
ssh pi@树莓派IP
cd /home/pi/deploy_arm_v2
pip install -r requirements.txt
chmod +x start.sh
./start.sh        # 首次自动装依赖并启动
./start.sh status # 查看状态
./start.sh log    # 实时日志
```

访问 `http://<树莓派IP>:3000`

### 4. 验证推理

```bash
python3 -c "
from inference import registry
from PIL import Image
import time

r = registry.get(list(registry.list_models())[0])
img = Image.new('RGB', (384, 384))
t0 = time.perf_counter()
for _ in range(10):
    r.predict(img)
print(f'平均: {(time.perf_counter()-t0)/10*1000:.0f}ms')
"
```

## 依赖

| 包 | 说明 |
|---|------|
| flask | Web 服务 |
| pillow, numpy | 预处理 (PIL BILINEAR + 中心裁切) |
| pyyaml | 配置读取 |
| ncnn | NCNN 推理 (ARM NEON 加速, FP16) |
