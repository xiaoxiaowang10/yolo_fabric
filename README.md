# 织物智能分类系统

基于 YOLO26 的面料图像分类，支持训练、推理、多模型对比、Web 可视化部署。

## 数据来源
下载路径在data_ori中

## 快速开始

```bash
conda create -n yolo python=3.12 -y && conda activate yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
pip install -r requirements.txt
```

## 统一入口 (main.py)

```bash
python main.py prepare [dataset]        # 数据准备 (默认 ibug)
python main.py train --data data/d1      # 训练
python main.py predict --input <路径>    # 推理
python main.py export                    # 导出 ONNX
```

### 数据准备

```bash
python main.py prepare ibug      # 使用 config/datasets/ibug.yaml
python main.py prepare wisudaa  # 使用 config/datasets/wisudaa.yaml
python main.py prepare fabric_yolo26
python main.py prepare deep_learning_testing
```

### 训练多个模型变体

```bash
python main.py train --data data/d1 --model yolo26n-cls.pt
python main.py train --data data/d1 --model yolo26s-cls.pt
python main.py train --data data/d1 --model yolo26m-cls.pt
python main.py train --data data/d1 --model yolo26l-cls.pt
```

### 图片推理 (直接使用 .pt 模型)

训练完成后，直接用 `.pt` 文件对图片/目录/视频进行预测，无需先导出 ONNX 或启动 Web 服务。

```bash
# 单张图片
python main.py predict --input_path fabric.jpg

# 指定训练好的模型
python main.py predict --input_path fabric.jpg  --model runs/classify/d1_yolo26n/weights/best.pt

# 显示 Top-5 预测
python main.py predict --input_path fabric.jpg --top 5

# 设置置信度阈值 (低于阈值返回"未识别")
python main.py predict --input_path fabric.jpg --threshold 0.6

# 整目录批量预测
python main.py predict --input_path data_ori/new_fabrics/

# 视频预测 (逐帧识别)
python main.py predict --input_path video.mp4 --save

# CPU 推理
python main.py predict --input_path fabric.jpg --device cpu
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_path` | 图片/目录/视频路径 | 必填 |
| `--model` | .pt 模型路径 | `best.pt` (当前目录) |
| `--top` | 显示 Top-K 预测结果 | 3 |
| `--threshold` | 置信度阈值 (低于此值返回"未识别") | 0.0 |
| `--imgsz` | 推理尺寸 | 384 |
| `--device` | 推理设备 (cpu / 0 / 1) | 0 |
| `--show` | 实时显示预测画面 (视频) | — |
| `--save` | 保存预测结果视频 | — |

### 批量导出

```bash
python main.py export --model <train_name>
python main.py export --model d1_yolo26n
python main.py export --model d1_yolo26s
python main.py export --model d1_yolo26m
# 导出到 deploy/models/ 并自动生成 deploy/config.yaml
```
### 继续训练
```bash
python main.py export --model continue --model runs/classify/<train_name>/weights/best.pt --data <dataset_dir>
python main.py export --model continue --model runs/classify/d1_yolo26n/weights/best.pt --data data/d5
```




## Web 部署 (Flask + ONNX)

### 导出模型

```bash
python main.py export --model d1_yolo26m
# .onnx → deploy/models/
```

### Windows

```bash
cd deploy && start.bat
```

### Linux (ECS 常驻后台)

```bash
cd deploy
chmod +x start.sh
./start.sh              # 启动（首次自动装依赖）
./start.sh status       # 查看状态
./start.sh log          # 实时日志
./start.sh stop         # 停止
./start.sh restart      # 重启
```

访问 `http://localhost:8564`

功能：单模型识别 / 实时拍照 / 多模型并排对比 / 类别总览

详见 `deploy/README.md`

## 目录

```
├── main.py                  # 统一入口
├── config/default.yaml      # 全局配置
├── config/datasets/         # 数据集配置
├── src/                     # 核心模块
│   ├── utils.py             # 工具函数
│   ├── prepare.py           # 数据准备
│   ├── train.py             # 模型训练
│   ├── predict.py           # 模型推理
│   ├── export.py            # 模型导出
│   └── incremental.py       # 增量训练
├── weights/                 # 预训练 & 训练后权重
├── data_ori/                # 原始数据
├── data/                    # 预处理后数据
├── runs/                    # 训练输出
└── deploy/                  # 自包含部署包 (Flask + ONNX Runtime)
```
