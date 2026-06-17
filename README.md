# 织物智能分类系统

基于 YOLO 系列 (yolo26) 的面料图像分类，支持训练、推理、Web 部署。

## 快速开始

```bash
conda create -n yolo python=3.12 -y && conda activate yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 工作流程

### 0. 采集

通过浏览器拍照采集图片，按类别保存到 `data_ori/<数据集>/`。

```bash
python collect/collect_server.py
```

访问 `http://localhost:5000`

### 1. 预处理

从 `data_ori/<数据集>/` 按类别目录读取原始图片，划分训练集/验证集，输出到 `data/<数据集>/`。

```bash
python main.py prepare [数据集名]
```

不指定数据集名时默认为 `yueli`。

### 2. 训练

`--model` 指定模型名称或 `.pt` 文件，自动识别 YOLO 版本并匹配 YAML 配置。训练完成后模型保存在 `runs/classify/<数据集>_<模型>/weights/best.pt`。

```bash
python main.py train --data data/<数据集> --model <模型名>
```

模型名示例：`yolo26n`、`yolo26s`、`yolo11n`、`yolov8s`。

### 3. 导出

```bash
python main.py export --model <名称>        # ONNX → deploy/models/
python main.py export-arm --model <名称>    # NCNN → deploy_arm_v2/models/
```

### 4. 部署

```bash
# Windows
cd deploy && start.bat

# Linux
cd deploy && ./start.sh
```

访问 `http://localhost:8564`

```bash
# 树莓派
rsync -avz deploy_arm_v2/ pi@<IP>:/home/pi/deploy_arm_v2/
ssh pi@<IP> "cd /home/pi/deploy_arm_v2 && pip install -r requirements.txt && ./start.sh"
```

访问 `http://<IP>:3000`

## 完整示例

```python
python collect/collect_server.py
python main.py prepare yueli
python main.py train --data data/yueli --model yolo26n
python main.py export --model yueli_yolo26n
```

```python
python main.py export-arm --model yueli_yolo26n
```

## 目录

```
├── main.py                  # 统一入口
├── config/                  # 超参数 + 模型 YAML + 数据集配置
├── src/                     # 核心模块
├── weights/                 # 预训练 & 训练后权重
├── data_ori/                # 原始数据
├── data/                    # 预处理后数据
├── runs/                    # 训练输出
├── deploy/                  # ONNX Runtime 部署
└── deploy_arm_v2/           # NCNN 部署 (树莓派)
```
