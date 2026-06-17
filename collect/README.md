# 数据采集工具

## 启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python collect_server.py
```

浏览器访问 `http://localhost:5000`

## 配置

修改 `config.yaml` 中的类别和数据集名称。

## 功能

- 采集：摄像头拍照 + 连拍，按类别保存到 `data_ori/{数据集}/{类别}/`
- 管理：浏览图片、归类（移动到其他类别）、删除
