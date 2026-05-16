# 织物分类 — 部署

## 准备

确保 `models/` 目录下有 `.onnx` 模型文件，`config.yaml` 配置正确。

```bash
# 从项目根目录导出 ONNX 模型到 deploy/models/
python main.py export --model runs/classify/d1_yolo26m/weights/best.pt
```

---

## Windows

```bash
cd deploy
start.bat
```
自动完成：检查 Python → 创建 venv → 安装依赖 → 校验模型 → 打开浏览器 → 启动服务

---

## Linux (ECS 服务器常驻后台)

```bash
cd deploy
chmod +x start.sh

./start.sh           # 首次自动安装依赖并启动
./start.sh status    # 查看运行状态
./start.sh log       # 实时查看日志
./start.sh stop      # 停止服务
./start.sh restart   # 重启服务
```

启动后访问 `http://<服务器IP>:8564`

**查看**：`./start.sh status` 显示 PID，`./start.sh log` 实时追尾日志

**删除**：`./start.sh stop` 停止服务；`rm -rf deploy/.venv deploy/.server.pid deploy/server.log` 清理残留

---

## 端口修改

编辑 `deploy/config.yaml` 中 `port` 字段，重启生效：

```bash
./start.sh restart
```
