# ECS Linux 部署指南

从头部署织物分类系统到阿里云 ECS，含 HTTPS 摄像头支持。

---

## 架构

```
https://IP:8564  ──→  Nginx :8564 (SSL)  ──→  Flask :8565 (HTTP)
```

浏览器通过 HTTPS 访问 → Nginx 解密 → 转发给本机 Flask。Flask 端口改为 8565，仅本机监听。

---

## 前置条件

- ECS 已安装 Python 3.10+
- 项目代码已上传至 `~/yolo_v2/`
- `deploy/models/` 下已有 `.onnx` 文件

```bash
# 导出 ONNX 模型（在本地或 ECS 有 GPU 环境执行）
python main.py export --model runs/classify/d1_yolo26m/weights/best.pt
```

---

## 部署步骤

### 1. 安装 Nginx

```bash
apt update && apt install nginx -y
```

### 2. 生成自签证书

```bash
mkdir -p /etc/nginx/ssl
openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/selfsigned.key \
  -out /etc/nginx/ssl/selfsigned.crt \
  -subj "/CN=ECS"
```

### 3. 写入 Nginx 配置

```bash
cat > /etc/nginx/sites-available/fabric << 'EOF'
server {
    listen 8564 ssl;
    server_name _;
    ssl_certificate     /etc/nginx/ssl/selfsigned.crt;
    ssl_certificate_key /etc/nginx/ssl/selfsigned.key;
    error_page 497 =301 https://$host:8564$request_uri;
    location / {
        proxy_pass http://127.0.0.1:8565;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF
ln -sf /etc/nginx/sites-available/fabric /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
rm -f /etc/nginx/conf.d/default.conf 2>/dev/null
nginx -t && systemctl reload nginx
```

### 4. 调整 Flask 端口（避免与 Nginx 冲突）

```bash
cd ~/yolo_v2
./start.sh stop
sed -i 's/port: 8564/port: 8565/' config.yaml
./start.sh start
```

变化：`config.yaml` 中 `port: 8564` → `port: 8565`

### 5. 开放安全组

阿里云控制台 → ECS → 安全组 → 入方向 → 添加：

| 协议 | 端口 | 授权对象 |
|---|---|---|
| TCP | 8564 | 0.0.0.0/0 |

### 6. 验证

```bash
# Flask 是否运行
./start.sh status

# Nginx SSL 链路是否通
curl -sk https://127.0.0.1:8564 | head -3
# 应返回 <!DOCTYPE html>
```

---

## 访问

```
https://<公网IP>:8564
```

首次访问浏览器提示"不安全" → **高级 → 继续前往**（自签证书正常现象，仅首次）。

页面功能：图片上传 / 实时拍照（摄像头可用）/ 多模型对比 / 类别总览。

---

## 日常运维

```bash
cd ~/yolo_v2

./start.sh status       # 查看运行状态
./start.sh log          # 实时日志
./start.sh stop         # 停止
./start.sh restart      # 重启
```

---

## 原理说明

`http://IP:8564` 下浏览器禁止摄像头（`getUserMedia` 要求安全上下文）。

Nginx 在 8564 端口提供 HTTPS，自签证书加密生效但无 CA 背书，所以浏览器标记"不安全"。这只是提示——连接仍为 TLS 加密，摄像头 API 正常放行。

若想消除"不安全"提示，需申请正规 SSL 证书（如 Let's Encrypt），需绑定域名。
