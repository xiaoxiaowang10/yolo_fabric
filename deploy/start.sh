#!/bin/bash
set -e

DIR=$(cd "$(dirname "$0")" && pwd)
PID_FILE="$DIR/.server.pid"
LOG_FILE="$DIR/server.log"
VENV_DIR="$DIR/.venv"

setup() {
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo "[*] 创建虚拟环境..."
        python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install -q --upgrade pip
        "$VENV_DIR/bin/pip" install -q -r "$DIR/requirements.txt"
        echo "[*] 依赖安装完成"
    fi
    if [ ! -f "$DIR/config.yaml" ]; then
        echo "[X] config.yaml 缺失" && exit 1
    fi
    if [ -z "$(ls "$DIR/models"/*.onnx 2>/dev/null)" ]; then
        echo "[X] models/ 目录缺少 .onnx 文件" && exit 1
    fi
}

start() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "[!] 服务已在运行 (PID: $(cat $PID_FILE))"
        return
    fi
    setup
    echo "[*] 启动服务..."
    nohup "$VENV_DIR/bin/python" "$DIR/server.py" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1
    status
}

stop() {
    if [ ! -f "$PID_FILE" ] || ! kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "[!] 服务未运行"
        rm -f "$PID_FILE"
        return
    fi
    kill $(cat "$PID_FILE") && rm -f "$PID_FILE"
    echo "[*] 服务已停止"
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "[*] 运行中  PID: $(cat $PID_FILE)"
    else
        echo "[!] 未运行"
    fi
}

log()  { tail -f "$LOG_FILE"; }
restart() { stop; start; }

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    status)  status ;;
    log)     log ;;
    restart) restart ;;
    *)       echo "用法: $0 {start|stop|status|log|restart}" ;;
esac
