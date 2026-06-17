#!/bin/bash
set -e

DIR=$(cd "$(dirname "$0")" && pwd)
PID_FILE="$DIR/.server.pid"
LOG_FILE="$DIR/server.log"
VENV_DIR="$DIR/.venv"
INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

ensure_uv() {
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv &>/dev/null && return 0
  echo "[*] 安装 uv ..."
  if command -v curl &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget &>/dev/null; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    pip install uv
  fi
  [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
  command -v uv &>/dev/null || { echo "[X] uv 安装失败"; exit 1; }
}

# ═══════════════════ 安装环境 ═══════════════════
setup() {
  ensure_uv

  if [ -f "$VENV_DIR/pyvenv.cfg" ] && ! "$VENV_DIR/bin/python" --version &>/dev/null; then
    echo "[*] 损坏的 venv, 删除重建 ..."
    rm -rf "$VENV_DIR"
  fi

  echo "[*] 创建虚拟环境 ..."
  rm -rf "$VENV_DIR"
  uv venv "$VENV_DIR"

  echo "[*] 安装系统编译依赖 ..."
  sudo apt-get install -y cmake build-essential libjpeg-dev zlib1g-dev 2>/dev/null || true

  echo "[*] 安装 pybind11 ..."
  uv pip install --index-url "$INDEX_URL" "pybind11>=2.10"

  echo "[*] 安装运行时依赖 ..."
  uv pip install --index-url "$INDEX_URL" \
    "flask>=3.0" "pillow>=10.0" "pyyaml>=6.0" "numpy>=1.24"

  echo "[*] 编译安装 ncnn (NO_OPENCV) ..."
  if "$VENV_DIR/bin/python" -c "import ncnn" 2>/dev/null; then
    echo "[*] ncnn 已安装, 跳过编译"
  else
    uv pip install --index-url "$INDEX_URL" "ncnn>=1.0" --no-build-isolation --no-deps
  fi

  uv cache clean 2>/dev/null || true

  [ -f "$DIR/config.yaml" ] || { echo "[X] config.yaml 缺失"; exit 1; }
  found=$(ls "$DIR/models"/*.tflite 2>/dev/null || ls -d "$DIR/models"/*_ncnn_model 2>/dev/null || true)
  [ -n "$found" ] || { echo "[X] models/ 缺少模型文件"; exit 1; }

  echo "[*] 环境就绪"
}

# ═══════════════════ 启动服务 ═══════════════════
start() {
  [ -f "$VENV_DIR/bin/python" ] || { echo "[X] 环境未安装, 请先执行: bash start.sh setup"; exit 1; }
  [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null && { echo "[!] 已在运行 PID $(cat $PID_FILE)"; return; }

  echo "[*] 启动服务..."
  export OMP_NUM_THREADS=4
  nohup "$VENV_DIR/bin/python" "$DIR/server.py" >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  for i in $(seq 1 30); do
    grep -q "Running on" "$LOG_FILE" 2>/dev/null && { echo "[*] 就绪"; break; } || sleep 1
  done
  status
}

# ═══════════════════ 其他 ═══════════════════
stop() {
  [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null && { kill $(cat "$PID_FILE"); rm -f "$PID_FILE"; echo "[*] 已停止"; return; }
  echo "[!] 未运行"; rm -f "$PID_FILE"
}

status() {
  [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null && echo "[*] 运行中  PID: $(cat $PID_FILE)" || echo "[!] 未运行"
}

log()      { tail -f "$LOG_FILE"; }
restart()  { stop;  [ -f "$VENV_DIR/bin/python" ] || { echo "[X] 环境未安装"; exit 1; }; start; }

case "${1:-start}" in
  setup)   setup ;;
  start)   start ;;
  stop)    stop ;;
  status)  status ;;
  log)     log ;;
  restart) restart ;;
  *)       echo "用法: $0 {setup|start|stop|status|log|restart}" ;;
esac
