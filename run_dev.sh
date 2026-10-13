#!/usr/bin/env bash
# Stock Pulse - Local Development Runner (Bash)
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="python3"
if [ -f "$ROOT_DIR/venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/venv/bin/python"
elif [ -f "$ROOT_DIR/venv/Scripts/python.exe" ]; then
    PYTHON="$ROOT_DIR/venv/Scripts/python.exe"
fi

echo "========================================"
echo "   Stock Pulse Development Server       "
echo "========================================"

echo "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000..."
$PYTHON -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

trap "echo 'Stopping backend...'; kill $BACKEND_PID" EXIT INT TERM

echo "[2/2] Starting Vite Frontend on http://localhost:5173..."
cd "$ROOT_DIR/app/frontend"
npm run dev
