# Stock Pulse - Local Development Runner (PowerShell)
# Starts FastAPI backend (port 8000) and Vite frontend (port 5173)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Stock Pulse Development Server       " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$ROOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT_DIR

# Determine Python path
$PYTHON = "python"
if (Test-Path "$ROOT_DIR\venv\Scripts\python.exe") {
    $PYTHON = "$ROOT_DIR\venv\Scripts\python.exe"
}

Write-Host "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Green
$backendProcess = Start-Process -FilePath $PYTHON -ArgumentList "-m", "uvicorn", "app.backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload" -PassThru -NoNewWindow

Write-Host "[2/2] Starting Vite Frontend on http://localhost:5173..." -ForegroundColor Green
Set-Location "$ROOT_DIR\app\frontend"

try {
    & npm.cmd run dev
} finally {
    Write-Host "`nStopping backend process..." -ForegroundColor Yellow
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force
    }
    Set-Location $ROOT_DIR
}
