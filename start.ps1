# start.ps1 — 用 docker build + docker run 重新建置並啟動 fb-clinics-agent
# 用法：在專案根目錄的終端機輸入  .\start.ps1

# 切換到腳本所在資料夾（確保在有 Dockerfile 的目錄執行）
Set-Location -Path $PSScriptRoot

# 檢查 .env 是否存在（容器需要它注入環境變數）
if (-not (Test-Path ".env")) {
    Write-Host "[錯誤] 找不到 .env 檔，容器會啟動失敗。請先建立 .env。" -ForegroundColor Red
    exit 1
}

# 若已有同名容器（不論在跑或已停），先移除，避免 docker run 撞名失敗
$existing = docker ps -aq -f "name=^fb-clinics-agent$"
if ($existing) {
    Write-Host "[清理] 移除舊的 fb-clinics-agent 容器..." -ForegroundColor Yellow
    docker rm -f fb-clinics-agent | Out-Null
}

Write-Host "[1/3] docker build 建置 image..." -ForegroundColor Cyan
docker build -t fb-clinics-agent .
if ($LASTEXITCODE -ne 0) {
    Write-Host "[錯誤] docker build 失敗，請看上面的訊息。" -ForegroundColor Red
    exit 1
}

Write-Host "[2/3] docker run 啟動容器..." -ForegroundColor Cyan
docker run -d `
    --name fb-clinics-agent `
    --env-file .env `
    -p 8004:8004 `
    --restart unless-stopped `
    fb-clinics-agent
if ($LASTEXITCODE -ne 0) {
    Write-Host "[錯誤] docker run 失敗，請看上面的訊息。" -ForegroundColor Red
    exit 1
}

Write-Host "[3/3] 啟動完成，開始跟隨 log（按 Ctrl+C 離開，不會停止容器）..." -ForegroundColor Green
docker logs -f fb-clinics-agent

