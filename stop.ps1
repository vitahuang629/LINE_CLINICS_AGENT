# stop.ps1 — 停止並移除 fb-clinics-agent 容器
# 用法：在專案根目錄的終端機輸入  .\stop.ps1

Set-Location -Path $PSScriptRoot

$existing = docker ps -aq -f "name=^fb-clinics-agent$"
if ($existing) {
    Write-Host "[停止] 移除 fb-clinics-agent 容器..." -ForegroundColor Yellow
    docker rm -f fb-clinics-agent | Out-Null
    Write-Host "[完成] 容器已停止並移除。" -ForegroundColor Green
} else {
    Write-Host "[略過] 找不到 fb-clinics-agent 容器，無需停止。" -ForegroundColor DarkGray
}

