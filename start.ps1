# start.ps1 — 重新建置並啟動 fb-clinics-agent
# 用法：在專案根目錄的終端機輸入  .\start.ps1
#
# ⚠️ 容器的所有設定（port / env_file / volume / logging / restart）一律只寫在
#    docker-compose.yml，這支腳本**刻意不重複**任何一項。
#
#    這裡曾經是 docker build + docker run，把 name / port / env / restart / log-opt
#    全部又寫了一遍 —— 結果就是漏掉 compose 裡的三個 chroma volume，
#    導致走這條路啟動時每次都要重新呼叫 OpenAI 重建向量索引
#    （慢 15~30 秒，且讓「容器能不能啟動」綁死在 OpenAI 通不通上）。
#    同一份設定寫兩遍，遲早有一遍會漏。所以改成呼叫 compose，讓設定只有一份。
#
#    本腳本只保留 compose 沒有的東西：.env 檢查、進度提示、自動跟 log。

# 切換到腳本所在資料夾（確保在有 docker-compose.yml 的目錄執行）
Set-Location -Path $PSScriptRoot

# 檢查 .env 是否存在（compose 的 env_file 指向它，缺了容器會啟動失敗）
if (-not (Test-Path ".env")) {
    Write-Host "[錯誤] 找不到 .env 檔，容器會啟動失敗。請先建立 .env。" -ForegroundColor Red
    exit 1
}

Write-Host "[1/2] docker compose 建置 image 並啟動容器..." -ForegroundColor Cyan
# up -d --build 會自動處理舊容器（重新建立），不需要先 docker rm -f
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "[錯誤] 啟動失敗，請看上面的訊息。" -ForegroundColor Red
    exit 1
}

Write-Host "[2/2] 啟動完成，開始跟隨 log（按 Ctrl+C 離開，不會停止容器）..." -ForegroundColor Green
docker compose logs -f
