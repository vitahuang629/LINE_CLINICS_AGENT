# =====================================================================
# Stage 1: 用 poetry 安裝套件（避開 poetry export 的 bug）
# =====================================================================
FROM python:3.11-slim-bookworm AS builder

# 系統工具（編譯 numpy / pandas / jieba 等套件用）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 裝 poetry
RUN pip install --no-cache-dir poetry==1.8.3

# 設定 poetry：不要建 venv（直接裝到系統 Python）
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# 先只 COPY 依賴設定，讓這層 cache 不會被原始碼變動影響
COPY pyproject.toml poetry.lock ./

# 安裝套件（不含 dev、不裝專案本身）
RUN poetry install --without dev --no-root && \
    rm -rf /tmp/poetry_cache


# =====================================================================
# Stage 2: Final image（slim、只含 runtime 必要檔案）
# =====================================================================
FROM python:3.11-slim-bookworm

# runtime 工具（curl 留著方便進 container 內 debug）
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 從 Stage 1 把已安裝的套件搬過來
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# COPY 程式碼
COPY agent.py main_webhook.py ./
COPY app ./app
COPY toolkit ./toolkit
COPY utils ./utils
COPY data ./data
COPY prompt_library ./prompt_library
COPY data_models ./data_models

# 注意：不再把 chroma 向量庫 / bm25.pkl 烤進 image。
# 容器啟動時會用 COPY 進來的最新 data/*.csv 自動重建索引，確保不會帶舊快取上線。

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8004

EXPOSE 8004

# 存活探測：打 GET /health（見 main_webhook.py）。
# 放在 Dockerfile 而不是 docker-compose.yml，是因為 start.ps1 走的是 docker run，
# 寫在 compose 裡那條路徑就吃不到；寫在這裡兩種啟動方式都有。
#
# start-period 給 120s：容器啟動時要先跑單進程 warmup 建索引（首次部署約 30 秒，
# 之後約 15 秒），這段期間根本還沒有 uvicorn 在聽 port，探測必然失敗。
# start-period 內的失敗不計入 retries，所以不會在 warmup 途中被誤判。
#
# interval 用 1h（不是常見的 30s）：探測每次都會在 uvicorn access log 留一行
# `GET /health 200`，30 秒一次等於一天 2,880 行雜訊，在 Portainer 捲 log 找 📥/📤
# 時會被洗版。這裡刻意用「log 可讀性」換「偵測速度」。
# retries 配合降為 2 —— 間隔拉長後，「連續兩次、相隔一小時都失敗」已是很強的訊號。
#
# ⚠️ 代價：服務壞掉後最壞要 2 小時才會顯示 unhealthy。這個取捨的前提是
#    healthcheck 在這裡只是「給人看的狀態燈」，不是自動復原機制（見下）。
#    若哪天要靠它做自動重啟或告警，interval 必須調回 30~60s，
#    改用 access log filter 濾掉 /health 來解決雜訊問題。
#
# 注意：Docker 的 healthcheck 只會把容器標成 unhealthy（Portainer / docker ps 看得到），
# restart: unless-stopped 只對「進程結束」生效，不會因為 unhealthy 就自動重啟。
# 要自動重啟得另外掛 autoheal 之類的東西。
HEALTHCHECK --interval=1h --timeout=5s --start-period=120s --retries=2 \
    CMD curl -fsS http://localhost:8004/health || exit 1

# 先用單一進程 warmup 建好索引（避免兩個 worker 同時重建 chroma 造成寫入衝突），
# 再啟動多 worker 的 uvicorn —— 此時 worker 只會載入已建好的索引。
CMD ["sh", "-c", "python -c 'import toolkit.toolkits' && uvicorn main_webhook:app --host 0.0.0.0 --port 8004 --workers 2"]
