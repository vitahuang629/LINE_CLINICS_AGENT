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

# 先用單一進程 warmup 建好索引（避免兩個 worker 同時重建 chroma 造成寫入衝突），
# 再啟動多 worker 的 uvicorn —— 此時 worker 只會載入已建好的索引。
CMD ["sh", "-c", "python -c 'import toolkit.toolkits' && uvicorn main_webhook:app --host 0.0.0.0 --port 8004 --workers 2"]
