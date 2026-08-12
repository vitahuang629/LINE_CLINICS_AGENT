
  # Clinics AI Agent

  一個醫美診所專用的 AI 客服系統，結合 **OpenAI / LangChain / LangGraph / LangSmith**，
  以 LangGraph 多代理核心對接 **Facebook Messenger**（後端串接 `/chat` API），可以：
  - 提供症狀諮詢與同理心回覆
  - 推薦本診所療程（嚴格限定療程白名單，避免幻覺）
  - 查詢療程費用、初診資訊與診所基本資料
  - 協助用戶預約並在需要時轉接真人客服

  ---

  ## 功能特色
  - **AI 諮詢助理**：理解使用者症狀與需求，動態追問收集資訊，提供精準療程推薦
  - **多代理協作**：以 LangGraph 編排 Supervisor → Information / Booking → Moderator 流程
  - **防幻覺機制**：療程白名單 + 費用工具化（`get_treatment_fee`）+ 輸出端價格守門，AI 只能依檢索結果與資料庫費用回覆
  - **圖片處理**：自動 OCR 辨識使用者上傳圖片，並依回覆內容帶出對應療程／對比照
  - **預約與轉真人**：整理預約資訊並通知管理群組，必要時觸發轉接真人客服（CallCS）
  - **通道無關核心**：LangGraph agent 與 I/O 接層解耦，目前對外為 FB Messenger `/chat` API；
    LINE 接層的實作已移除，`main_webhook.py` 僅保留註解掉的路由骨架

  ---

  ## 系統架構

  請求經由各通道接層進入，交給共用的 LangGraph 多代理流程處理，最後以
  「文字 + 圖片清單 + CallCS 標記」回傳：

  ```
  FB /chat API（後端串接層）
          │
          ▼
   start ─► guard（prompt injection 防護）─► supervisor（路由判斷）
                                                │
                            ┌───────────────────┴───────────────────┐
                            ▼                                        ▼
                   information_node                            booking_node
              （症狀同理、療程推薦）                      （費用、診所資訊、預約）
                            └───────────────────┬───────────────────┘
                                                ▼
                                         moderator（法規／語氣審查）─► 回傳
  ```

  詳細架構與各節點職責，見 [`clinic_agent_architecture.md`](./clinic_agent_architecture.md)。

  ---

  ## 環境需求
  - Python 3.11
  - 建議使用虛擬環境 `venv`（或 Poetry）
  - 需設定 `.env`（OpenAI API Key 等，不會進版控）
  - 部署另需 Docker / Docker Compose

  ---

  ## 部署（Docker）

  ```bash
  docker compose up -d --build
  ```

  - 服務跑在 `:8004`，對外只有 `POST /chat`
  - FastAPI 設了 `root_path="/fb-clinics-agent"`（`main_webhook.py:7`），
    所以掛在反向代理後面時實際路徑是 **`/fb-clinics-agent/chat`**；直接打容器則是 `/chat`
  - `.env` 由 `env_file` 在 runtime 注入，**不會包進 image** —— 部署到新機器時要手動放一份
  - 容器啟動時先用單一進程 warmup 建好索引，再開 2 個 uvicorn worker
    （避免兩個 worker 同時往同一個 Chroma 目錄寫入）
  - 另有 `GET /health` 供探活，只回 `{"status":"ok","pid":...}`，不打 OpenAI / MySQL
    （外部依賴抽風時不該把容器標成 unhealthy）。`pid` 可用來分辨是哪個 worker 回的

  ### 探活與 log

  `HEALTHCHECK` 寫在 **Dockerfile** 而非 compose —— 它是「這個 image 怎麼確認自己活著」的屬性，
  寫在 image 裡的話，不管用 compose、`docker run` 還是在 Portainer UI 上直接建容器都適用。
  `--start-period=120s` 是留給啟動時的 warmup（那段期間還沒有 uvicorn 在聽 port，
  探測必然失敗，但不計入 retries）。

  **interval 用 1 小時**（不是常見的 30s）：探測每次都會在 access log 留一行 `GET /health 200`，
  30 秒一次等於一天 2,880 行雜訊，在 Portainer 捲 log 找 `📥`/`📤` 時會被洗版。
  這是拿「偵測速度」換「log 可讀性」—— 代價是服務壞掉後最壞要 2 小時才顯示 `unhealthy`
  （interval 1h × retries 2）。前提是它只是狀態燈，不是自動復原機制（見下）。
  哪天要靠它做告警或自動重啟，就得調回 30~60s，改用 access log filter 濾掉 `/health` 解決雜訊。

  > Docker 的 healthcheck 只會把容器標成 `unhealthy`（Portainer / `docker ps` 看得到），
  > `restart: unless-stopped` 只對「進程結束」生效，**不會**因為 unhealthy 就自動重啟。
  > 需要自動復原的話得另外掛 autoheal。

  > ⚠️ `/health` 是 `async def`、跑在 event loop 上，**不經過 threadpool**。所以萬一 40 條
  > threadpool thread 全被卡住的請求佔滿、`/chat` 完全癱瘓，它照樣秒回 200。
  > 那種情況真正的防線是 LLM 呼叫的 timeout（見 `utils/llms.py`），不是 healthcheck。

  log 上限設 10MB × 5 檔，只寫在 `docker-compose.yml`（`start.ps1` 已改成呼叫 compose，
  不再自己帶參數 —— 設定只有一份，不會漂移）。本服務每個 request 都會把整包 response 連
  `trace` 一起印出來，不設上限磁碟遲早被吃滿。改了設定要 **recreate 容器**才生效，
  restart 不會套用，且舊 log 檔不會被回頭截斷。

  每個 request 會多印一行 `⏱️ /chat fb_account=... elapsed=..s pid=...`，
  量測點在 `main_webhook.py` 的 endpoint 層（`execute_backend_agent` 有多個提早 return，
  放外層用 `finally` 才每條路徑都涵蓋得到，連丟 500 也算得進去）。併發時 log 會交錯，
  用 `fb_account` 跟 `📥` / `📤` 那兩行對起來。

  ### volume 掛載

  `docker-compose.yml` 把三個向量庫目錄掛到容器外，重啟就不必重跑 embedding：

  ```yaml
  volumes:
    - ./chroma_clinic_qa:/app/chroma_clinic_qa
    - ./chroma_treatment_qa:/app/chroma_treatment_qa
    - ./chroma_token_split:/app/chroma_token_split
  ```

  **三個都要掛** —— 漏掉的那個每次啟動都會重新呼叫一次 embedding API。首次部署時目錄是空的，
  會自動建一次索引（服務就緒約 30 秒），之後啟動約 15 秒。

  BM25 的 pickle 刻意不掛：它只是 jieba 斷詞、沒有 API 呼叫，重建成本趨近於零，
  而 bind mount 單一檔案在 host 端檔案不存在時 Docker 會建成「目錄」，反而是坑。

  > 此設定假設 host 有持久的檔案系統（VM + docker compose）。若改用 Cloud Run / Fargate
  > 這類無狀態容器平台，bind mount 不會生效，每次冷啟都會重建索引。

  ### `start.ps1`（Windows 便利腳本）

  ```powershell
  .\start.ps1     # 檢查 .env → docker compose up -d --build → 跟 log
  ```

  它**只是 compose 的包裝**，中間走的就是 `docker compose up -d --build`。
  容器設定（port / env / volume / logging / restart）一律只寫在 `docker-compose.yml`。

  > 這支腳本原本是自己跑 `docker build` + `docker run`，把 name / port / env / restart
  > 又寫了一遍 —— 結果漏掉 compose 裡的三個 chroma volume，走這條路啟動每次都重跑
  > embedding，還讓「容器能不能啟動」綁死在 OpenAI 通不通上。同一份設定寫兩遍，
  > 遲早有一遍會漏，所以改成呼叫 compose。**不要再把參數搬回這支腳本裡。**

  > ⚠️ 檔案必須存成 **UTF-8 with BOM**。沒有 BOM 的話 Windows PowerShell 5.1 會用
  > cp950 解讀，中文變亂碼且會破壞字串解析 —— 症狀是腳本靜默跳過 build 步驟卻印出成功訊息。

  ---

  ## 觀測儀表板（Grafana，選用）

  ```bash
  docker compose -f docker-compose.grafana.yml up -d
  # http://localhost:3000  帳號 admin / 密碼見 compose 檔的 GF_SECURITY_ADMIN_PASSWORD
  ```

  獨立 stack（`name: clinic-grafana`），跟 AI 服務生命週期分開 —— AI 服務常 rebuild，
  Grafana 設好幾乎不動。資料源直接接**後端的 MySQL**（trace 存在 `message` 表），
  不需要 Prometheus / ClickHouse 之類的額外資料庫，整包只多一個約 200MB 記憶體的容器。

  資料源與儀表板都走 provisioning（`grafana/` 目錄），搬到別台機器只要複製檔案。

  寫查詢時有三個**會安靜給出錯誤答案**的坑，面板 SQL 都已處理：

  | 坑 | 症狀 | 處理 |
  |---|---|---|
  | `JSON_EXTRACT` 不轉型 | 變字串比較，`MAX` 可能小於 `AVG`，且不報錯 | 一律 `CAST(... AS UNSIGNED)` |
  | 一個請求兩列 | `message_type` 1／2 各一列、同一份 trace → 成本算兩遍 | 加 `message_type = 1` |
  | `created_at` 超前 8 小時 | 後端把台北時間寫進 UTC 連線的 TIMESTAMP 欄位，「最近 7 天」撈不到資料 | 用 `${tzfix}` 變數補償 |

  > 第三點是**後端的資料 bug**，不只影響儀表板 —— 任何用 `NOW()` 或時間範圍的查詢都會中。
  > 後端修正後把儀表板的 `tzfix` 變數改成 `0` 即可，不用改 SQL。

  > ⚠️ port 綁在 `127.0.0.1`，不要對公網開 —— 它後面接的是診所正式資料庫。
  > 正式環境請另建**唯讀且只授權 `message` 一張表**的 MySQL 帳號，別沿用 `CLINIC_*` 那組。

  ---

  ## 資料搜尋比對

  檢索分成**三組獨立索引**，各吃不同的 CSV。分開建是為了避免不同領域的資料灌進同一個索引
  互相污染排名 —— 例如「效果如何」這種問題在每個療程底下都成立，混在一起就分不出是哪一筆。
  每組都是「向量 + BM25」的 hybrid：

  | 索引 | 來源 CSV | 向量庫 | BM25 快取 | 權重（向量 : 關鍵字） |
  |---|---|---|---|---|
  | 療程介紹 | `clinics_introductions3.csv` | `chroma_token_split/` | `bm25.pkl` | 0.3 : 0.7 |
  | 療程 QA | `treatment_qa.csv` | `chroma_treatment_qa/` | `bm25_treatment_qa.pkl` | 0.7 : 0.3 |
  | 診所 QA | `clinics_qa.csv` | `chroma_clinic_qa/` | `bm25_clinic_qa.pkl` | 0.7 : 0.3 |

  - **向量檢索**：Chroma + OpenAI `text-embedding-3-small`
  - **關鍵字檢索**：BM25 + jieba 中文斷詞。建索引時把特定欄位重複寫入以加權
    （療程介紹的 `suitable_for` ×3；QA 的 `keywords` ×3、`category` ×2）

  權重方向不同是刻意的：療程介紹偏重關鍵字（0.7），因為使用者多半直接說出症狀詞；
  QA 偏重語意（0.7），因為同一個問題有很多種問法。

  ### 分店交通／停車：刻意不進 RAG
  `data/clinic_branch_info.csv` 走 `lookup_branch_info()` 靜態查表，**不建進任何索引**。
  「哪一家店、怎麼去、能不能停車」是事實型問題，用檢索很容易撈到隔壁分店的答案，查表才能保證對。

  ### 索引快取與更新
  索引由 `utils/index_cache.py` 以**來源指紋**（CSV 內容的 SHA-256 + 建索引邏輯版本）把關：

  | 情況 | 行為 |
  |---|---|
  | CSV 沒變 | 直接載入現有索引，不呼叫 embedding API |
  | CSV 有改 | 自動偵測並重建 —— **不需要手動刪快取** |
  | 索引目錄是空的（如 Docker volume 首次掛載） | 視為過期並重建，避免載入空 collection 後檢索靜默回 0 筆 |

  > ⚠️ 指紋只涵蓋 CSV 內容。如果你改的是 `Document` 的**組法**（欄位、加權重複次數）而 CSV 沒動，
  > 指紋不會變、索引不會重建。這時要把 `utils/qa_retriever.py` 或 `utils/ensemble_retriever.py` 裡的
  > `INDEX_LOGIC_VERSION` 字串 +1 才會強制重建。

  ---

  ## 🛡️ Guardrail（多層防護機制）

  從入口到出口共五層，任何一層都能攔下風險或轉真人客服：

  | 層級 | 位置 | 機制 | 失敗處置 |
  |---|---|---|---|
  | **① 入口守門** | `guard_node`（gpt-4o-mini） | prompt injection 偵測（`GuardResult`），揪出「忽略前述指令 / 洩漏 system prompt」等注入 | 直接婉拒並結束對話 |
  | **② 生成期白名單** | information Composer prompt | **療程白名單**：只能推薦白名單內且檢索有撈到的療程；白名單外（肉毒、玻尿酸、電波拉皮…）一律禁提，找不到就誠實說沒有 | 不推薦、不自編療程 |
  | **③ 費用工具化** | `get_treatment_fee`（booking） | 報價一律查工具（讀 `treatment_fees` 表），**嚴禁**從記憶／歷史自編價 | 查無方案就誠實說沒有 |
  | **④ 出口事實核對** | `moderator_node` faithfulness（gpt-4o） | grounding 切成**編號 chunk**（去重）；抽草稿裡的療程硬事實，每條標 **`source_id`（哪一號 chunk 支持它，無則 -1）** + 逐字 quote → 逐條**只對那一號 chunk**驗證：`-1` 判無依據、quote 逐字命中或 `SequenceMatcher ≥ 0.7` 放行、對不上（改寫過）→ 對該 chunk 做**語意蘊涵判斷** → 無依據就刪除／中性化 | 刪完無法回答 → `[[HANDOFF]]` → 轉真人 |
  | **④ 出口合規／語氣** | `moderator_node` cleaning | 移除誇大保證字眼（「一定會好」「完全消除」「治癒」）符合醫療法規；強制繁中；錯字修正 | — |
  | **⑤ 價格守門** | backend `_extract_prices` / `_rewrite_price_reply` | 抓回覆裡的 NT$ 價，出現「不屬本療程的價」→ 帶正確價**重寫一次**；仍錯 → 轉真人 | `price_fabrication` 轉真人 |
  | **轉真人** | `CS_KEYWORDS` | 客人主動要真人（轉專人／投訴／退費…）命中 → `CallCS=1`、清空 text/images | 立即轉接 |

  > 轉真人原因統一記在 `handoff_reason`：`customer_keyword` / `fact_check` / `price_fabrication` / `booking` / `null`。

  ---

  ## 📊 Eval（評估機制）

  分**線上**與**離線**兩軌。

  ### A. 線上 —— `trace` 內建評估軌跡
  每個 request 回傳一個 `trace`（`Optional[Dict]`），由 graph 各節點沿途寫入、backend 補完，**後端存起來即是一份 LLM-as-judge 用的 eval dataset**；後端不接完全不影響 `text` / `images` / `CallCS`。可觀測的環節：

  | 來源 | key | 看什麼 |
  |---|---|---|
  | guard | `guard` | 是否判為注入、理由 |
  | supervisor | `route` / `route_reasoning` | 路由對不對、為什麼 |
  | worker | `draft` / `grounding` | moderator 前草稿、本輪事實依據（有序去重的 chunk 清單） |
  | moderator | `final` / `moderator` | 最終文字；`{fact_check, unsupported_facts, force_handoff}` |
  | backend | `user_input` / `price_guard` / `handoff_reason` | 完整輸入（含 OCR）、價格守門過程、轉真人原因 |

  ### B. 離線 —— Replay + 回歸測試（`tests/`）
  確保「**改 code 後，原本答對的題目不會變答錯**」：

  - **`replay.py`**：把 `test_reply.txt` 的真實對話丟給 AI 跑一遍，產出「真人 vs AI」對照（`replay_result.md`）＋題庫草稿（`suggested_cases.yaml`），人工逐題標對／錯。
  - **`cases.yaml`**：人工挑出「AI 答對」的回歸題庫。`expect` 可驗 `call_cs` / `prices_present` / `prices_absent` / `text_contains` / `text_not_contains` / `images_include_any` / `images_empty` / `handoff_reason`。
  - **`test_regression.py`**：pytest 回歸，改完 code 就跑；**紅的那題 = 這次改動弄壞了**。

  ```powershell
  $env:PYTHONUTF8=1
  poetry run pytest tests/test_regression.py -v
  ```

  詳細測試流程見 [`tests/README.md`](./tests/README.md)。

  ---

  ## 詳細文件
  - 📐 [系統架構說明](./clinic_agent_architecture.md)
  - 🔌 [後端 API 串接文件](./API_INTEGRATION_GUIDE.md)
  - 🖼️ [圖片處理邏輯說明](./IMAGE_LOGIC_GUIDE.md)
  - 🗂️ [後端串接修改總結](./BACKEND_INTEGRATION_SUMMARY.md)
