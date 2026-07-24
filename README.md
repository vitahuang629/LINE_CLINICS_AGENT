
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
  - **通道無關核心**：LangGraph agent 與 I/O 接層解耦，目前對外為 FB Messenger `/chat` API；LINE 接層程式碼保留待啟用

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

  ---

  ## 資料搜尋比對
  本專案使用 Retriever 結合兩種檢索方式：

  1. **向量檢索（Vector Retriever）**
     使用 Chroma + OpenAI Embeddings 將診所療程 CSV 資料轉成向量，搜尋最相關的療程內容，
     存放於 `./chroma_token_split`，第一次建立後會快取。

  2. **關鍵字檢索（BM25 Keyword Retriever）**
     使用 BM25 演算法對療程的 `name`、`suitable_for`、`keywords` 欄位加權搜尋，
     `suitable_for` 欄位加權三倍以提高匹配度，使用 pickle 快取於 `bm25.pkl`，第二次執行直接載入快取。

  最後將 Vector Retriever 與 BM25 Keyword Retriever 以權重 **[0.3, 0.7]** 組合，更偏向關鍵字匹配。

  > ⚠️ 若更新了療程 CSV 內容，需刪除 `bm25.pkl` 與 `chroma_token_split/` 讓索引重建，否則會載入舊快取。

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
