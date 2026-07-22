# Code Review：`line_service.py` → `backend_agent_service.py` 架構重構

> **審查日期：** 2026-05-06
> **審查範圍：** main 分支上未提交的全部變動（git diff HEAD）
> **審查者角色：** Code Reviewer（mentor 視角）
> **重構主要內容：**
> - 刪除 `app/line_service.py`
> - 新增 `app/backend_agent_service.py`
> - 修改 `agent.py`、`main_webhook.py`、`app/models.py`、`utils/profile_db.py`、`utils/llms.py`
> - 將使用者識別碼從 `phone_number` 改為 `line_uuid`

---

## 📋 整體印象

**重構意圖很對，但執行有幾個關鍵安全與正確性風險需要先處理。**

| 面向 | 評價 |
|------|------|
| 🎯 架構方向 | ✅ 將 LINE 事件處理 vs Agent 邏輯解耦是正確的 |
| 🧹 邏輯抽離 | ✅ `determine_additional_images` 取代亂麻 if-elif 是大進步 |
| 🆕 圖片擴展 | ✅ 動態 API 取圖（`search_clinic_images`）優於硬編碼 |
| 🛡️ 安全性 | 🔴 SQL injection、敏感資訊外洩、checkpoint thread 安全 |
| 🔄 並行/效能 | 🔴 sync 函式被 async endpoint 阻塞 |
| 🧪 測試 | 🟡 完全沒有測試 |
| 📚 留下的死碼 | 💭 大量註解區塊與備份檔，影響可讀性 |

---

## 🔴 BLOCKER — 必須修正

### 1. SQL Injection 漏洞（`utils/profile_db.py:45-56`）

```python
query = text(f"""
    SELECT ... WHERE LA.line_id = '{line_uuid}' ...
""")
```

**為什麼是問題：** `line_uuid` 是從 API 端 `BackendUserQuery` 直接傳進來的外部輸入，使用 f-string 拼接。雖然套了 `text()`，但 SQL 字串在送進 SQLAlchemy 前就已經被 f-string 插值完成，`text()` 只是再包一層。攻擊者送 `U' OR '1'='1` 就能拖庫；送 `U'; DROP TABLE line_account; --` 就能刪表。

**怎麼改：** 使用 SQLAlchemy 的 bind parameter：

```python
query = text("""
    SELECT LA.line_id, LA.line_name, LM.message, LM.created_at
    FROM line_account LA
    LEFT JOIN line_message LM ON LA.id = LM.line_account_id
    WHERE LA.line_id = :line_uuid
    ORDER BY LM.created_at DESC
    LIMIT 20
""")
resultDf = pd.read_sql(query, conn, params={"line_uuid": line_uuid})
```

---

### 2. SQL 查詢欄位缺失導致 KeyError（`utils/profile_db.py:46-75`）

SQL 沒 SELECT `LM.role`，但程式碼 `row['role']` 會直接拋 `KeyError`：

```python
SELECT LA.line_id, LA.line_name, LM.message, LM.created_at  # ← 沒 role！
...
'role': row['role'] if pd.notna(row['role']) else 'user',   # ← KeyError
```

**為什麼是問題：** 這個函數實際被呼叫到時就直接崩潰，但目前在 `backend_agent_service.py` 已經 comment out（line 9），所以才沒爆掉。如果未來真的要用這個欄位，會立刻壞。

**怎麼改：** 修正 SQL 加入 `LM.role`，或如果 `LM` 表沒有 role 欄位，根據實際 schema 調整。建議寫個 unit test 防止再犯。

---

### 3. SQLite Checkpointer 在 FastAPI 並行下不安全（`agent.py:18-21`）

```python
sqlite_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
sqlite_memory = SqliteSaver(sqlite_conn)
```

**為什麼是問題：** `check_same_thread=False` 只是讓 Python 不**檢查**，並不代表 SQLite **支援**並發寫入。在 FastAPI 多 worker / 多 thread 環境下，兩個請求同時 commit checkpoint 會引發 `database is locked` 或寫入競爭。線上一旦上量，會偶發超時、訊息遺失。

**怎麼改：** 三選一
- 改用 `AsyncSqliteSaver` + `aiosqlite`（LangGraph 有提供）
- 改用 PostgreSQL checkpointer（線上正解，pyproject.toml 已經有 `langgraph-checkpoint-postgres`）
- 在 connection 上加 lock，或開啟 SQLite WAL mode：
  ```python
  sqlite_conn.execute("PRAGMA journal_mode=WAL")
  ```

---

### 4. Sync 函式阻塞 async event loop（`main_webhook.py:12-33` × `backend_agent_service.py:158`）

```python
@app.post("/chat", response_model=BackendResponse)
async def backend_chat(query: BackendUserQuery):
    return execute_backend_agent(query)   # ← sync function with blocking I/O
```

**為什麼是問題：** `execute_backend_agent` 內含 LLM 呼叫（OpenAI API）、`requests.post`（影像 API）、SQLite 寫入——**全部都是阻塞 I/O**。在 async 端點裡直接呼叫 sync function 會卡住整個 event loop。一個慢請求 = 全部使用者卡住。

**怎麼改：** 用 `run_in_threadpool` 把 sync work 丟去 thread pool：

```python
from fastapi.concurrency import run_in_threadpool

@app.post("/chat", response_model=BackendResponse)
async def backend_chat(query: BackendUserQuery):
    return await run_in_threadpool(execute_backend_agent, query)
```

或更好：把 `execute_backend_agent` 改 async，內部呼叫 `httpx.AsyncClient` 取代 `requests`、用 `app_graph.ainvoke` 取代 `invoke`。

---

### 5. 強制覆蓋 `clean_text` 會丟失 AI 回覆（`backend_agent_service.py:241-245`）

```python
if parking_image in all_images:
    if "春光公園" not in clean_text:
        clean_text = "🅿️ 停車資訊：可以到走5分鐘的永春停車場..."
        # ⚠️ AI 原本說的內容全部被丟掉！
```

**為什麼是問題：** 這是 `=` 不是 `+=`。如果客人問「你們在哪裡？」，AI 會先回「我們在忠孝東路五段...」並包含停車場圖片 keyword 觸發停車邏輯——但如果 AI 回覆裡沒含「春光公園」字樣，**整段地址資訊就被替換成停車場一句話**，客人收到的等於完全不同的回覆。

**怎麼改：**

```python
if parking_image in all_images and "春光公園" not in clean_text:
    clean_text += "\n\n🅿️ 停車資訊：可以到走5分鐘的永春停車場..."
```

---

## 🟡 SUGGESTION — 應該修正

### 6. `thread_id` 從 `phone_number` 換成 `line_uuid` 是 breaking change

`backend_agent_service.py:213` 用 `line_uuid` 當 LangGraph thread_id，但 `checkpoints.sqlite` 裡舊資料是用 `phone_number` 當 key。所有舊使用者的對話狀態實際上都「失憶」了。

**怎麼改：** 在 deployment notes 標明這是預期行為，或寫個遷移腳本把舊 thread_id 對應到新 line_uuid。如果是 breaking change 而非 migration，至少加個 deprecation log。

---

### 7. AgentState 的欄位名稱與內容不符（`backend_agent_service.py:202`）

```python
query_data = {
    ...
    "phone_number": user_input.line_uuid,  # 命名地獄
}
```

**為什麼是問題：** 把 `line_uuid` 塞進名為 `phone_number` 的欄位，未來看 agent.py 的人會以為這真的是手機號碼。`agent.py:67-72` 的 `start_node` 也是塞 `phone_number`，supervisor 還會把它當「使用者個手機號碼」傳給 LLM：

```python
HumanMessage(content=f"使用者個手機號碼是: {state['phone_number']}"),
```

LLM 收到的會是 `使用者個手機號碼是: U1234abcd...`，這是一致性與隱私洩漏的雙重問題。

**怎麼改：** 把 `AgentState` 裡的 `phone_number` 改成 `user_id` 或 `line_uuid`，並更新 supervisor prompt 的措辭。

---

### 8. OCR 流程無法處理「真正的圖片諮詢」（`backend_agent_service.py:178-194`）

目前 OCR 只抽圖片中的「文字」。但醫美場景常見的是「客人傳自己的肌膚/體態照來問」——那種圖根本沒文字，OCR 會回空字串，圖片內容完全沒進到 LLM。

**怎麼改：** 兩條路
- (a) 直接傳 image URL 給 GPT-4o vision，讓它做諮詢判讀（要小心醫療 safety filter，如註解所述）
- (b) 用兩個獨立 LLM call：一個 OCR（純文字辨識）+ 一個 medical-aware vision（諮詢判讀），結果合併

目前的設計只覆蓋「文字截圖」這一種 case，**功能性 gap 應該在註解或文件明說**。

---

### 9. 全域 `agent` 實例的 LLM model 共用問題（`backend_agent_service.py:14`）

```python
agent = DoctorAppointmentAgent()  # module level
```

每個請求都呼叫 `agent.workflow()` 重新編譯 graph（line 168），但 LLM model 是 share 的（`agent.py:60`）。LangChain 的 ChatOpenAI 內部有 client、retry config 等狀態，並行呼叫**通常**安全，但若未來改用有狀態的 LLM（例如自訂 provider）會有風險。

**怎麼改：** 對於穩定性，建議 `app_graph` 也 cache 起來而不是每個 request 都 `workflow()`：

```python
# module level
agent = DoctorAppointmentAgent()
app_graph = agent.workflow()  # 只 compile 一次

def execute_backend_agent(...):
    response = app_graph.invoke(...)
```

---

### 10. `BackendChatMessage.image_url` 只接受單一 URL（`app/models.py:14-17`）

```python
class BackendChatMessage(BaseModel):
    role: str
    content: str
    image_url: Optional[str] = None  # 只能一張？
```

LINE 客人一次傳多張圖很常見，這個 schema 強制只能一張，前端需要拆成多筆 message。建議：

```python
image_urls: List[str] = []
```

---

### 11. 沒有任何測試（整個專案）

`line_service_test.py` / `profile_db_test.py` 是新加的，但實際看內容（line_service_test 與 line_service_old_backup 一模一樣）只是備份，不是測試。這次重構動到 SQL、API、State、Graph 流程，**完全沒有 unit test 或 integration test 保護**，未來改動會非常脆弱。

**怎麼改（最低限度）：**

- `tests/test_image_logic.py`：cover `determine_additional_images` 的所有分支（停車、自律神經、Emface、療程互斥邏輯）
- `tests/test_clean_text.py`：URL 提取與清理
- `tests/test_backend_agent.py`：mock LLM，驗證 message 轉換邏輯

詳細測試策略請見另一份文件 `TESTING_STRATEGY.md`（如有）。

---

### 12. 沒有 input validation（`app/models.py`）

```python
class BackendUserQuery(BaseModel):
    line_uuid: str  # 任何字串都收
```

LINE UUID 格式是 `U` + 32 hex chars。建議：

```python
from pydantic import Field
line_uuid: str = Field(..., pattern=r"^U[0-9a-f]{32}$")
```

訊息列表也應該有上限（防 DoS）：`messages: List[BackendChatMessage] = Field(..., max_length=100)`

---

## 💭 NIT — 加分項

### 13. 大量死碼

| 檔案 | 行數 | 動作 |
|------|------|------|
| `app/line_service_old_backup.py` | 308 行 | 完全重複，git 已有歷史，**刪除** |
| `app/line_service_test.py` | 308 行 | 與 backup 一模一樣，**刪除** |
| `backend_agent_service.py:252-278` | 28 行 | commented-out code，**刪除** |
| `agent.py:412-528` | 117 行 | commented-out code，**刪除** |
| `utils/llms.py:25-86` | 60 行 | commented-out code，**刪除** |
| `agent.py:31-42` | ~12 行 | comment-out 的 helper，**刪除** |

這些不是「保險」，是讓未來閱讀者困惑的雜訊。Git 是它們唯一該存在的地方。

---

### 14. Print 風暴改用 logging（全檔）

```python
print('444444444444444444444')         # backend_agent_service.py
print('uuuuuuuuuuuuuuuuuuuuuuuurrrrrrrrrrrr')
print('starttttttttttt pick an image')
print(f"執行 OCR 辨識圖片: {msg.image_url}")
```

**為什麼是問題：** 線上環境這些 print 全部會打到 stdout，污染 log，且無法依照 level 過濾。

**怎麼改：** 換成 `logging`，並把 debug print 改成 `logger.debug(...)` 之後線上預設 level 設 INFO 自動隱藏。

```python
import logging
logger = logging.getLogger(__name__)

# 取代所有 print
logger.debug("starting image picker")
logger.info("OCR processing image: %s", msg.image_url)
```

---

### 15. Magic strings 應該抽 constants

`backend_agent_service.py` 與 `agent.py` 充斥硬編碼的 S3 URL：

```python
"https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/parking_lots.jpg"
```

出現 N 次。S3 一旦改 bucket 或路徑，就要全文搜尋替換。建議集中到 `app/constants.py`：

```python
S3_BASE = "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS"
PARKING_LOTS_IMG = f"{S3_BASE}/parking_lots.jpg"
TREATMENT_PROCEDURE_IMG = f"{S3_BASE}/treatment_procedure.jpg"
AUTONOMIC_FEES_IMG = f"{S3_BASE}/autonomic_fees.jpg"
BODY_CONSULT_IMG = f"{S3_BASE}/body_consult.jpg"
EMFACE_INTRO_IMG = f"{S3_BASE}/emface_intro.jpg"
SKIN_PENCIAL_INTRO_IMG = f"{S3_BASE}/skin_pencial_intro.jpg"
NEO_INTRO_IMG = f"{S3_BASE}/neo_intro.jpg"
SIS_INTRO_IMG = f"{S3_BASE}/sis_intro.jpg"
```

---

### 16. 註解編號錯誤（`backend_agent_service.py:78, 84`）

```python
    # 1. 自律神經檢測相關
    if ...

    # 3. 停車場/地址相關        ← 跳號到 3
    elif ...

    # 3. 療程相關（使用 API ...） ← 又是 3
    else:
```

註解編號跳號重複，是搬移過程中沒整理。改成 1/2/3 一致。

---

### 17. Treatment 正規化邏輯在迴圈外重複處理（`backend_agent_service.py:93-97`）

```python
def normalize_treatment(t):
    if t == "冷凍": return "冷脈衝"
    if t == "瘦瘦筆": return "週纖達"
    return t
```

這個 mapping 應該抽出 module level，避免每次呼叫都重建 closure：

```python
TREATMENT_ALIAS = {"冷凍": "冷脈衝", "瘦瘦筆": "週纖達"}

def normalize_treatment(t: str) -> str:
    return TREATMENT_ALIAS.get(t, t)
```

---

### 18. 環境變數泄漏到 log（`line_service_old_backup.py:62-64` + 其他舊檔）

```python
print("LINE_CHANNEL_ACCESS_TOKEN =", LINE_CHANNEL_ACCESS_TOKEN)
print("LINE_CHANNEL_SECRET =", LINE_CHANNEL_SECRET)
```

**為什麼是問題：** Channel Access Token 是 LINE Bot 的長效憑證，被印到 log 等於洩漏。

新的 `backend_agent_service.py` 沒這問題（很好），但記得 backup 檔刪掉時順便確認沒任何環境別印過 secrets。

---

## ✨ 做得好的地方

讓我也誇獎一下這次重構中的好決策：

1. **`determine_additional_images` 抽出純函式**：之前那串 8 層 if-elif 真的不忍直視，現在有 in/out 邊界、可單元測試。
2. **動態 API 取圖**：用 embedding API 取代硬編碼 if-elif 是正確方向，未來新增療程不用改程式。
3. **`search_clinic_images` 有 timeout 與 score threshold**：避免下游服務掛掉時整個 chat 卡住，min_score=0.7 有實際守門意義。
4. **OCR 用獨立 LLM 實例避開醫療 safety filter**（line 145）：這是個有經驗才知道的細節，註解寫得很清楚——非常加分。
5. **回應結構化（`BackendResponse(text, images)`）**：解耦 LINE message 組裝邏輯，後端工程師串接乾淨太多了。

---

## 🎯 建議的修正順序

| 順序 | 任務 | 預估時間 |
|------|------|---------|
| 1 | 修 SQL injection（`profile_db.py`） | 15 min |
| 2 | 修 `clean_text` 覆蓋 bug（`backend_agent_service.py:241`） | 5 min |
| 3 | 修 `phone_number`→`line_uuid` 命名（agent.py + backend_agent_service.py） | 30 min |
| 4 | sync 阻塞改 `run_in_threadpool` | 10 min |
| 5 | SQLite WAL + 考慮 PG checkpointer | 1 hr |
| 6 | 刪除死碼（5 個檔案） | 15 min |
| 7 | 補基本 unit test（image logic） | 2 hr |
| 8 | OCR 加 vision 路徑 | 視需求 |

---

## 💬 結語

整體架構走向是對的——**把 LINE I/O 與 Agent business logic 拆開**這個決策值得鼓勵。但這次重構同時動了 ID schema、執行流程、checkpoint 後端、圖片來源——**改動面積有點大**，建議下次拆成兩個 PR：

1. 純粹搬程式（line_service → backend_agent_service，不動 schema）
2. 變更使用者識別碼（phone_number → line_uuid）

下一個 PR 我會優先看 SQL injection 與 clean_text 覆蓋這兩個 BLOCKER 是否修復。

---

## 📑 附錄：問題索引

### Blockers
1. SQL Injection 漏洞 — `utils/profile_db.py:45-56`
2. SQL 查詢欄位缺失 — `utils/profile_db.py:46-75`
3. SQLite Checkpointer 並行不安全 — `agent.py:18-21`
4. Sync 函式阻塞 async event loop — `main_webhook.py:12-33`
5. 強制覆蓋 `clean_text` 會丟失 AI 回覆 — `backend_agent_service.py:241-245`

### Suggestions
6. `thread_id` 從 `phone_number` 換成 `line_uuid` 是 breaking change
7. AgentState 欄位名稱與內容不符 — `backend_agent_service.py:202`
8. OCR 流程無法處理「真正的圖片諮詢」 — `backend_agent_service.py:178-194`
9. 全域 agent 實例 graph 重編譯 — `backend_agent_service.py:14`
10. `BackendChatMessage.image_url` 只接受單一 URL — `app/models.py:14-17`
11. 完全沒有測試
12. 沒有 input validation — `app/models.py`

### Nits
13. 大量死碼（5 個檔案）
14. Print 風暴改用 logging
15. Magic strings 應該抽 constants
16. 註解編號錯誤 — `backend_agent_service.py:78, 84`
17. Treatment 正規化 closure 應抽 module level
18. 環境變數泄漏到 log（舊備份檔）
