# 後端串接總結（FB / Messenger AI 客服）

本文件說明後端工程師如何串接本醫美診所 AI 客服服務：單一端點 `POST /chat`、
請求 `BackendUserQuery`、回應 `BackendResponse`（`text` / `images` / `CallCS` / `trace`）。
服務為 **stateless**（無 checkpointer、無 session、無 DB），對話歷史每次隨 request 傳入。

> 更完整的內部架構（LangGraph 多代理、Retriever、防幻覺機制）請見 `clinic_agent_architecture.md`。

---

## 1. API 端點

| 項目 | 值 |
|---|---|
| Method / Path | `POST /chat` |
| Request model | `BackendUserQuery`（`app/models.py`） |
| Response model | `BackendResponse`（`app/models.py`） |
| Entrypoint | `main_webhook.py`（FastAPI, `root_path="/fb-clinics-agent"`） |
| Handler | `app/backend_agent_service.execute_backend_agent()` |
| Production | `POST https://ai.gastom.com.tw/fb-clinics-agent/chat` |
| Swagger UI | `https://ai.gastom.com.tw/fb-clinics-agent/docs` |
| 本地啟動 | `python main_webhook.py`（uvicorn, port 8004） |

識別碼為 **FB Messenger PSID**（`fb_account`）。**不再**使用 line_uuid，也**不再**從資料庫查歷史。

---

## 2. Request：`BackendUserQuery`

```json
{
  "fb_account": "1234567890abcdef",
  "content": "我想了解 NEO 療程",
  "image_url": [],
  "message_history": [
    {"type": "ai",    "content": "您好！我是霍普艾小編"},
    {"type": "human", "content": "你好"}
  ],
  "ad_referral": null,
  "treatment_fees": [
    {"name": "NEO-熱磁減脂(30分鐘) + 冷凍單點(60分鐘)", "price": 15999},
    {"name": "Emface 全臉拉提", "price": 8000}
  ]
}
```

| 欄位 | 型別 | 說明 |
|---|---|---|
| `fb_account` | str | Messenger PSID，識別客人（必填） |
| `content` | str / null | 客人此輪傳的文字（圖片訊息時為 null） |
| `image_url` | List[str] | 客人此輪傳的圖片（可多張；文字訊息時為空 list）。服務內部會 OCR |
| `message_history` | List[HistoryMessage] | 過去對話（不含本次），**由新到舊**（服務內部會自動反轉還原時序） |
| `ad_referral` | str / null | 客人從 Meta 廣告進來的療程關鍵字，建議只在首次對話（history 為空）時帶 |
| `treatment_fees` | List[TreatmentFee] | 目前各療程的最新體驗價（每次都傳完整列表；後端可加 cache） |

`HistoryMessage`：`{"type": "human" | "ai", "content": str}`
- `type="human"` = 客人；`type="ai"` = AI 或真人客服回覆。
- 若某則是真人客服回覆，請在 `content` 前加 `"[真人客服] "` 前綴，AI 才能區分自己與真人講過的話。

`TreatmentFee`：`{"name": str, "price": int}`
- `name` 可含時長 / 組合細節（如 `"NEO-熱磁減脂(30分鐘) + 冷凍單點(60分鐘)"`）。
- `price` 為體驗價純數字（int）。下架 / 非適用療程後端就不要送進來。

### treatment_fees 如何被使用（已改為「工具化」）
服務**不再**於 graph 前預先過濾費用、也不再注入 `[費用資訊]` SystemMessage。
每個 request 開始時會把**整張費用表原封不動**存入一個 request 範圍的 ContextVar（`treatment_fees_var`）。
booking 代理在對話中**認出療程的當下**才呼叫 `get_treatment_fee(療程名)` 工具即時查該療程的體驗價，
再由輸出端「價格守門」做最終把關。因此後端只要把最新完整費用表送進來即可，無須關心過濾邏輯。

---

## 3. Response：`BackendResponse`

```json
{
  "text": "NEO 是一款熱磁減脂療程...",
  "images": ["https://hopkins-main.s3.../neo_intro.jpg"],
  "CallCS": 0,
  "trace": { "route": "information_node", "draft": "...", "final": "...", "...": "..." }
}
```

| 欄位 | 型別 | 說明 |
|---|---|---|
| `text` | str | 回覆文字（`CallCS=1` 時為空字串） |
| `images` | List[str] | 圖片 URL 列表（`CallCS=1` 或 `2` 時為空 list） |
| `CallCS` | int | 0 / 1 / 2（見下表） |
| `trace` | dict / null | LLM-as-judge 評估用的內部軌跡；後端可存起來當 eval dataset。**未接前完全可忽略**，不影響 `text` / `images` / `CallCS` |

### CallCS 三值

| CallCS | 意義 | text / images | 後端動作 |
|---|---|---|---|
| 0 | 一般對話 | AI 回覆 + 相關圖片 | 直接推給客人 |
| 1 | 轉真人客服 | 空字串 / 空 list | 不推 AI 內容，直接通知真人客服 |
| 2 | 預約流程（客人已填完整預約資訊、觸發 `confirm_booking`）| AI 預約引導文字 / 空 list | 先推 text 給客人，再通知真人客服接手 |

`CallCS=1` 的觸發來源（任一即可，優先序見下）：
1. 客人主動找真人（訊息命中 `CS_KEYWORDS`，如「轉專人」「投訴」「退費」）。
2. Moderator 事實核對失敗（療程幻覺刪不掉 / 無法回答核心問題）→ `force_handoff`。
3. 輸出端價格守門重寫兩次仍報錯價 → 價格捏造 handoff。

`CallCS=2` 為 booking 代理呼叫 `confirm_booking`（客人本輪訊息含完整姓名+療程+時間+電話）後觸發。

---

## 4. `trace` 欄位內容（給後端存 eval dataset 用）

`trace` 由 graph 各節點累積寫入，backend 再補幾塊。後端不接不影響回覆契約。可能出現的 key：

| 來源 | key | 內容 |
|---|---|---|
| guard 節點 | `guard` | `{blocked, reason}` — prompt injection 守門結果 |
| supervisor | `route` / `route_reasoning` | 路由到哪個 worker 與理由 |
| worker 節點 | `draft` | 送進 moderator 前的草稿 |
| worker / 工具 | `grounding` | 本輪 retriever / 查表撈到、可當事實依據的原始內容清單 |
| moderator | `final` | 最終回覆文字 |
| moderator | `moderator` | `{fact_check, unsupported_facts, force_handoff}`（原文直出時多一個 `skip_moderation`）|
| backend | `user_input` | 含圖片 OCR 文字在內的完整輸入（後端手上只有原始 content）|
| backend | `price_guard` | **只有價格守門啟動時**才有：`{leaked, allowed, reply_treatments, retried, resolved, handoff}` |
| backend | `handoff_reason` | `customer_keyword` / `fact_check` / `price_fabrication` / `booking` / `null` |

---

## 5. 圖片 OCR 與純圖短路

- 客人傳圖時（`image_url` 非空），服務用獨立的 `gpt-4o-mini`（**無醫療 system prompt**）逐張 OCR，
  避開 OpenAI 對醫療影像的 safety filter；抓到的文字附加在客人原文後再進主 Agent。
- 圖片無文字（純臉照 / 純物件照）→ 濾掉 sentinel（`NO_TEXT_FOUND` 及「無 / 沒有 / 空字串」等變體）。
- **純圖、OCR 抓不到任何文字** → 不跑 LLM，直接回：`{"text": "請問您是想諮詢哪個部位呢？", "images": [], "CallCS": 0}`。
- 歷史訊息中的圖片後端**不需**傳，`message_history` 只帶 `content` 即可（上一輪 AI 回覆已內化圖片重點）。

---

## 6. Stateless 設計

- 無 LangGraph checkpointer、無 MemorySaver、無 session / cookie / DB。
- 對話記憶完全由後端在每次 request 的 `message_history` 帶入。
- 服務重啟 / 水平擴展不會掉資料；`fb_account` 僅作識別，不查任何內部資料表。

---

## 7. 快速測試

```bash
# 啟動
python main_webhook.py

# 呼叫（純文字）
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "fb_account": "test_user_001",
    "content": "我想了解減重療程",
    "image_url": [],
    "message_history": [],
    "ad_referral": null,
    "treatment_fees": [
      {"name": "NEO-熱磁減脂(30分鐘) + 冷凍單點(60分鐘)", "price": 15999}
    ]
  }'
```

---

## 8. 與舊版（line_uuid / `/execute`）的差異

| 項目 | 舊版 | 現況 |
|---|---|---|
| 識別碼 | line_uuid（DB 查歷史） | `fb_account`（PSID，不查 DB） |
| 對話記憶 | LangGraph MemorySaver（thread_id=uuid） | **無**，歷史隨 request 傳入（stateless） |
| Request | `line_uuid` + `messages` | `BackendUserQuery`（fb_account / content / image_url / message_history / ad_referral / treatment_fees） |
| Response | `text` + `images` | `BackendResponse`（text / images / CallCS / trace） |
| 費用處理 | graph 前 pre-filter 注入 `[費用資訊]` | 整表存 `treatment_fees_var`，`get_treatment_fee` 工具即時查 + 輸出端價格守門 |
| 轉真人 | 無 | `CallCS` 三值（客人關鍵字 / 事實核對 / 價格捏造 / 預約流程） |
| profile / format_user_profile_text | 有 | 已移除，不再組 profile 文字 |
