# 醫美診所 FB AI 客服 — 系統架構說明

本文件描述 `fb_clinics_agent` 專案的整體架構：從後端串接 (`/chat` API) 進入請求，經過 OCR 圖片預處理、LangGraph 多代理協作 (Supervisor → Information / Booking → Moderator)，到最後以「文字 + 圖片清單 + CallCS 標記」回傳的完整流程。

---

## 1. 高層流程概觀

```
[後端 / FB Messenger 整合層]
        │  POST /chat
        │  (fb_account + content/image_url + message_history + ad_referral + treatment_fees)
        ▼
┌──────────────────────────────────────────────┐
│  main_webhook.py  (FastAPI, root_path=       │
│   "/fb-clinics-agent")                       │
│  └─ /chat → backend_agent_service.execute_…  │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  backend_agent_service.py                    │
│   1. 反轉 message_history → LangChain msgs   │
│   2. OCR 圖片 (GPT-4o-mini, 無醫療 prompt)   │
│   3. 純圖無文字 → 短路回「請問哪個部位」     │
│   4. 整表存 treatment_fees_var (不預先 filter)│
│   5. 注入 ad_referral SystemMessage (首次)    │
│   6. 餵入 DoctorAppointmentAgent.workflow()  │
│   7. 輸出端「價格守門」(抓錯價→重寫→轉真人)  │
│   8. 從回覆抽圖 URL + 動態補圖 + 去重        │
│   9. 判定 CallCS (0/1/2) + 清空對應欄位      │
│  10. 清理文字 + 組 trace → BackendResponse   │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  agent.py  (LangGraph)                       │
│   start_profilo → guard → supervisor         │
│        ├─► information_node ─► moderator ─► END
│        ├─► booking_node     ─► moderator ─► END
│        └─► FINISH                            │
│   * information_node = Planner(InfoPlan)      │
│     booking_node    = ReAct + get_treatment_fee
│     moderator       = 事實核對 + 合規/語氣    │
│   * Stateless: 不使用 checkpointer，          │
│     歷史由後端隨 request 傳入                 │
│   * 各節點寫 trace（LLM-as-judge 評估軌跡）  │
└──────────────────────────────────────────────┘
```

---

## 2. 系統架構圖 (Mermaid)

![系統架構圖](./clinic_agent_architecture_diagram.png)

---

## 3. 對外 API 介面 (`main_webhook.py`)

| Method | Path | 說明 |
| --- | --- | --- |
| POST | /chat | 後端工程師串接用的主要端點，回傳 text + images + CallCS |

對外完整 URL（經 Nginx 反向代理）：

| 項目 | 值 |
|---|---|
| Production | `POST https://ai.gastom.com.tw/fb-clinics-agent/chat` |
| Swagger UI | `https://ai.gastom.com.tw/fb-clinics-agent/docs` |
| OpenAPI JSON | `https://ai.gastom.com.tw/fb-clinics-agent/openapi.json` |

FastAPI 透過 `root_path="/fb-clinics-agent"` 告訴自己處於前綴底下，確保 Swagger UI 抓 openapi.json 走對的 URL。

### Request (`BackendUserQuery` — `app/models.py`)
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
    {"name": "NEO-熱磁減脂(30分鐘) + 冷凍單點(60分鐘)", "price": 15999}
  ]
}
```

### 欄位說明

| 欄位 | 型別 | 用途 |
|---|---|---|
| `fb_account` | str | Messenger PSID，識別客人 |
| `content` | str/null | 客人此輪傳的文字（圖片訊息時為 null）|
| `image_url` | List[str] | 客人此輪傳的圖片陣列（可多張；文字訊息時為空 list）|
| `message_history` | List[HistoryMessage] | 過去對話，由新到舊（service 內部會 reverse 還原時序）|
| `ad_referral` | str/null | 客人從 Meta 廣告進來的療程關鍵字，只在首次對話時帶 |
| `treatment_fees` | List[TreatmentFee] | 目前各療程體驗價（後端從 DB / Sheets 即時抓送進來）|

### Response (`BackendResponse`)
```json
{
  "text": "NEO 是一款熱磁減脂療程...",
  "images": ["https://hopkins-main.s3.../neo_intro.jpg"],
  "CallCS": 0,
  "trace": { "route": "information_node", "draft": "...", "final": "...", "...": "..." }
}
```

| CallCS | 意義 | text / images 內容 | 後端動作 |
|---|---|---|---|
| 0 | 一般對話 | AI 回覆 + 相關圖片 | 推給客人 |
| 1 | 轉真人客服 | 空字串 / 空 list | 不推 AI 內容，直接通知客服 |
| 2 | 預約流程 (`confirm_booking` 觸發) | AI 預約引導文字 / 空 list | 先推 text 給客人，再通知客服 |

`CallCS=1`（轉真人）的觸發來源，任一即可：① 客人主動找真人（命中 `CS_KEYWORDS`）；
② Moderator 事實核對失敗（`force_handoff`）；③ 輸出端價格守門重寫兩次仍報錯價（價格捏造）。
`CallCS=2` 由 booking 代理呼叫 `confirm_booking`（客人本輪已填完整預約資訊）觸發。

### `trace` 欄位
`trace` 是給後端當 **eval dataset** 用的內部評估軌跡（型別 `Optional[Dict]`），
由 graph 各節點累積寫入、backend 再補幾塊。**後端不接 trace 完全不影響** `text` / `images` / `CallCS`。可能出現的 key：

| 來源 | key | 內容 |
|---|---|---|
| guard | `guard` | `{blocked, reason}` prompt injection 守門結果 |
| supervisor | `route` / `route_reasoning` | 路由到哪個 worker 與理由 |
| worker | `draft` / `grounding` | moderator 前的草稿；本輪可當事實依據的原始內容清單 |
| moderator | `final` / `moderator` | 最終文字；`{fact_check, unsupported_facts, force_handoff}`（原文直出時多 `skip_moderation`）|
| backend | `user_input` | 含圖片 OCR 文字在內的完整輸入 |
| backend | `price_guard` | **只有價格守門啟動時**才有：`{leaked, allowed, reply_treatments, retried, resolved, handoff}` |
| backend | `handoff_reason` | `customer_keyword` / `fact_check` / `price_fabrication` / `booking` / `null` |

### Stateless 設計
本服務**不保留任何客人狀態**：
- 沒有 LangGraph checkpointer
- 沒有 session / cookie / DB
- 對話歷史完全由後端從 DB 撈出後隨 request 帶入
- 重啟 / horizontal scale 不會掉資料

---

## 4. 預處理層 — 圖片 OCR

實作於 `backend_agent_service.ocr_image_with_llm()`：

* 用獨立 `ChatOpenAI(model="gpt-4o-mini", temperature=0)` 實例。
* **刻意不掛醫美 system prompt**，prompt 只要求「輸出圖片中的所有文字」，目的是**避開 OpenAI 對醫療類圖片的 Safety Filter**。
* 圖片無文字時 LLM 回傳 sentinel `NO_TEXT_FOUND`，service 端會濾掉（同時也濾掉「空字串 / 無 / 沒有文字」等常見變體，防止 LLM 字面解讀 prompt）。
* OCR 結果若有抓到文字，會以 `\n\n[客人上傳的圖片內容：]\n{...}` 形式附加在使用者原文後面，再丟進主 Agent。

### 短路機制
若客人**只傳圖、文字 OCR 抓不到任何內容**（純臉照 / 純物件照），**不跑 LLM** 直接回傳：

```json
{
  "text": "請問您是想諮詢哪個部位呢？",
  "images": [],
  "CallCS": 0
}
```

省 token 又避開 OpenAI 對人臉照的 safety filter。

---

## 5. 多代理工作流 (`agent.py`)

### 5.1 State 結構 (`AgentState`)
```python
class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    fb_account: str
    next: str
    query: str
    current_reasoning: str
    booking_completed: bool       # booking_node 偵測到 confirm_booking 被呼叫時設 True
    should_terminate: bool
    force_handoff: bool           # moderator 事實核對失敗 → 通知 backend 轉真人 (CallCS=1)
    skip_moderation: bool         # 原文直出（診所地址 / 療程介紹）時設 True → moderator 直通不改寫
    skip_fact_check: bool         # booking 回覆設 True → moderator 只做語氣/合規清理，跳過檢索式事實核對
    trace: Annotated[dict, merge_trace]  # 各節點累積寫入的 LLM-as-judge 評估軌跡
```

### 5.2 節點清單

| 節點 | 角色 |
| --- | --- |
| `start_profilo` | 初始化節點：注入 `fb_account`，導向 guard |
| `guard_node` | prompt injection 守門（輕量模型）：偵測到注入攻擊直接婉拒並結束，否則放行給 supervisor |
| `supervisor_node` | 「醫美診所經理」，用 `with_structured_output(Router)` 決定下一步 |
| `information_node` | **Planner 版**（非 ReAct）：輕量模型一次輸出結構化 `InfoPlan` → 確定性工具呼叫（同理查表 / 療程檢索 / 療程 QA / 療程介紹原文直出）→ 單次 Composer (gpt-4o) 生成 |
| `booking_node` | ReAct agent，處理預約、體驗價（`get_treatment_fee`）、初診費、診所資訊；偵測 `confirm_booking` 觸發後寫 `booking_completed=True` 給 backend 判定 CallCS=2；回覆設 `skip_fact_check=True` |
| `moderator_node` | 出口審查：有 grounding 時做事實核對 (faithfulness)，並統管法規 / 語氣 / 錯字，最後一道把關 |

### 5.3 Graph 連線
```
START → start_profilo → guard_node → supervisor
                                     ├─► information_node → moderator_node → END
                                     ├─► booking_node     → moderator_node → END
                                     └─► FINISH (AIMessage: 「感謝您的諮詢…」) → END
        * guard_node 攔截到 prompt injection → 直接 END（婉拒訊息）
```

### 5.4 持久化
**無**。Graph 用 `self.graph.compile()` 不帶 checkpointer。

---

## 6. Supervisor 路由規則

`supervisor_node` 用一段中文 system prompt + `with_structured_output(Router)` 強制輸出：
```python
class Router(TypedDict):
    next: Literal["information_node", "booking_node", "FINISH"]
    reasoning: str
```

判斷原則摘要：

| 條件 | 路由 |
| --- | --- |
| 描述症狀 / 詢問療程原理 / 改善方向（皮膚、體態、私密、睡眠神經）| `information_node` |
| 包含「費用 / 價錢 / 多少 / 初診 / 地址 / 電話 / 預約 / 改期 / 取消 / 時間」| `booking_node` |
| **包含「活動 / 優惠 / 促銷 / 檔期 / 方案 / 套裝 / 打折」**（醫美脈絡下 = 費用範疇）| `booking_node` |
| 「謝謝 / 沒事 / 沒問題了」等結束語 | `FINISH` |
| 與醫美無關（政治、法律、技術等）| 禮貌拒絕並導回醫美 |

---

## 7. Information Node（Planner 版，非 ReAct）

不再是 ReAct 決策迴圈。改為「**先規劃、再確定性執行、最後單次生成**」三階段，降低 gpt-4o 反覆往返的延遲與成本：

### 7.1 階段 1 — Planner（輕量模型，一次結構化輸出 `InfoPlan`）
用 `gpt-4o-mini` 一次判斷並輸出下列欄位（`with_structured_output(InfoPlan)`）：

| 欄位 | 用途 |
|---|---|
| `symptom_tag` | 歸一化標準標籤（皺紋類 / 私密療程 / 睡眠與神經 / 體態管理 / 皮膚其他 / ""）|
| `need_empathy` | 是否首次偵測新症狀、需要同理追問 |
| `need_search` | 是否需查療程資料庫（介紹 / 推薦）|
| `search_query` | 丟給 retriever 的**原始症狀詞**（空白分隔，不可用分類標籤）|
| `need_qa` | 是否在問某療程的具體問答（效果 / 修復期 / 會不會痛 / 副作用 / 與他者差別…）|
| `qa_treatment` / `qa_query` | 問答對應的療程名 / 問題本身 |
| `intro_treatment` | 客人要「介紹某**單一指名**療程 / 有沒有 X / X 是什麼 / 功效」時填療程名；症狀求推薦、比較、具體子問題則空 |

### 7.2 階段 2 — 確定性工具呼叫（零 / 極少 LLM）
依 plan 觸發，皆為程式化查表 / 檢索：
- **療程介紹「原文直出」**：`intro_treatment` 能對到 `TREATMENT_INTRO_ROWS`（`get_treatment_intro`，載自 `data/clinics_introductions3.csv`，**big5**）→ 直接把該療程官方介紹**一字不改**回給客人（前後加招呼與引導），並設 `skip_moderation=True`（moderator 直通不改寫）。杜絕 AI 自編英文全名 / 縮寫 / 原理。
- **同理素材**：`get_empathy_questions_by_symptom(symptom_tag)` 查內建字典（首次歸類到某症狀類別時原文短路直出，逐類別只發一次）。
- **療程檢索**：`search_clinics_by_keyword(symptom)` 呼叫 `ensemble_retriever`（`clinics_introductions3.csv`）；命中療程登錄進 `authorized_treatments_var` 與 grounding。
- **療程問答（treatment_qa）**：`need_qa` 時查 `treatment_qa_retriever`（見 §11）。
  - `qa_treatment` 空 → 用 `_resolve_qa_treatment_from_history`（掃歷史 + `TREATMENT_SYNONYMS`）確定性回補療程。
  - 檢索 query 以「問題為主」：`f"{qa_query} {treatment}"`（療程名只放一次、不 double），避免同療程多筆問答分不出來而撈到最泛那筆。
  - 檢索回來依 `category` 過濾成「本療程」候選（`_treatment_group` / `_cat_matches_group`）再取最相關那筆；回補路徑對不上就放棄 grounding，避免抓到別療程答案。

### 7.3 階段 3 — Composer（單次 gpt-4o 生成）
把階段 2 的同理素材 / 療程檢索 / 療程問答組成【系統提供資料】，單次生成回覆。關鍵 Prompt 規則：
- 介紹療程一律以檢索結果為準，**嚴禁**用訓練知識補英文全名 / 縮寫 / 原理；本輪未檢索則不介紹任何具體療程。
- 「療程名稱白名單」以外絕不推薦；先推薦再談價（用部位 / 目標問價時先推薦不報價）。
- 不主動誇大療效；提到療程用「可以幫助改善 / 有些人會選擇」這類保守語氣。
- 對比照用固定格式：`「這是[療程名稱]的對比照: <https URL>」`，下游才能正確抽圖。

---

## 8. Booking Node

- 模式：`create_react_agent`，tools = `set_appointment` / `confirm_booking` / `search_clinics_info` / `get_treatment_fee`。
- 節點入口先做一次**診所靜態資訊「原文直出」**：`clinic_info_direct_answer()` 判斷客人問的是地址 / 停車 / 看診時間 / 電話且指明分店（台北信義 / 竹北）→ 直接回 `CLINIC_INFO_ROWS`（載自 `clinics_qa.csv`）該筆原文、設 `skip_moderation=True`，**不進 LLM**，杜絕地址幻覺；非此類才交給 ReAct agent。
- 工具：
  - `set_appointment(symptom)`：客人**首次**表達預約意願時呼叫，回傳制式預約欄位表單（姓名 / 療程 / 時間 / 特殊需求 / 電話）。**只顯示表單，不觸發轉真人**。
  - `confirm_booking(name, treatment, datetime_pref, contact, special_needs)`：客人**本輪訊息已含完整預約資訊**時呼叫。**呼叫此 tool 後 booking_node 會寫 `booking_completed=True` 到 state**，backend 端據此設 CallCS=2（轉真人）。
  - `get_treatment_fee(treatment_name)`：查該療程的**體驗價**（讀 `treatment_fees_var` 整張費用表）。用 `TREATMENT_SYNONYMS` 別名群組比對 fee name → 回單做 + 組合方案（含組合防呆），並 `register_grounded_content`。查無方案 / 未指定明確療程時回明確訊息（叫 AI 反問或誠實說沒有，**嚴禁編價**）。詳見 §8 費用查詢分流。
  - `search_clinics_info(treatment_name, category)`：依 `category` 分流：
    1. 含「初診 / 諮詢」→ `get_consult_info` 查 `data/consult_plan.csv`（結構化，免費 / 收費看 `consult_free` 欄）。
    2. 「健保 / 保險 / 理賠 / 自費」政策問題 → 呼叫 `search_clinics_info("診所", "健保")`（保險理賠類用 category「保險」），走 clinic_qa（見 §8 健保 / 保險路由）。
    3. 其餘（地址 / 停車 / 付款 / 預約流程等診所交易型 FAQ）→ 用 `category` 做 boosted query 走 **`clinic_qa_retriever`**（`treatment_name` 只在第 1 路用到）。
- 關鍵 Prompt 規則：

### 費用查詢有兩種，需嚴格分流

| 客人問 | 路徑 | 來源 |
|---|---|---|
| **初診費 / 諮詢檢測費 / 第一次來多少** | 呼叫 `search_clinics_info(name, "初診")` → `get_consult_info` 結構化查表 | `data/consult_plan.csv`（免費/收費由 `consult_free` 欄決定，不用語意檢索） |
| **療程體驗價 / {療程}多少錢 / 單次費用** | 呼叫 **`get_treatment_fee(療程名)`**（讀 `treatment_fees_var`）| `treatment_fees` (request payload，整表原封不動存入) |

> ⚠️ 免費與否一律由 `consult_plan.csv` 的 `consult_free` 欄決定，**提示詞不再主動寫「免費」**（見 §16.1）。

### 報體驗價時必須呼叫 get_treatment_fee，並同時呼叫 search_clinics_info("X","初診")
Booking prompt 已改：報體驗價「**一定呼叫 `get_treatment_fee(療程名)`**」，價格一律以工具回傳為準、**嚴禁**自己從記憶 / 歷史編價；再搭配初診評估整合回客人，例如：

> 「NEO 熱磁減脂搭配冷凍的組合有兩種方案：搭配單點 NT$ 15,999、搭配雙點 NT$ 18,999。
> 療程前我們會先為您安排諮詢檢測評估，會檢測皮下脂肪、內臟脂肪、肌肉量、基礎代謝率等指數...」

### 防幻覺：體驗價「工具化」+ 輸出端「價格守門」
舊的「graph 前 pre-filter 注入 `[費用資訊]` SystemMessage」機制已**移除**（`identify_treatments_from_context` / `filter_fees_by_treatments` 不再用於注入；`filter_fees_by_treatments` 仍被價格守門用到）。改為兩層：

**① 體驗價工具化 `get_treatment_fee`（toolkits.py）**
- backend 每個 request 開始時 `treatment_fees_var.set(user_input.treatment_fees)`，把**整張費用表原封不動**存入（不預先 filter）。
- booking 的 ReAct agent 在**認出療程的當下**呼叫 `get_treatment_fee(療程名)`：讀 `treatment_fees_var` → 用 `TREATMENT_SYNONYMS` 別名群組比對 fee name → 回單做 + 組合方案（含組合防呆）→ `register_grounded_content`。查無方案 / 未指定療程時回明確訊息，嚴禁編價。
- 原因：舊 graph 前的弱解析對代名詞 / 序數（「這個臉部」「第一個療程」）常失敗 → 相關費用空掉 → AI 沒價可用而編價。改用 booking 認出療程當下的可靠解析。

**② 輸出端「價格守門」（`backend_agent_service.py`，組 BackendResponse 前）**
- 抓 AI 回覆裡的 NT$ 價格數字，若出現「**不屬本療程的價格**」（別療程真實價或捏造價，不在合法價集合內）→ 讓 AI 帶正確價「**重寫一次**」；重寫後仍錯（或回 `[[HANDOFF]]`）→ `force_handoff`（CallCS=1，`handoff_reason=price_fabrication`）。
- 合法價集合：從 AI 回覆掃出的療程（`_resolve_reply_treatments`，用 `TREATMENT_SYNONYMS`）回查真實 fees（`filter_fees_by_treatments`）∪ 本輪 grounding 出現過的價格。
- 相關 helper：`_extract_prices` / `_resolve_reply_treatments` / `_rewrite_price_reply`（重寫用獨立 gpt-4o 實例）。trace 補 `price_guard` 欄位（僅守門啟動時）。

### 健保 / 保險路由
- supervisor：「健保 / 保險 / 理賠 / 自費」路由到 **booking_node**（原本誤入 information_node，那邊只查 treatment_qa 查不到；答案在 clinics_qa）。
- booking：`search_clinics_info` 可處理清單加入「健保 / 保險政策問題」→ 呼叫 `search_clinics_info("診所", "健保")`（保險理賠類用 category「保險」），走 clinic_qa。（並移除原本自相矛盾、寫「能否申請保險不由此工具處理」的錯誤說明。）

### 預約兩階段規則（set_appointment / confirm_booking）
- 客人**首次表達預約意願** → 呼叫 `set_appointment`（只發表單，不轉真人）；填完後**不要再呼叫**（避免重發表單模板）。
- 客人**本輪訊息已含完整資訊**（姓名 + 療程 + 時間 + 電話）→ 呼叫 `confirm_booking`（觸發轉真人 / CallCS=2）；歷史裡有過完整資訊但本輪在問別的事 → **不可**回頭 submit。

---

## 9. Moderator Node — 出口審查

每次 `information_node` 或 `booking_node` 的回覆都會經過 `moderator_node`。

### 9.1 共同規則（合規 / 語氣 / 語言，不論有無檢索內容）
1. **錯字修正**：例如 `NEOT → NEO`；但**嚴禁**把「猛健樂 / 瘦瘦筆 / 週纖達」自動改成 `EMBODY`。
2. **法規 / 語氣**：移除誇大、保證性字眼（「即時效果」「一定會好」「完全消除」「治癒」）。
3. **語言一致性**：強制繁體中文，僅特定療程名（NEO, EMBODY）保留英文，其他 fat/muscle 等翻成中文。
4. 原封不動保留所有網址 / 圖片連結 / Markdown / emoji / 換行；直接回傳純文字，不能加「這是修改後版本」之類前言。
- 本輪沒檢索到療程內容（純預約 / 問地址 / 閒聊）時，只做上述清理（用 `gpt-4o-mini`）；清理 API 失敗 → 保留原草稿並 `force_handoff=True`（最保守）。

### 9.2 事實核對 (faithfulness)：`do_fact_check = bool(grounded) and not skip_fact_check`
本輪 retriever / 查表撈到「事實來源」（`grounded_content_var`）且非 booking route 時，走 `_fact_check_and_clean`（強模型 gpt-4o）。`grounded_content_var` 是「有序、去重的 chunk 清單」——`register_grounded_content` 把每份撈到的內容按 `\n\n---\n\n` 切成 chunk、只留首次出現（chunk 級去重，消掉同一段原文經不同管道重複塞入的重疊）：
1. sources 以 `[n]` 編號逐段送給 LLM。LLM 抽出草稿裡的療程硬事實（英文全名 / 縮寫 / 原理 / 數據 / 機器品牌名 / 是否提供某療程），每條標 **`source_id`（哪一號 chunk 支持它，無則 -1）** + 從該 chunk **逐字複製的 quote**，並同時輸出合規清理版。
2. 程式驗證，每條縮到它那一號 chunk：`source_id=-1` → 無依據；quote 對該 chunk 逐字命中或 `SequenceMatcher` 涵蓋率 ≥ 0.7 → 過；對不上（多半被改寫過）→ 只對**該 chunk** 做語意蘊涵判斷（`_entail_unsupported`）。作用域縮到單一小 chunk，逐字與語意兩關都又快又準，且不會被別段的字誤命中。
3. 有無依據的事實 → 二次改寫刪除 / 中性化；刪完已無法回答核心問題 → 回 `[[HANDOFF]]` → `force_handoff`（backend 據此設 CallCS=1，`handoff_reason=fact_check`）。
- 組 sources 時會把 `authorized_treatments_var`（本輪從診所 DB 合法檢索到的療程）補一行「本診所有提供以下療程：…」進來源，讓「我們有提供 SIS」這種 claim 對得到依據（否則介紹文裡沒有「我們提供」字樣會被誤砍）。
- 費用（`get_treatment_fee`）與 consult（初診）內容也會 `register_grounded_content`，正確價格 / 初診描述不會被事實核對誤砍。

### 9.3 兩個直通旗標
- **`skip_moderation`**：原文直出（診所地址、療程介紹原文）時 moderator **完全直通、不改寫**（保護門牌 / 官方介紹）。
- **`skip_fact_check`**：**booking route** 的回覆設 True → moderator 只做語氣 / 合規清理、**跳過檢索式事實核對**（booking 內容來自費用表 / consult 表等確定性來源＋已有價格守門，不該拿去跟療程介紹檢索比對，否則正確價格 / 初診描述 / 框架句會被誤判無依據而轉真人）。

---

## 10. 後處理 — 動態圖片決策、去重與 CallCS 判定

回覆從 LangGraph 出來後，`backend_agent_service.execute_backend_agent` 會做：

### 10.1 圖片決策 (`determine_additional_images`)
1. `extract_image_urls(text)` — 用 regex 撈出 `https://*.{jpg,jpeg,png,webp}`。
2. 依以下優先順序補圖：

   | 條件 | 補的圖 |
   | --- | --- |
   | 回覆包含「兩種方案」或「自律神經檢測」 | `treatment_procedure.jpg` + `autonomic_fees.jpg` |
   | 回覆談「地址 / 地點 / 位於」或使用者問「哪裡 / 停車 / 開車」 | `parking_lots.jpg` |
   | 否則：偵測 `Emface / NEO / SIS / 瘦瘦筆 / 週纖達 / EMBODY / 體態檢測 / 冷凍 / 冷脈衝` | 呼叫 `clinics_image_embedding_api` (score ≥ 0.7) 取對應療程圖 |

   特殊規則：
   - `冷凍 → 冷脈衝`、`瘦瘦筆 → 週纖達` 在搜圖前先正規化。
   - **Emface 過濾**：若回覆已含圖 URL 或提到「電波」，跳過補圖。
   - **互斥療程清單** (`NEO / SIS / 週纖達 / EMBODY / 冷脈衝`)：同一回覆出現 2 個以上時，全都不補介紹圖（避免一次塞太多療程圖）。

### 10.2 圖片去重
從 `message_history` 中所有 `type=="ai"` 的訊息抓出已發過的圖 URL，本輪要送的 `images` 過濾掉這些 URL → **同一張圖在歷史窗口內不會反覆發送**。

### 10.3 CallCS 判定
依序套用（`CS_KEYWORDS` 命中為最高優先，會覆蓋前面的 handoff）：

1. **`force_handoff=True`**（moderator 事實核對失敗）→ `CallCS=1`
2. **價格守門重寫兩次仍錯**（`price_guard_handoff`）→ `CallCS=1`
3. **CS_KEYWORDS 命中** (`user_query` 含「真人客服 / 轉專人 / 投訴 / 退費」等；片語見 §16.4) → `CallCS=1`，text/images 清空（最高優先）
4. **`booking_completed=True`** (booking_node 偵測到 `confirm_booking` 被呼叫) → `CallCS=2`，保留 text、images 清空
5. **以上都沒** → `CallCS=0`

對應 `trace.handoff_reason`：`customer_keyword`（3）/ `fact_check`（1）/ `price_fabrication`（2）/ `booking`（4）/ `null`。

### 10.4 文字後處理
1. `clean_text_from_urls(text, urls)` — 移除正文中的圖片引用，四步：①markdown 圖片語法 `![alt](url)`/`![alt]()`/`![alt]` ②裸 URL ③只剩 `<圖片網址N>:`/條列/標點的殘骸行 ④收多餘空行（見 §16.5）。
2. **停車補丁**：當最終 images 含 `parking_lots.jpg` 但正文沒提到「春光公園」，會強制覆蓋為固定的停車說明。

---

## 11. Retrievers

全部 Retriever 都是 **Vector + BM25 的 EnsembleRetriever**，共用 `utils/shared_resources.py` 的 embedding model 與中文 jieba tokenizer。
共 **兩套 builder、三份互不污染的索引**：療程介紹（`utils/ensemble_retriever.py`）＋ QA builder 建出的 clinic_qa / treatment_qa 兩份（`utils/qa_retriever.py`）。

> ⚠️ 三份索引都在 `toolkit/toolkits.py` **模組載入時**建立（import 副作用）——`import toolkit.toolkits` 當下就會讀 CSV、連 Chroma。Dockerfile 的 warmup 步驟（§16.6）正是利用這點。

### 11.1 `ensemble_retriever` — 療程介紹（給 information_node）
- 來源：`data/clinics_introductions3.csv`（big5）。
- 兩份文件視角：
  - vector：原文。
  - keyword：把 `suitable_for` 重複 3 次以加強 BM25 對「適合對象」的命中。
- 持久化：Chroma → `./chroma_token_split`，BM25 → `./bm25.pkl`。
- 權重：`vector 0.3 / keyword 0.7`，k = 5 (vector) / 3 (BM25)。

### 11.2 QA Retriever（`get_qa_retriever`，共用 builder / 兩份獨立索引）
同一段建索引邏輯 (`utils/qa_retriever.py`)，靠不同 `persist_dir` / `bm25_path` 建出**兩份互不污染**的索引；builder 新增 `k` 參數（BM25 pickle 快取載入後會強制 `bm25_retriever.k = k`，不必刪快取重建）。權重固定 `vector 0.7 / keyword 0.3`，keyword 文件把 `keywords` 欄重複 3 次、`category` 加權 ×2。

| 索引 | 來源 CSV | persist / BM25 | k | 使用者 |
|---|---|---|---|---|
| `clinic_qa_retriever` | `data/clinics_qa.csv` | `./chroma_clinic_qa` / `./bm25_clinic_qa.pkl` | 2 | booking_node 的 `search_clinics_info`（診所交易型 FAQ：地址 / 付款 / 預約流程 / 健保 / 保險…）|
| `treatment_qa_retriever` | `data/treatment_qa.csv` | `./chroma_treatment_qa` / `./bm25_treatment_qa.pkl` | 6 | information_node 的療程問答（效果 / 修復期 / 會不會痛…，`category`=療程名）|

- **treatment_qa** 檢索 query 以「問題為主」`f"{qa_query} {treatment}"`（療程名只放一次），回來再依 `category` 過濾成「本療程」候選（`_treatment_group` / `_cat_matches_group`）取最相關那筆；`qa_treatment` 空時先用 `_resolve_qa_treatment_from_history` 回補（見 §7.2）。
- **clinic_qa**：`search_clinics_info` 的 query boosting 為 **`"{category} {category} {category}"`**（不灌療程名）；初診 / 諮詢與地址 / 停車 / 電話等會在進入 retriever 前先被結構化查表 / 原文直出攔截（§16.1、§16.3、§8），實際走到 `clinic_qa_retriever` 的多為其餘長尾 FAQ（含健保 / 保險政策）。

---

## 12. 模型與外部依賴

| 用途 | 模型 / 服務 |
| --- | --- |
| 主 LLM (`utils/llms.py`) — information Composer / booking ReAct / moderator 事實核對 | `ChatOpenAI("gpt-4o")` |
| OCR | `ChatOpenAI("gpt-4o-mini", temperature=0)` 獨立實例 |
| guard / supervisor 路由 / moderator 語氣清理 / information Planner | `ChatOpenAI("gpt-4o-mini")` 各自實例 |
| 價格守門重寫 (`_rewrite_price_reply`) | `ChatOpenAI("gpt-4o", temperature=0)` 獨立實例 |
| Embedding | `utils/shared_resources.embedding_model` |
| 圖片檢索 API | `https://ai.gastom.com.tw/clinics_image_embedding_api/api/search` (POST) |
| 對話持久化 | **無**（由後端 AWS DB 負責，每次 request 傳 message_history 進來）|

> 註：舊的「代名詞解析 (find_relevant_fees Stage 2)」LLM 已隨費用 pre-filter 注入機制移除，改為 booking 認出療程時呼叫 `get_treatment_fee`（純查表、零 LLM）。

---

## 13. 重要目錄結構

```
fb_clinics_agent/
├── Dockerfile                    # Multi-stage build (poetry install)
├── docker-compose.yml            # 本地起服務用
├── .dockerignore
├── main_webhook.py               # FastAPI entrypoint (/chat, root_path="/fb-clinics-agent")
├── agent.py                      # LangGraph 多代理 (Supervisor / Info / Booking / Moderator)
├── app/
│   ├── backend_agent_service.py  # OCR + pre-filter + 動態圖片 + CallCS 判定
│   └── models.py                 # BackendUserQuery / HistoryMessage / TreatmentFee / BackendResponse
├── toolkit/
│   └── toolkits.py               # set_appointment / confirm_booking / search_clinics_info / get_treatment_fee /
│                                 #   search_clinics_by_keyword / get_empathy_questions_by_symptom / get_treatment_intro
│                                 #   ContextVar: authorized_treatments_var / grounded_content_var / treatment_fees_var
├── utils/
│   ├── llms.py                   # LLMModel (預設 gpt-4o；可傳 "gpt-4o-mini")
│   ├── ensemble_retriever.py     # 療程介紹 (vector+BM25)
│   ├── qa_retriever.py           # QA builder get_qa_retriever(k=...)：clinic_qa + treatment_qa 兩份索引
│   ├── consult_plan.py           # 初診/諮詢費結構化查表 (get_consult_info)
│   └── shared_resources.py       # embedding model + 中文 tokenizer
├── prompt_library/prompt.py      # 早期 supervisor system prompt (現已內嵌於 agent.py)
├── data/
│   ├── clinics_introductions3.csv  # 療程介紹 (big5)；也供 get_treatment_intro 原文直出
│   ├── clinics_qa.csv              # 診所交易型 QA + 分店靜態資訊 (原文直出)
│   ├── treatment_qa.csv            # 療程內容問答 (category=療程名)
│   └── consult_plan.csv            # 各療程初診費 (treatment/consult_free/consult_fee/plans)
├── chroma_token_split/           # 療程介紹 vector store (runtime 用 CSV 重建，不進 image)
├── chroma_clinic_qa/             # clinic_qa vector store (runtime 重建)
├── chroma_treatment_qa/          # treatment_qa vector store (runtime 重建)
└── bm25.pkl / bm25_clinic_qa.pkl / bm25_treatment_qa.pkl   # BM25 retriever 快取 (runtime 自動產生)
```

---

## 14. 一次完整請求的時序

```
Backend ─► POST /chat (fb_account, content/image_url, message_history, ad_referral, treatment_fees)
        │
        ▼
backend_agent_service.execute_backend_agent
  1. reset context vars: authorized_treatments_var / grounded_content_var；
     treatment_fees_var.set(整張 treatment_fees)  # 不預先 filter
  2. for each msg in message_history (reversed):  轉成 LangChain HumanMessage/AIMessage
  3. for each url in image_url:  ocr_image_with_llm()  (gpt-4o-mini, no medical prompt)
     → 過濾 NO_TEXT_FOUND / "空字串" 等假文字
  4. 純圖無文字 → 短路回「請問哪個部位」（不跑 LLM）
  5. (首次對話有 ad_referral 時) 注入 referral_note SystemMessage
  6. agent.workflow().invoke(query_data, config={recursion_limit: 20})  # 無 thread_id
        │
        ▼
    LangGraph
      start_profilo → guard_node（注入攻擊 → 直接 END 婉拒）→ supervisor
        Router(next, reasoning)
          ├─ information_node  (Planner InfoPlan → 確定性工具 → Composer gpt-4o)
          ├─ booking_node      (ReAct: set_appointment / confirm_booking /
          │   │                        search_clinics_info / get_treatment_fee)
          │   └─ 若呼叫 confirm_booking → state["booking_completed"]=True；回覆設 skip_fact_check
          └─ FINISH (固定收尾語)
        → moderator_node (有 grounding → 事實核對；統管法規/錯字/中文化；skip_moderation 直通)
        → END
        （各節點沿途寫入 state["trace"]）
        │
        ▼
  7. force_handoff 讀出 (moderator 事實核對失敗)
  8. 價格守門：_extract_prices → 抓到不屬本療程的價 → _rewrite_price_reply(gpt-4o) 重寫一次；仍錯 → handoff
  9. extract_image_urls(text)
  10. determine_additional_images(...)  → 視內容呼叫 image embedding API
  11. 圖片 dedup：從 message_history 撈出已發過的 URL，本輪過濾
  12. CallCS 判定 (force_handoff → 1 / price_guard → 1 / CS_KEYWORDS → 1 / booking_completed → 2 / else → 0)
      - CallCS=1：text/images 清空
      - CallCS=2：images 清空，保留 text
  13. clean_text_from_urls(...) + strip_markdown(...)
  14. 組 trace：補 user_input(含 OCR) / price_guard / handoff_reason
        │
        ▼
BackendResponse(text, images, CallCS, trace) ─► Backend
                                          ├─ CallCS=0：推給客人
                                          ├─ CallCS=1：不推 AI 內容，通知客服
                                          └─ CallCS=2：先推 text，再通知客服
```

---

## 15. 設計上的幾個關鍵決定

1. **Stateless AI service**：所有對話狀態由後端 AWS DB 管理，AI service 重啟不掉資料、可水平擴展。
2. **OCR 與主 Agent 解耦**：避開 OpenAI 對醫療影像的 Safety Filter，OCR 用無領域 prompt 的乾淨模型實例。
3. **意圖路由用 structured output**：`Router` TypedDict 強制 LLM 只能回 `information_node / booking_node / FINISH`，不會出現自由發揮。
4. **Moderator 作為最後一道閘**：所有對外文字都經過合規 / 語氣審查，把醫療法規風險集中處理在單點。
5. **體驗價走工具、初診費走結構化查表**：費用本質上是兩種來源（動態 `treatment_fees` vs 靜態 `consult_plan.csv`），分流防止 AI 混淆；體驗價改由 booking 認出療程時呼叫 `get_treatment_fee`（讀整表、當下才 filter）；初診費的免費/收費由 `consult_free` 欄明確決定，不再靠語意檢索推論（見 §16.1）。
11. **固定事實不賭模糊檢索**：診所地址/交通/電話/看診時間/停車改為分店 + 主題判斷即原文直出，初診費改為結構化查表——短查詢用向量/BM25 排名不穩，固定事實一律走確定性查表（見 §16.3）。
6. **費用防幻覺＝工具化 + 價格守門**：移除舊的 graph 前 pre-filter 注入 `[費用資訊]`（弱解析對代名詞/序數易失敗導致 AI 編價）；改由 `get_treatment_fee` 在 booking 認出療程當下可靠 filter，再由輸出端價格守門抓「不屬本療程的價」重寫 / 轉真人（見 §8）。
7. **圖片不交給 LLM 自由生成**：療程圖以「關鍵字判斷 + 對外 embedding API」決定，避免 LLM 幻覺出不存在的 URL；對比照才允許在 prompt 裡用固定格式內嵌。
8. **圖片去重靠 history**：從 message_history 反推已發過的 URL，避免 AI 反覆推同一張圖騷擾客人。
9. **Vector + BM25 雙路檢索**：療程介紹偏 BM25（命中具體療程名 / 適合對象），QA 偏 vector（語意相近的問法），權重分別調校。
10. **CallCS 三值設計**：把「客人主動找真人 (1)」跟「AI 觸發預約流程 (2)」分開，後端可以對兩種情境採取不同的客戶體驗（前者立刻轉接、後者先讓客人看到表單再轉接）。

---

## 16. 變更記錄 (2026-06-10)

本次調整聚焦三類問題：**初診費「免費」誤判**、**診所資訊／療程指代檢索不準**、**圖片殘骸與轉真人漏接**，並修掉部署時的快取 stale 問題。

### 16.1 初診 / 諮詢費 → 結構化查表（取代語意檢索）
- 新增 `data/consult_plan.csv`（`treatment, consult_free, consult_fee, plans`，一療程一列，免費/收費是明確欄位）。
- 新增 `utils/consult_plan.py`：`get_consult_info(treatment_name, synonyms)` 以療程名查表（同義詞群組優先、子字串次之），同義詞表由呼叫端傳入避免循環 import。
- `search_clinics_info`：`category` 含「初診/諮詢」時先查 consult_plan，查到直接回、查不到才 fallback `qa_retriever`。
- 從 `clinics_qa.csv` 移除各療程初診費 row（單一來源）；移除提示詞/模板主動寫「免費」，免費與否一律由 `consult_free` 決定。

### 16.2 療程指代（序數/代名詞）檢索修正
- `identify_treatments_from_context` prompt 新增「序數（第一個/第二個/前者/後者）」處理：由新往回找「最近一則列出療程/編號的 AI 訊息」再對照；並把「相關療程」收緊為「實際指向的療程」。

### 16.3 診所基本資訊（地址/交通/電話/看診時間/停車）→ 確定性查表
- `search_clinics_info` 的 `treatment_name` 改為**只用於 `get_consult_info`**；retriever 路徑改用 `category` 查（不再被占位字「診所」灌爆）。
- 新增 `CLINIC_BASIC_INFO`（載入時抓 `clinics_qa.csv` 中 `question==診所地點` 或 `category==交通` 那筆）與 `CLINIC_INFO_INTENT`（地址/交通/怎麼去/停車/電話/看診…）。
- `category` 命中意圖 → 直接回 `CLINIC_BASIC_INFO`，不賭模糊排名；缺資料才 fallback。加 `[qa]` 檢索觀測 log。
- 待辦：`qa_retriever` 權重可由 `[0.7,0.3]→[0.3,0.7]`（偏 BM25）＋ 補強各 row keywords，提升長尾 FAQ 命中。

### 16.4 轉真人客服（CallCS=1）漏接修正
- `CS_KEYWORDS` 補片語（不含單獨「真人」以免誤判「真人案例」）：`專人`、`轉接`、`轉人工`、`真人服務`、`要真人`。修正「轉專人」被誤判成 CallCS=2 的問題。

### 16.5 圖片殘骸清理（`clean_text_from_urls`）
- 改為四步：①拔 markdown 圖片語法 `![alt](url)`/`![alt]()`/`![alt]` ②拔裸 URL ③逐行清掉只剩 `<圖片網址N>:`/條列/標點的殘骸行 ④收多餘空行。處理「URL 被抽走後留下空標籤/空 `![]()`」的三種格式。

### 16.6 部署：不再把快取烤進 image（修 stale）
- `Dockerfile` 移除 `COPY chroma_*`；CMD 先單進程 warmup（`python -c 'import toolkit.toolkits'`）建索引，再起 `uvicorn --workers 2`（避免兩 worker 同時重建 chroma 衝突）。
- `.dockerignore` 新增排除 `chroma_qa/`、`chroma_token_split/`（原已排除 `*.pkl`）。
- 部署：上傳 → `docker compose build --no-cache` → 用新 image 重新部署（勿只 restart 舊容器）。

### 16.7 驗證清單
| 測試輸入 | 預期 |
|---|---|
| 你們交通? / 地址? / 怎麼去? | 回地址＋捷運＋看診時間（`[clinic_info]` 命中） |
| 腦波機初診多少? | $4,800 / $1,000（consult_plan，收費） |
| Emface 諮詢要錢嗎? | 免費（consult_plan） |
| 第一個也有體驗方案嗎?（承上文編號清單） | 對應到清單第 1 項療程 |
| 我想轉專人 | CallCS=1（清空 text/images） |
| 問對比圖 | 文字無空標籤/空 `![]()`，圖以獨立訊息送出 |

---

## 17. 變更記錄（本批：體驗價工具化、防幻覺強化、療程介紹原文直出）

本批聚焦：**費用編價**與**療程事實幻覺**兩大風險，並把診所資訊 / 療程介紹改為確定性「原文直出」。
> 註：本批部分機制取代了 §16 的舊實作——§16.2 的費用注入用途、§16.3 的 `CLINIC_BASIC_INFO`/`CLINIC_INFO_INTENT`（已改為分店別 `CLINIC_INFO_ROWS` + `clinic_info_direct_answer`）、§16.6 的 `chroma_qa`（已拆成 `chroma_clinic_qa`/`chroma_treatment_qa`）。以現況 §8~§13 為準。

### 17.1 費用「體驗價」工具化（重大架構變更）
- 移除 graph 前 `[費用資訊]` SystemMessage 預注入與 backend 的費用 pre-filter 注入邏輯（`identify_treatments_from_context` / `filter_fees_by_treatments` 不再用於注入；`filter_fees_by_treatments` 仍供價格守門）。
- 新增 `treatment_fees_var`（ContextVar），backend 每 request 開始把整表存入（不預先 filter）。
- 新增 `get_treatment_fee(treatment_name)`（@tool，掛 booking），booking 認出療程當下查體驗價（別名比對 + 組合防呆 + grounding）。booking prompt 改：報價一律呼叫此工具、以工具回傳為準、嚴禁自編。（詳見 §8）

### 17.2 輸出端「價格守門」（price guard）
- 組 BackendResponse 前抓 NT$ 價，出現「不屬本療程的價」→ 帶正確價重寫一次；仍錯或 `[[HANDOFF]]` → force_handoff（CallCS=1，`price_fabrication`）。helper：`_extract_prices` / `_resolve_reply_treatments` / `_rewrite_price_reply`；trace 補 `price_guard`。（詳見 §8）

### 17.3 療程介紹「原文直出」
- `InfoPlan` 新增 `intro_treatment`；`TREATMENT_INTRO_ROWS`（`clinics_introductions3.csv`, big5）+ `get_treatment_intro`；information_node 對到介紹則一字不改直出、設 `skip_moderation=True`，杜絕 AI 自編英文全名/縮寫/原理。（詳見 §7）

### 17.4 QA（treatment_qa）檢索修正
- `qa_treatment` 空時用 `_resolve_qa_treatment_from_history` 回補；query 改「問題為主」`f"{qa_query} {t}"`；`get_qa_retriever` 新增 `k`（treatment_qa k=6、clinic_qa k=2，BM25 快取載入後強制覆寫 k）；回來依 category 過濾成本療程候選（`_treatment_group`/`_cat_matches_group`），回補路徑對不上就放棄 grounding。（詳見 §7、§11.2）

### 17.5 information_node 改 Planner 版
- 輕量模型一次輸出結構化 `InfoPlan` → 確定性工具呼叫 → 單次 Composer（gpt-4o），取代原 ReAct 迴圈。（詳見 §7）

### 17.6 Moderator 事實核對（faithfulness）與旗標
- `do_fact_check = bool(grounded) and not skip_fact_check`；`grounded` 是「有序、去重的 chunk 清單」（`register_grounded_content` 按 `\n\n---\n\n` 切 chunk、首次出現才留，消掉跨管道的近似重疊）。
- `_fact_check_and_clean`（gpt-4o）把 grounding 以 `[n]` 編號逐段送 LLM，抽療程硬事實時每條標 `source_id`（哪一號 chunk 支持它，無則 -1）+ 逐字 quote → 程式驗證**每條縮到它那一號 chunk**：`source_id=-1` 判無依據；quote 逐字命中或 `SequenceMatcher` ≥ 0.7 放行；對不上（多半改寫過）→ 只對**該 chunk** 做語意蘊涵判斷（`_entail_unsupported`）。無依據就刪 / 中性化，刪完無法回答 → `[[HANDOFF]]` → force_handoff。作用域縮到單一 chunk，逐字與語意兩關都又快又準，且不會被別段的字誤命中。
- `skip_moderation`（原文直出直通）、`skip_fact_check`（booking route 只做語氣/合規）；組 chunk 清單時補 `authorized_treatments_var` 一個「本診所有提供以下療程…」chunk、各療程官方介紹原文各補一個 chunk（chunk 級去重）；fee / consult 內容也 register 成 grounding。（詳見 §9）

### 17.7 健保 / 保險路由
- supervisor：健保 / 保險 / 理賠 / 自費 → booking_node；booking `search_clinics_info` 可處理清單加入「健保 / 保險政策」→ `search_clinics_info("診所", "健保")`（保險理賠用「保險」），走 clinic_qa。（詳見 §8）

### 17.8 AgentState 新增欄位
- `force_handoff` / `skip_moderation` / `skip_fact_check` / `trace`（`Annotated[dict, merge_trace]`）。（詳見 §5.1）

### 17.9 給後端的 output 補 `trace`
- `BackendResponse` 頂層新增 `trace`（節點寫 guard/route/draft/grounding/final/moderator；backend 補 user_input/price_guard/handoff_reason），供後端當 eval dataset；不接不影響 text/images/CallCS。（詳見 §3）

### 17.10 其他
- 移除兩段只印在後端 log 的 debug print（`📤 [Response to Backend]`、`🔍 Agent invoke result`）——不影響 output 契約。
