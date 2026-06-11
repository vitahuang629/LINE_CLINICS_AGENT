# 後端串接修改總結

## 修改概述

將原本需要手機號碼的系統改為使用 LINE UUID 作為唯一識別碼，並從資料庫查詢歷史對話記錄。

---

## 主要修改檔案

### 1. `app/models.py`
新增後端串接專用的資料模型：
- `BackendChatMessage`: 對話訊息格式
- `BackendUserQuery`: 請求格式（使用 line_uuid）
- `BackendResponse`: 回應格式（包含 text 和 images）

### 2. `app/backend_agent_service.py` (新增)
專門給後端工程師串接用的服務：
- 使用 `line_uuid` 作為識別碼
- 從資料庫查詢歷史對話記錄
- 自動提取圖片 URL 並分離文字內容
- 回傳結構化的 JSON 格式

### 3. `utils/profile_db.py`
新增 UUID 查詢功能：
- `get_user_profile_by_uuid()`: 根據 LINE UUID 查詢歷史對話
- 查詢 `line_account` 和 `line_message` 表
- 回傳格式：
  ```python
  {
      'line_name': '使用者暱稱',
      'conversation_history': [
          {'role': 'user', 'content': '訊息內容', 'timestamp': '時間'},
          ...
      ]
  }
  ```

### 4. `agent.py`
修改 `format_user_profile_text()` 函數：
- 支援新的資料格式（歷史對話記錄）
- 向下相容舊格式（手機號碼查詢）
- 顯示最近 5 則對話記錄

### 5. `main_webhook.py`
新增 `/chat` API 端點：
- 接收 LINE UUID 和對話歷史
- 回傳 AI 回覆的文字和圖片 URL

---

## 資料庫結構假設

```sql
-- line_account 表
CREATE TABLE line_account (
    id INT PRIMARY KEY,
    line_id VARCHAR(255),  -- LINE UUID
    line_name VARCHAR(255), -- 使用者暱稱
    ...
);

-- line_message 表
CREATE TABLE line_message (
    id INT PRIMARY KEY,
    line_account_id INT,
    message TEXT,
    role VARCHAR(50),  -- 'user' 或 'assistant'
    created_at DATETIME,
    ...
);
```

---

## API 使用方式

### 端點
```
POST http://your-domain:8004/chat
```

### 請求範例
```json
{
  "line_uuid": "U1234567890abcdef...",
  "messages": [
    {
      "role": "user",
      "content": "我想了解減重療程"
    }
  ]
}
```

### 回應範例
```json
{
  "text": "我們有多種減重療程可以選擇，包括 EMBODY、NEO 等...",
  "images": [
    "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/embody_intro.jpg"
  ]
}
```

---

## 與舊版的差異

| 項目 | 舊版（手機號碼） | 新版（LINE UUID） |
|------|-----------------|------------------|
| 識別碼 | 手機號碼 (09xxxxxxxx) | LINE UUID (Uxxxxx...) |
| 資料來源 | member + doctor_comments | line_account + line_message |
| 資料內容 | 個人資料（年齡、性別、症狀） | 歷史對話記錄 |
| API 端點 | /execute | /chat |
| 使用場景 | LINE webhook（測試用） | 後端系統串接 |

---

## 測試方式

1. 啟動服務：
```bash
python main_webhook.py
```

2. 執行測試腳本：
```bash
python test_backend_api.py
```

3. 或使用 curl：
```bash
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "line_uuid": "test_user_001",
    "messages": [
      {"role": "user", "content": "我想了解減重療程"}
    ]
  }'
```

---

## 注意事項

1. **資料庫連線**: 確保 `.env` 檔案中的資料庫設定正確
2. **歷史對話**: 系統會自動從資料庫查詢最近 20 則對話記錄
3. **新客戶處理**: 如果查無歷史記錄，會當作新客戶處理
4. **圖片處理**: 系統會自動識別回覆中的圖片 URL 並分離出來
5. **對話記憶**: 使用 LangGraph 的 MemorySaver，以 UUID 作為 thread_id

---

## 後續建議

1. **對話記錄儲存**: 後端工程師需要將每次的對話記錄存入 `line_message` 表
2. **錯誤處理**: 建議加入重試機制和錯誤通知
3. **效能優化**: 如果對話記錄很多，可以考慮只查詢最近的記錄
4. **安全性**: 建議加入 API 認證機制（如 API Key）

---

## 聯絡資訊

如有任何問題，請參考 `API_INTEGRATION_GUIDE.md` 或聯繫開發團隊。
