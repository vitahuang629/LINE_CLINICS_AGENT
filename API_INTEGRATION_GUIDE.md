# 後端串接 API 文件

## 概述
此 API 提供給後端工程師串接，用於處理 LINE 官方帳號的客戶諮詢。

## API 端點

### POST /chat

處理客戶的醫美諮詢問題，回傳 AI 回覆的文字和圖片。

**URL**: `http://your-domain:8004/chat`

**Method**: `POST`

**Content-Type**: `application/json`

---

## 請求格式

```json
{
  "line_uuid": "U1234567890abcdef...",
  "messages": [
    {
      "role": "user",
      "content": "我想了解減重療程"
    },
    {
      "role": "assistant", 
      "content": "我們有多種減重療程可以選擇..."
    },
    {
      "role": "user",
      "content": "EMBODY 的效果如何？"
    }
  ]
}
```

### 參數說明

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| line_uuid | string | 是 | LINE 使用者的唯一識別碼 (User ID) |
| messages | array | 是 | 對話歷史記錄 |
| messages[].role | string | 是 | 訊息角色，可為 "user" 或 "assistant" |
| messages[].content | string | 是 | 訊息內容 |

### 注意事項

1. `messages` 陣列應包含完整的對話歷史，最新的訊息放在最後
2. 第一次對話時，`messages` 只需包含一則 user 訊息
3. 後續對話需包含之前的對話記錄，以維持上下文

---

## 回應格式

```json
{
  "text": "EMBODY 是一種非侵入式的減脂療程...",
  "images": [
    "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/embody_intro.jpg"
  ]
}
```

### 回應欄位說明

| 欄位 | 類型 | 說明 |
|------|------|------|
| text | string | AI 回覆的文字內容（已移除圖片 URL） |
| images | array | 需要顯示的圖片 URL 列表，可能為空陣列 |

---

## 使用範例

### 範例 1: 首次諮詢

**Request:**
```json
{
  "line_uuid": "Uabcdef1234567890",
  "messages": [
    {
      "role": "user",
      "content": "我最近失眠很嚴重"
    }
  ]
}
```

**Response:**
```json
{
  "text": "了解您最近失眠的困擾，這確實會影響生活品質。請問您的失眠情況大約持續多久了呢？是入睡困難還是容易醒來？",
  "images": []
}
```

---

### 範例 2: 持續對話

**Request:**
```json
{
  "line_uuid": "Uabcdef1234567890",
  "messages": [
    {
      "role": "user",
      "content": "我最近失眠很嚴重"
    },
    {
      "role": "assistant",
      "content": "了解您最近失眠的困擾..."
    },
    {
      "role": "user",
      "content": "大概兩個月了，很難入睡"
    }
  ]
}
```

**Response:**
```json
{
  "text": "持續兩個月的入睡困難確實需要重視。我們診所有 Deep TMS 腦波科技療程，這是國際認證的非藥物、無侵入性治療...",
  "images": [
    "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/deep_tms_intro.jpg"
  ]
}
```

---

### 範例 3: 詢問療程資訊

**Request:**
```json
{
  "line_uuid": "Uabcdef1234567890",
  "messages": [
    {
      "role": "user",
      "content": "我想了解 Emface 的效果"
    }
  ]
}
```

**Response:**
```json
{
  "text": "Emface 是一種結合射頻和 HIFES 技術的臉部療程，可以幫助改善臉部輪廓和膚質。這是 Emface 的對比照：",
  "images": [
    "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/emface_ollie_ba.jpg"
  ]
}
```

---

## 錯誤處理

### HTTP 狀態碼

| 狀態碼 | 說明 |
|--------|------|
| 200 | 請求成功 |
| 422 | 請求格式錯誤（缺少必填欄位或格式不正確） |
| 500 | 伺服器內部錯誤 |

### 錯誤回應範例

```json
{
  "detail": [
    {
      "loc": ["body", "line_uuid"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## 整合建議

1. **對話記錄管理**: 建議在你們的後端維護每個 LINE UUID 的對話歷史
2. **圖片處理**: `images` 陣列中的 URL 可以直接用於 LINE 的 ImageMessage
3. **錯誤重試**: 建議實作重試機制，避免網路問題導致對話中斷
4. **超時設定**: 建議設定 30 秒的請求超時時間

---

## 測試工具

可以使用 `curl` 或 Postman 進行測試：

```bash
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "line_uuid": "test_user_001",
    "messages": [
      {
        "role": "user",
        "content": "我想了解減重療程"
      }
    ]
  }'
```

---

## 聯絡資訊

如有任何問題，請聯繫開發團隊。
