# 圖片處理邏輯說明

## 概述

系統會根據 AI 回覆的內容和使用者的問題，自動決定要回傳哪些圖片。

---

## 圖片決策流程

### 1. 提取 AI 回覆中的圖片 URL
首先，系統會從 AI 的回覆文字中提取所有圖片 URL（格式：https://...jpg/png/webp）

### 2. 根據關鍵字判斷額外圖片
系統會檢查以下條件，決定是否加入額外的圖片：

---

## 圖片規則

### 規則 1: 自律神經檢測
**觸發條件**: AI 回覆包含「兩種方案」或「自律神經檢測」

**回傳圖片**:
1. `treatment_procedure.jpg` - 療程流程圖
2. `autonomic_fees.jpg` - 費用說明圖

---

### 規則 2: 體態檢測
**觸發條件**: AI 回覆包含「體態檢測」、「EMBODY」、「NEO」或「瘦瘦筆」，且包含「檢測」

**回傳圖片**:
- `body_consult.jpg` - 體態諮詢圖

---

### 規則 3: 停車場/地址
**觸發條件**: 
- AI 回覆包含「地址」、「地點」或「位於」
- 或使用者問題包含「哪裡」、「停車」、「停車場」或「開車」

**回傳圖片**:
- `parking_lots.jpg` - 停車場位置圖

---

### 規則 4: Emface 療程
**觸發條件**: AI 回覆包含「Emface」且不包含「電波」

**回傳圖片**:
- 如果 AI 已提供對比照 URL → 使用 AI 提供的圖片
- 如果 AI 回覆包含「免費」→ `emface_intro.jpg`

---

### 規則 5: 瘦瘦筆（單獨）
**觸發條件**: AI 回覆包含「瘦瘦筆」，但不包含「EMBODY」、「NEO」、「SIS」

**回傳圖片**:
- `skin_pencial_intro.jpg` - 瘦瘦筆介紹圖

---

### 規則 6: NEO 療程（單獨）
**觸發條件**: AI 回覆包含「NEO」，但不包含「EMBODY」、「SIS」、「瘦瘦筆」

**回傳圖片**:
- `neo_intro.jpg` - NEO 介紹圖

---

### 規則 7: SIS 療程（單獨）
**觸發條件**: AI 回覆包含「SIS」，但不包含「EMBODY」、「NEO」、「瘦瘦筆」

**回傳圖片**:
- `sis_intro.jpg` - SIS 介紹圖

---

## 圖片 URL 基礎路徑

所有圖片都存放在 S3：
```
https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/
```

---

## 完整圖片列表

| 檔案名稱 | 用途 | 觸發條件 |
|---------|------|---------|
| `treatment_procedure.jpg` | 療程流程 | 自律神經檢測 |
| `autonomic_fees.jpg` | 費用說明 | 自律神經檢測 |
| `body_consult.jpg` | 體態諮詢 | 體態檢測 |
| `parking_lots.jpg` | 停車場位置 | 地址/停車查詢 |
| `emface_intro.jpg` | Emface 介紹 | Emface + 免費 |
| `emface_ollie_ba.jpg` | Emface 對比照 | AI 自動提供 |
| `skin_pencial_intro.jpg` | 瘦瘦筆介紹 | 單獨提到瘦瘦筆 |
| `neo_intro.jpg` | NEO 介紹 | 單獨提到 NEO |
| `sis_intro.jpg` | SIS 介紹 | 單獨提到 SIS |

---

## 範例

### 範例 1: 停車場查詢
**使用者問題**: "請問有停車場嗎？"

**AI 回覆**: "診所附近有兩個停車場..."

**回傳圖片**:
```json
{
  "images": [
    "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/parking_lots.jpg"
  ]
}
```

---

### 範例 2: Emface 詢問
**使用者問題**: "Emface 的效果如何？"

**AI 回覆**: "Emface 是一種結合射頻和 HIFES 技術的臉部療程。這是 Emface 的對比照: https://...emface_ollie_ba.jpg"

**回傳圖片**:
```json
{
  "images": [
    "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/emface_ollie_ba.jpg"
  ]
}
```

---

### 範例 3: 體態檢測
**使用者問題**: "我想了解 EMBODY"

**AI 回覆**: "可以先幫您安排體態檢測..."

**回傳圖片**:
```json
{
  "images": [
    "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/body_consult.jpg"
  ]
}
```

---

## 注意事項

1. **優先順序**: AI 提取的圖片 > 額外加入的圖片
2. **去重**: 系統會自動去除重複的圖片 URL
3. **順序**: 保持圖片的原始順序
4. **文字清理**: 回傳的 text 欄位會自動移除圖片 URL

---

## 修改建議

如果需要新增或修改圖片規則，請編輯 `app/backend_agent_service.py` 中的 `determine_additional_images()` 函數。

範例：
```python
# 新增規則：雷射療程
elif "雷射" in ai_reply:
    additional_images.append("https://.../laser_intro.jpg")
```
