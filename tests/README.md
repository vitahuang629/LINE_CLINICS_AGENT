# 回歸測試 / Replay 使用說明

目的：**每次改 code 後,確保原本答對的題目不會變答錯。**

## 檔案

| 檔案 | 作用 |
|---|---|
| `fees.py` | 費用表 —— 改體驗價數字就改這 |
| `cases.yaml` | 回歸題庫 —— 「答對的特徵」都存這 |
| `test_regression.py` | pytest 回歸測試,改完 code 跑這個 |
| `replay.py` | 建題庫工具:把 `test_reply.txt` 的對話丟給 AI 跑一遍、跟真人答案並排 |
| `_harness.py` | 共用核心(組 request / 跑 AI / 驗特徵) |
| `../test_reply.txt` | 你收集的真實對話原始檔 |

## ⚠️ 執行前提

- 需要 `.env`(OpenAI key 等)與向量庫,和跑 app 同一個環境。
- **Windows 本機一定要設 `PYTHONUTF8=1`**,否則 app 印 emoji 會撞 cp950 編碼崩掉。
- 會真的呼叫 OpenAI,**每個測試點約 10~40 秒、會花 API 錢**。

---

## 流程 A:建題庫(用 replay)

把真實對話跑一遍,人工挑出「AI 答對」的存進題庫。

```powershell
# Windows 本機（poetry）
$env:PYTHONUTF8=1
poetry run python tests/replay.py --limit 5     # 先小量試跑 5 題
poetry run python tests/replay.py               # 跑全部（約 42 個測試點）
```

產出：
- `tests/replay_result.md` —— 「真人 vs AI」對照表 + 自動檢查,**你逐題標對/錯**
- `tests/suggested_cases.yaml` —— 幫你生好的題庫草稿

把**你判定「AI 答對」**的題,從 `suggested_cases.yaml` 挑進 `cases.yaml`,並確認/微調 `expect`。

## 流程 B:跑回歸(改完 code 每次跑)

```powershell
$env:PYTHONUTF8=1
poetry run pytest tests/test_regression.py -v          # 全部
poetry run pytest tests/test_regression.py -k 冷凍 -v   # 只跑名字含「冷凍」的
```

**紅的那題 = 這次改動把它弄壞了**(或該題 expect 需要更新)。

---

## 題庫格式(`cases.yaml`)

```yaml
- name: 冷凍_廣告_問幾次_報5999
  input:
    ad_referral: "冷凍減脂"        # 選填,只在首次對話(無歷史)對 AI 生效
    message_history:               # 選填,由「舊到新」自然順序;客服句加 [真人客服] 前綴
      - { type: human, content: "..." }
      - { type: ai,    content: "[真人客服] ..." }
    content: "5999療程幾次？"       # 必填,客人這句
  expect:                          # 只驗有寫的欄位
    call_cs: 0                     # 0 正常 / 1 轉真人 / 2 預約
    prices_present: [5999]         # 這些價格都要出現(自動比對 5,999)
    prices_absent: [16800]         # 這些價格都不可出現
    text_contains: ["竹北"]        # 這些字都要出現
    text_not_contains: ["17,997"]  # 這些字都不可出現
    images_include_any: ["emface"] # images 至少一張含任一子字串
    images_empty: true             # images 必須為空
    handoff_reason: null           # customer_keyword/fact_check/price_fabrication/booking/null
```

## 在 Docker 容器裡跑(Linux,免煩惱編碼)

若本機環境不齊,可在容器內跑(需先把 tests/ 一起 build 進 image 或掛載進去):

```powershell
docker exec -it fb-clinics-agent python tests/replay.py --limit 3
docker exec -it fb-clinics-agent pytest tests/test_regression.py -v
```
