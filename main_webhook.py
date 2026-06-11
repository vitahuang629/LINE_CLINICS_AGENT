from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
# from app.line_service import line_webhook
from app.backend_agent_service import execute_backend_agent
from app.models import BackendUserQuery, BackendResponse

app = FastAPI(root_path="/fb-clinics-agent")

# @app.post("/line_webhook")
# async def webhook(request: Request):
#     return await line_webhook(request)

@app.post("/chat", response_model=BackendResponse)
async def backend_chat(query: BackendUserQuery):
    """
    給後端工程師串接用的 API 端點

    Request Body (兩種情境)：

    A. 客人傳文字訊息：
    {
        "fb_account": "1234567890abcdef",
        "content": "客人這次說的話",
        "image_url": [],
        "message_history": [
            {"type": "human", "content": "你好"},
            {"type": "ai",    "content": "您好！我是霍普艾小編"}
        ],
        "ad_referral": null
    }

    B. 客人傳圖片訊息（可能多張）：
    {
        "fb_account": "1234567890abcdef",
        "content": null,
        "image_url": [
            "https://customer-uploaded.com/img1.jpg",
            "https://customer-uploaded.com/img2.jpg"
        ],
        "message_history": [ ... ],
        "ad_referral": null
    }

    C. 首次對話、從 Meta 廣告進來：
    {
        "fb_account": "fb_new_customer_001",
        "content": "你好",
        "image_url": [],
        "message_history": [],
        "ad_referral": "NEO",
        "treatment_fees": [
            {"name": "NEO-熱磁減脂(30分鐘) + 冷凍單點(60分鐘)", "price": 15999},
            {"name": "Emface 全臉拉提", "price": 8000}
        ]
    }

    Notes:
    - content 與 image_url 互斥：純文字訊息 → content 有值、image_url 為空陣列；
                                   圖片訊息 → content 為 null、image_url 為一或多張。
    - message_history 為過去對話，不包含本次的 content / image_url。
      排序：由新到舊（依後端 DB 預設順序即可，本服務內部會自動反轉處理）。
    - 過去訊息的圖片 OCR 由本服務內部處理，OCR 文字不回傳、不需存 DB；
      message_history 只傳 content 即可（AI 上一輪回覆已內化該圖片重點）。
    - type:
        "human" = 客人說的話
        "ai"    = AI 或真人客服回覆給客人的話
      若該則訊息是「真人客服」回覆，請後端在 content 前面加上 "[真人客服] " 前綴，
      例如：{"type": "ai", "content": "[真人客服] 您好，已幫您安排..."}
      這樣 AI 才能正確區分自己講的話與真人講的話，避免延續錯誤承諾。
    - ad_referral：客人從 Meta 廣告點進來時，後端解析廣告對應的療程關鍵字後傳入
      （例如 "NEO"、"Emface"、"自律神經"）。建議只在 message_history 為空（首次對話）時傳，
      之後設為 null 即可。AI 只會在「首次對話」這一情境把它注入 SystemMessage 作開場提示。
    - treatment_fees：目前各療程的最新體驗價，每次都傳完整列表（建議後端加 Redis cache 5-15 分鐘）。
      欄位：
        name  = 療程名稱（可含時長/組合等細節，例如 "NEO-熱磁減脂(30分鐘) + 冷凍單點(60分鐘)"）
        price = 體驗價，純數字（int），例如 15999
      AI 會把整份注入 SystemMessage，客人問價時自動參考；下架或非適用的療程後端就不要送進來。

    Response:
    {
        "text": "AI 回覆的文字內容",
        "images": ["https://example.com/image1.jpg"],
        "CallCS": 0
    }

    CallCS 三種值：
      0 = 正常對話，text 是 AI 完整回覆，後端把 text/images 推給客人即可。
      1 = 客人主動找真人客服（命中 CS_KEYWORDS）→ text 與 images **會是空的**，
          後端不用推任何 AI 回覆給客人，直接觸發真人客服通知（Discord/LINE 群組 push 等）。
      2 = 預約流程（AI 呼叫了 set_appointment tool）→ text 是 AI 的預約引導
          （例如「請提供姓名、療程、時間、電話...」），images 會是空 list。
          後端應該：
          1) 先把 text 推給客人，讓客人看到要填的欄位
          2) 接著觸發真人客服通知，讓專人接手後續預約處理
    """
    # 把 sync 的 execute_backend_agent 丟到 threadpool 跑，避免阻塞 event loop
    # 這樣多個 request 進來時可以並行處理（threadpool 預設 40 個 worker）
    return await run_in_threadpool(execute_backend_agent, query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)