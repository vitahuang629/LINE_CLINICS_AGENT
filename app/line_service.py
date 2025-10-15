import os
import requests
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, TemplateSendMessage, ButtonsTemplate, URITemplateAction, MessageTemplateAction, JoinEvent
from fastapi import Request
from .utils import is_valid_phone_number
from dotenv import load_dotenv
import mysql.connector
from toolkit.notify_kits import check_appointment_keywords

load_dotenv()

DB_PASS=os.getenv("OPENAI_API_KEY")
GROUP_ID=os.getenv("GROUP_ID")


# === MySQL 連線 ===
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password=DB_PASS,
        database="line_test"
    )

# 新增訊息
def insert_message(user_id, role, content):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO line_messages (user_id, role, content)
        VALUES (%s, %s, %s)
    """, (user_id, role, content))
    conn.commit()
    cursor.close()
    conn.close()

# 取最近 limit 筆歷史訊息
def get_message_history(user_id, limit=20):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT role, content FROM line_messages
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    """, (user_id, limit))
    messages = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(reversed(messages))  # DESC 取出後反轉

# === 載入環境變數 ===
load_dotenv()
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
API_URL = os.getenv("EXECUTE_API_URL", "http://127.0.0.1:8003/execute")

print("LINE_CHANNEL_ACCESS_TOKEN =", LINE_CHANNEL_ACCESS_TOKEN)
print("LINE_CHANNEL_SECRET =", LINE_CHANNEL_SECRET)
print("API_URL =", API_URL)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 暫存 userId -> phone_number（可以放 DB，但先暫存記憶體）
user_phone_map = {}

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    # 檢查是否來自群組
    if event.source.type == "group":
        group_id = event.source.group_id
        user_message = event.message.text
        
        print(f"📨 收到群組訊息 - 群組 ID: {group_id}")
        
        # 當用戶輸入特定指令時，回傳群組 ID
        if user_message.strip().lower() in ["取得群組id", "群組id", "/groupid"]:
            try:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(
                        text=f"這個群組的 ID 是：\n`{group_id}`\n\n請保存這個 ID 以供後續使用！"
                    )
                )
                print(f"✅ 已回傳群組 ID: {group_id}")
            except Exception as e:
                print(f"回覆訊息失敗: {e}")
    user_text = event.message.text.strip()
    user_id = event.source.user_id
    reply_token = event.reply_token

    # 如果沒輸入過手機
    if user_id not in user_phone_map:
        if not is_valid_phone_number(user_text):
            line_bot_api.reply_message(
                reply_token,
                [TextSendMessage(text="""請先輸入您的手機號碼（格式：09xxxxxxxx）\n您好，我是霍普艾小編，很高興為您服務😊"""),
                 TextSendMessage(text="""我們霍普金斯聯名診所位於（永春捷運站５號出口旁）\n🔹診所地點：台北市信義區忠孝東路五段477-6號\n🔹看診時段：週一到週六：11:15~20:00\n🔺最晚預約時間為19:00，建議提前預約\n⚠️週日固定公休（特定假期另行公告）
                    """)])
            return
        # 存手機號碼（記憶體或 DB 都可以）
        user_phone_map[user_id] = user_text
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=f"手機號碼已記錄：{user_text}，可以開始諮詢療程了！")
        )
        return

    phone_number = user_phone_map[user_id]

    # 1️⃣ 儲存使用者訊息到 DB
    insert_message(user_id, "user", user_text)

    # 2️⃣ 取得歷史對話（包含剛剛存的）
    history = get_message_history(user_id)

    # 3️⃣ 呼叫後端 API
    payload = {
        "messages": history,  # 把全部歷史對話傳給 API
        "phone_number": phone_number
    }
    try:
        res = requests.post(API_URL, json=payload, verify=False)
        ai_reply = res.json().get("messages", "抱歉，目前無法回覆。")
    except Exception as e:
        ai_reply = f"呼叫 API 發生錯誤: {e}"

    # 4️⃣ 儲存 AI 回覆到 DB
    insert_message(user_id, "assistant", ai_reply)

    # 找客人line user profile
    profile = line_bot_api.get_profile(event.source.user_id)
    # print(profile.display_name)   # 使用者暱稱
    # print(profile.user_id)        # 使用者ID
    # print(profile.picture_url)    # 頭像
    # print(profile.status_message) # 狀態訊息

    if "睡眠障礙與自律神經失調" in user_text:
        # 5️⃣ 回覆 LINE
        # line_bot_api.reply_message(reply_token, TextSendMessage(text=ai_reply))   #0926 如果只有單一回復文字
        message = [# ImageSendMessage( #傳圖片
        #                 original_content_url="https://www.bing.com/images/search?view=detailV2&ccid=6PW5h7ap&id=946E94F5329BE74B489D75E59638585FB2E94040&thid=OIP.6PW5h7ap9WzNHV0mP2bOtAHaEK&mediaurl=https%3A%2F%2Fi.pinimg.com%2Foriginals%2F51%2F49%2F8e%2F51498ef3c0cb498e6c64eceaa3c332d0.jpg&exph=405&expw=720&q=imgurl%3Ahttps%3A%2F%2Fwww.nornsblog.com%2Fwp-content%2Fuploads%2FLotso9-768x432.jpg&form=vissbi&ck=FF82B673E49143F530968658A70B6B37&selectedindex=2&itb=0&cw=1721&ch=832&first=1&insightstoken=ccid_SxoXKjT8*cp_3C414757C366C5DFE2ED0CFC01736CC0*mid_2A0B9D63CFB61DA5122142475093EA1CAE43E00F*thid_OIP.SxoXKjT8MyvKLTbjE-nrNwHaEK&iss=VSI&vt=2&vsimg=https%3A%2F%2Fwww.nornsblog.com%2Fwp-content%2Fuploads%2FLotso9-768x432.jpg&sim=11&pivotparams=insightsToken%3Dccid_pNYytZoE*cp_C1B3BEF5C77302D46BF65E4481A49DC6*mid_1F8028C823C0227546EE6748347870CCCFB3D32F*thid_OIP.pNYytZoETY4!_o4PG6kdJWgHaHa&cdnurl=https%3A%2F%2Fth.bing.com%2Fth%2Fid%2FR.e8f5b987b6a9f56ccd1d5d263f66ceb4%3Frik%3DQEDpsl9YOJbldQ%26pid%3DImgRaw%26r%3D0",
        #                 preview_image_url="https://www.bing.com/images/search?view=detailV2&ccid=pNYytZoE&id=1F8028C823C0227546EE6748347870CCCFB3D32F&thid=OIP.pNYytZoETY4_o4PG6kdJWgHaHa&mediaurl=https%3A%2F%2F64.media.tumblr.com%2Ftumblr_nj1j4qAKXD1t2csxzo1_1422706202_cover.jpg&exph=500&expw=500&q=imgurl%3Ahttps%3A%2F%2Fwww.nornsblog.com%2Fwp-content%2Fuploads%2FLotso9-768x432.jpg&form=vissbi&ck=C1B3BEF5C77302D46BF65E4481A49DC6&selectedindex=1&itb=0&cw=1721&ch=832&first=1&insightstoken=ccid_SxoXKjT8*cp_3C414757C366C5DFE2ED0CFC01736CC0*mid_2A0B9D63CFB61DA5122142475093EA1CAE43E00F*thid_OIP.SxoXKjT8MyvKLTbjE-nrNwHaEK&iss=VSI&vt=2&vsimg=https%3A%2F%2Fwww.nornsblog.com%2Fwp-content%2Fuploads%2FLotso9-768x432.jpg&sim=11&pivotparams=insightsToken%3Dccid_%252BCusoKTw*cp_D40C56886C7BEAE888406E02D9C4998F*mid_312250FD39FE39B141B031E555FF6768AF2F8E4F*thid_OIP.-CusoKTw8ICcYkwKlZBWMgHaDm&cdnurl=https%3A%2F%2Fth.bing.com%2Fth%2Fid%2FR.a4d632b59a044d8e3fa383c6ea47495a%3Frik%3DL9Ozz8xweDRIZw%26pid%3DImgRaw%26r%3D0"
        #             ),
                    TextSendMessage(text=ai_reply),
                    TemplateSendMessage(
                    alt_text="睡眠與自律神經失調",
                    template=ButtonsTemplate(
                        thumbnail_image_url="https://th.bing.com/th/id/R.f82baca0a4f0f0809c624c0a95905632?rik=T44vr2hn%2f1XlMQ&riu=http%3a%2f%2fwww.falpala.it%2fwp-content%2fuploads%2f2013%2f08%2forso-rosa.jpg&ehk=XzHyy3P3%2bmprTkahh65p3dtu5mEp9UYZ%2fvsLzr1eMRY%3d&risl=&pid=ImgRaw&r=0",
                        title="Deep TMS-腦波科技",
                        text="國際認證非藥物、無侵入性治療，以磁場技術，刺激大腦神經元迴路，調節自律神經、改善失眠焦慮，幫助身心重新找回平衡與放鬆。",
                        actions=[
                            URITemplateAction(
                                label="了解更多",
                                uri="https://www.hopkins.com.tw/"
                            ),
                            MessageTemplateAction(
                                label='✨總是睡不好？失眠、憂鬱焦慮',
                                text='從根本調節大腦神經元與修復機制，促進代謝清除、提升深層睡眠與記憶力，改善腦部功能🧠穩定情緒幫助放鬆。',
                            ),
                            MessageTemplateAction(
                                label='✨擺脫藥物依賴｜延長深層睡眠',
                                text='不靠藥物也能改善失眠、焦慮、壓力過大？Deep TMS腦波科技，延長深層睡眠💤讓您晝夜節奏回歸。',
                            )
                        ]
                    )
                )
        ]
    else:
        message = TextSendMessage(text=ai_reply)

    line_bot_api.reply_message(reply_token, message)


    if check_appointment_keywords(ai_reply) is True:
        group_id = GROUP_ID
        line_bot_api.push_message(
        group_id,
        TextSendMessage(text=f"📌 客人預約待處理:\n{profile.display_name}")
    )



# Webhook endpoint
async def line_webhook(request: Request):
    signature = request.headers.get('X-Line-Signature', '')
    body = (await request.body()).decode("utf-8")
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400
    return "OK"