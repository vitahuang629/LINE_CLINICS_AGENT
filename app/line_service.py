import os
import requests
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, TemplateSendMessage, ButtonsTemplate, URITemplateAction, MessageTemplateAction, JoinEvent
from fastapi import Request
from .utils import is_valid_phone_number
from dotenv import load_dotenv
import mysql.connector
from toolkit.notify_kits import check_appointment_keywords
import re
import random

load_dotenv()

DB_PASS=os.getenv("DB_PASS")
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
                 TextSendMessage(text="""我們診所結合【科學根據及科技原理】提供減脂減重、身材雕塑、睡眠問題、肌膚保養、抗衰逆齡、私密處保養，針對不同狀況做加強協助達到更理想的狀況😊）
                    """)])
            return
        # 存手機號碼（記憶體或 DB 都可以）
        user_phone_map[user_id] = user_text
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=f"手機號碼已記錄：{user_text}，請告訴我您的年齡以及症狀喔~")
        )
        return

    phone_number = user_phone_map[user_id]

    # 1️⃣ 儲存使用者訊息到 DB
    insert_message(user_id, "user", user_text)

    # 🧩 高風險關鍵詞安全檢查
    HIGH_RISK_KEYWORDS = ["FM2", "fm2", "安眠藥", "鎮定劑"]
    if any(word in user_text for word in HIGH_RISK_KEYWORDS):
        warning_message = """提醒：
    藥物治療僅適合用於初期短效輔助，很多人為了解決睡眠或情緒問題而依賴藥物，但其實藥物只是「強迫大腦關機」，會使得無法正常運行進入（深層睡眠狀態來分泌脊髓液，幫助排毒、修復細胞，讓身心徹底恢復能量）。

    藥物僅適合在專業醫師指導下短期使用，長期服用可能造成依賴與副作用：
    🔸 記憶力、專注力下降
    🔸 睡眠品質惡化、焦慮加劇
    🔸 肝腎代謝負擔增加
    建議您與專業醫師討論替代的非藥物治療方式（如腦波調節）。
    """
        line_bot_api.reply_message(reply_token, TextSendMessage(text=warning_message))
        return  # 🚫 不呼叫 AI，直接結束這輪處理

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
        # photo_urls = res.json().get("photo_urls", [])     #1023
        # send_line_message(user_id=user_id, ai_reply=ai_reply, image_urls=photo_urls) #1023
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

    # url_pattern = r'https?://[^\s]+(?:jpg|jpeg|png|webp)'
    url_pattern = r'https://[^\s)>\]\'"]+\.(?:jpg|jpeg|png|webp)'
    urls = re.findall(url_pattern, ai_reply)
    print('all_urlssssssssssss', urls)

    clean_text = re.sub(url_pattern, '', ai_reply).strip()
    # 移除多餘的空行
    clean_text = re.sub(r'\n\s*\n', '\n', clean_text)

    if "睡眠障礙與自律神經失調" in user_text:
        # 5️⃣ 回覆 LINE
        # line_bot_api.reply_message(reply_token, TextSendMessage(text=ai_reply))   #0926 如果只有單一回復文字
        message = [
                    TextSendMessage(text=clean_text),
                    TemplateSendMessage(
                    alt_text="睡眠與自律神經失調",
                    template=ButtonsTemplate(
                        #thumbnail_image_url="https://th.bing.com/th/id/R.f82baca0a4f0f0809c624c0a95905632?rik=T44vr2hn%2f1XlMQ&riu=http%3a%2f%2fwww.falpala.it%2fwp-content%2fuploads%2f2013%2f08%2forso-rosa.jpg&ehk=XzHyy3P3%2bmprTkahh65p3dtu5mEp9UYZ%2fvsLzr1eMRY%3d&risl=&pid=ImgRaw&r=0",
                        thumbnail_image_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/parking_lots.jpg",
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
    elif "兩種方案" in ai_reply or "自律神經檢測" in ai_reply:
        message = [
                  TextSendMessage(text=clean_text),
                  ImageSendMessage(
                      original_content_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/treatment_procedure.jpg",
                      preview_image_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/treatment_procedure.jpg"
                  )
        ]
        message.append(ImageSendMessage(
                original_content_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/autonomic_fees.jpg",
                preview_image_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/autonomic_fees.jpg"
            ))
    elif any(kw in ai_reply for kw in ["體態檢測", "EMBODY", "NEO", "瘦瘦筆"]) and "檢測" in ai_reply:
        print('444444444444444444444')
        message = [
                  #TextSendMessage(text="""可以先幫您安排【免費諮詢＋檢測評估】可以更加了解目前的實際狀況，✅️例如：皮下脂肪、內臟脂肪、肌肉量、基礎代謝率、腹直肌分離、 BMI等指數，依據科學指標數據來規劃適合您的方式，確保達到理想效果！再考慮是否進行💕
                  #            """),
                  TextSendMessage(text=clean_text),
                  ImageSendMessage(
                      original_content_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/body_consult.jpg",
                      preview_image_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/body_consult.jpg"
                  )
        ]
    elif "地址" in ai_reply or "地點" in ai_reply or "哪裡" in user_text or "位於" in ai_reply or "停車" in user_text or "停車場" in user_text or "開車" in user_text:
        message = [TextSendMessage(text = """🔶愛買超市停車場➜(忠孝東路五段297號附有地下三樓)\n🔶春光公園停車場➜(台北市信義區忠孝東路五段666號)\n兩個停車場步行診所約莫5分鐘內可到達，建議可以停放『春光公園停車場』，車位比較多唷😉
                                    """),
                    ImageSendMessage( #傳圖片
                    original_content_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/parking_lots.jpg",
                    preview_image_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/parking_lots.jpg"
                            )]
    elif "Emface" in ai_reply and "電波" not in ai_reply:
        print('5555555555555')
        print(urls)
        if urls: 
            print('uuuuuuuuuuuuuuuuuuuuuuuurrrrrrrrrrrr')
            contrast_url = urls[0]
            message = [
                    TextSendMessage(text=clean_text),
                    ImageSendMessage(
                        original_content_url=contrast_url,
                        preview_image_url=contrast_url
                    )
            ]
            # message.append(ImageSendMessage(
            #     original_content_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/emface_intro.jpg",
            #     preview_image_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/emface_intro.jpg"
            # ))
        elif "免費" in ai_reply:
            print('nonooooooooooooooooooooooo')
            message = [
                    TextSendMessage(text=ai_reply),
                    ImageSendMessage(
                        original_content_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/emface_intro.jpg",
                        preview_image_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/emface_intro.jpg"
                    )
            ]
    elif "瘦瘦筆" in ai_reply and ("EMBODY" not in ai_reply and "NEO" not in ai_reply and "SIS" not in ai_reply):
        message = [
                    TextSendMessage(text=ai_reply),
                    ImageSendMessage(
                        original_content_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/skin_pencial_intro.jpg",
                        preview_image_url="https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/skin_pencial_intro.jpg"
                    )
            ]
    elif urls:
        print('6666666666666666666')
        print(urls)
        message = [
            TextSendMessage(text=clean_text),
            ImageSendMessage(original_content_url=urls[0], preview_image_url=urls[0])
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