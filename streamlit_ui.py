import streamlit as st
import requests
import base64

API_URL = "http://127.0.0.1:8003/execute"


def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# 圖片轉 base64
image_path = "stich.jpg"
encoded_image = get_base64_image(image_path)

# 背景 CSS（正確版本）
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: "Microsoft JhengHei", sans-serif;
        color: #000000;
    }}

    .chat-message.user {{
        background-color: rgba(0, 132, 255, 0.15);
        padding: 0.8rem 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        max-width: 80%;
        word-wrap: break-word;
    }}

    .chat-message.assistant {{
        background-color: rgba(240, 240, 240, 0.85);
        padding: 0.8rem 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        max-width: 80%;
        word-wrap: break-word;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 醫美療程諮詢系統 - 聊天介面")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 處理使用者 ID 輸入 ---
# 如果會話狀態中沒有 user_id，則顯示輸入框
if "phome_number" not in st.session_state or not st.session_state.phone_number:
    st.session_state.phone_number = st.text_input("請輸入你的 手機號碼：", "")
    # 如果使用者輸入了 ID，顯示歡迎訊息
    if st.session_state.phone_number:
        st.success(f"哈囉，使用者 {st.session_state.phone_number}！")
    else:
        # 如果還沒輸入 ID，顯示警告
        st.warning("請輸入你的 手機號碼 以開始對話。")
        # 這裡不使用 st.stop()，讓程式碼繼續執行，但後續的聊天區塊會因為 user_id 未設定而不顯示
else:
    # 如果 user_id 已經存在，可以直接顯示歡迎訊息（可選）
    st.info(f"當前使用者 手機號碼: {st.session_state.phone_number}")


# --- 只有在 user_id 存在時才顯示聊天介面 ---
if st.session_state.phone_number:
    # 這就是關鍵的修改！在顯示聊天輸入框之前，遍歷並顯示所有歷史訊息。
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): # 使用正確的角色 ('user' 或 'assistant')
            st.markdown(message["content"])

    # chat input for new messages
    user_query = st.chat_input("詢問有關預約的問題...")

    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        # 由於上面的 for 迴圈會在下一次 Streamlit 重新執行時渲染此訊息，
        # 這裡可以選擇是否立即顯示，但通常為了即時反饋會在這裡也顯示一次
        with st.chat_message("user"):
            #st.markdown(user_query)
            st.markdown(f'<div class="chat-message user">{user_query}</div>', unsafe_allow_html=True)


        try:
            payload = {
                'messages': st.session_state.messages, # 傳送完整的對話物件列表
                'phone_number': st.session_state.phone_number
            }
            response = requests.post(API_URL, json=payload, verify=False)

            if response.status_code == 200:
                api_response_content = response.json().get("messages", "No response message from API.")
                st.session_state.messages.append({"role": "assistant", "content": api_response_content})
                # 顯示 AI 的回應
                with st.chat_message("assistant"):
                    #st.markdown(api_response_content)
                    st.markdown(f'<div class="chat-message assistant">{api_response_content}</div>', unsafe_allow_html=True)

            else:
                error_message = f"Error {response.status_code}: Could not process the request. Details: {response.text}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.error(error_message)
        except Exception as e:
            exception_message = f"Exception occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": exception_message})
            with st.chat_message("assistant"):
                st.error(exception_message)





















# import streamlit as st
# import requests

# API_URL = "http://127.0.0.1:8003/execute" 

# st.title("🩺 Doctor Appointment System")

# user_id = st.text_input("Enter your ID number:", "")
# query = st.text_area("Enter your query:", "Can you check if a dentist is available tomorrow at 10 AM?")

# if st.button("Submit Query"):
#     if user_id and query:
#         try:
#             response = requests.post(API_URL, json={'messages': query, 'id_number': int(user_id)},verify=False)
#             if response.status_code == 200:
#                 response_data = response.json()
#                 st.success("Response Received:")
#                 print("**********my response******************")
#                 print(response_data)
#                 last_message = response_data["messages"][-1]
#                 last_content = last_message["content"]
#                 st.write(last_content)
#             else:
#                 st.error(f"Error {response.status_code}: Could not process the request.")
#         except Exception as e:
#             st.error(f"Exception occurred: {e}")
#     else:
#         st.warning("Please enter both ID and query.")