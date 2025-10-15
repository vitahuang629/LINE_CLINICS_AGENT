import pandas as pd
from typing import Literal, Dict, Any
from langchain_core.tools import tool
from data_models.models import DateModel, DateTimeModel, IdentificationNumberModel
import os
from utils.ensemble_retriever import get_ensemble_retriever
from utils.qa_retriever import get_qa_retriever
import random

print("toolkits.py is running")
qa_retriever = get_qa_retriever()
ensemble_retriever = get_ensemble_retriever()

@tool
def set_appointment(symptom: str) -> dict:
    """
    根據用戶輸入的文字，如果有想要預約療程的動機，
    接著詢問想要體驗的醫美療程，並回傳要詢問的欄位。
    """
    print(f"Appointment Tool called with: {symptom}")

    return {
        "appointment_info": (
            "請提供以下資訊：\n"
            "1. 預約姓名：\n"
            "2. 想做的療程：\n"
            "3. 期望時間：(mm/dd/yyyy)：\n"
            "4. 期望時間範圍（平日 / 假日 / 上午 / 下午）：\n"
            "5. 特殊需求（怕痛、敏感膚質、懷孕等）：\n"
            "6. 聯絡方式（電話、Email）：\n"
            "收到您的預約訊息後，我會盡快請專人幫您安排！"
        ),
        "should_terminate": True
    }

@tool
def search_clinics_by_keyword(symptom: str) -> str:
    """
    根據用戶輸入的症狀或關鍵字，根據這個症狀合理詢問大約兩次並關心用戶，接著查詢診所內相關醫美療程，並提供介紹說明。
    """
    print(f"Tool called with: {symptom}")


    # retriever = get_ensemble_retriever()  # ← 呼叫函式，取得 EnsembleRetriever 實體
    docs = ensemble_retriever.get_relevant_documents(symptom)
    # print('dddddddddddd', docs)

    if not docs:
        return f"目前找不到與「{symptom}」相關的療程。"

    results = [doc.page_content.strip() for doc in docs]
    # return "\n\n---\n\n".join(results)
    
    # ✅ 正確拼接成單一字串
    combined_results = "\n\n---\n\n".join(results)

    # ✅ 加上統一結尾
    return (
        f"{combined_results}\n\n"
        "這些療程皆能幫助改善您的狀況，"
        "建議您可以與本診所的專業醫師或諮詢師進一步討論，"
        "我也可以協助您預約本診所的療程喔！"
    )

@tool
def search_clinics_info(question: str) -> str:
    """
    根據用戶問診所資訊（地址、電話、初診費用），只回傳固定答案。
    """
    print(f"Info tool called with: {question}")


    # retriever = get_qa_retriever()  # ← 取得 Q&A retriever
    print('I got the retriever')
    docs = qa_retriever.get_relevant_documents(question)
    print('relatedddddddddddddddddd_docs', docs)

    if not docs:
        return f"抱歉，我找不到「{question}」的資訊。"

    # 只回傳答案，不加 AI 生成
    content = docs[0].page_content
    if "答案：" in content:
        answer = content.split("答案：", 1)[1].split("\n")[0].strip()
    else:
        answer = content.strip()

    # return answer
    return docs[0].page_content

@tool
def get_empathy_questions_by_symptom(symptom: str) -> dict:
    """
    根據用戶輸入的症狀，提供適當的同理心關懷語句 (1 句) 與需要追問的問題 (最多 2 個)。
    若症狀越具體，則提問越聚焦。
    - 關懷語句要溫柔簡短。
    - 追問的問題要開放式，但避免超過 2 個。
    - 如果 symptom 太模糊，只要問 1 個關鍵問題即可。
    """
    print(f"Empathy tool called with: {symptom}")


# 假設你有一份事先整理好的字典
    symptom_map = {
        "失眠": {
            "empathy": "睡不好真的很辛苦，影響整天的精神狀態。",
            "questions": [
                "想先了解一下，您比較常遇到的狀況是哪一種呢？\n1️⃣ 入睡困難\n2️⃣ 睡不安穩、容易醒\n3️⃣ 半夜驚醒或打呼、呼吸中斷",
                "平常這些狀況大概持續多久了呢？"
            ]
        },
        "痘痘": {
            "empathy": "痘痘冒出來一定很困擾，尤其會影響心情與自信。",
            "questions": [
                "目前痘痘大多是集中在哪些部位？",
                "最近有特別熬夜、壓力大或飲食改變嗎？",
                "你有嘗試使用藥膏或保養品處理嗎？"
            ]
        },
        "皺紋類": {
            "keywords": ["皺紋", "法令紋", "木偶紋"],
            "empathy": "皺紋的形成通常和膠原蛋白流失、表情肌活動或生活作息有關，很多人都會在意這部分的變化～別擔心，我們可以一起看看有哪些療程能幫助肌膚恢復緊緻與彈性。",
            "general_questions":[
                "想請問您皺紋主要集中在哪些部位呢？例如眼周、法令紋或額頭？",
                "這些皺紋是靜態的（平常就明顯）還是表情時才比較明顯呢？",
                "過去有嘗試過玻尿酸、肉毒或音波拉提等相關療程嗎？"
            ],
            "specific_questions": [
                "您提到的是{}，想了解它是靜態的還是表情時才會比較明顯呢？",
                "過去有嘗試過玻尿酸、肉毒或音波拉提等相關療程嗎？"
            ]
        },
        "胖": {
            "empathy": "讓我們來協助你。體重問題常讓人焦慮，但你願意聊已經是很好的開始。",
            "questions": [
                "平時有在運動嗎？",
                "飲食有特別控制嗎？"
            ]
        },
        "減脂": {
            "empathy": "讓我們來協助你。體重問題常讓人焦慮，但你願意聊已經是很好的開始。",
            "questions": [
                "請問本身有比較偏向哪些類型？\n1️. 無法控制飲食\n2️. 缺乏運動族群\n3️. 產後媽媽族群]\n4. 生活作息不規律\n5. 基因遺傳家族史\n6. 年紀漸長代謝下降 ",
                "有接觸過療程的經驗嗎？或是實際諮詢檢測評估過狀況呢？"
            ]
        },
        "打呼":{
            "empathy": "睡覺打呼不僅影響自己，也可能影響身邊的人。",
            "questions": [
                "之前有看過甚麼門診治療嗎？",
                "平時有側睡習慣嗎？"
            ]
        }
        # 其他症狀……
    }

    # 檢查是否與上次回應相同 10/14
    # if last_response and last_response.get("empathy"):
    #     #如果症狀相同且上次已經回應過，避免重複
    #     last_empathy = last_response.get("empathy", "")
    #     if any(keyword in symptom for keyword in ["皺紋", "法令紋", "木偶紋"]) and "皺紋的形成通常和膠原蛋白流失" in last_empathy:
    #         return {
    #             "empathy": "了解，讓我們繼續探討您的皺紋問題。",
    #             "questions": []  # 不問重複問題
    #         }

    # === 關鍵字比對邏輯 ===
    # for key, info in symptom_map.items():
    #     if "keywords" in info:  # 有多關鍵字設定的情況
    #         if any(k in symptom for k in info["keywords"]):
    #             return {
    #                 "empathy": info["empathy"],
    #                 "questions": info["questions"][:2] #最多2個問題
    #             }
    #     elif key in symptom:
    #         return {
    #             "empathy": info["empathy"],
    #             "questions": info["questions"][:2]
    #         }
    # --- 找出對應類別 ---
    for category, data in symptom_map.items():
        for keyword in data["keywords"]:
            if keyword in symptom:
                empathy = data["empathy"]

                # 根據具體程度分流
                if symptom == "皺紋":
                    # 模糊症狀 → 問範圍
                    return {
                        "empathy": empathy,
                        "questions": data["general_questions"]
                    }
                else:
                    # 具體症狀 → 問狀況/需求
                    specific_qs = [q.format(symptom) if "{}" in q else q for q in data["specific_questions"]]
                    return {
                        "empathy": empathy,
                        "questions": specific_qs
                    }
        
    
    fallbacks = [
        {
            "empathy": "我懂，這樣的狀況一定不好受。",
            "questions": ["方便多分享一些細節嗎？"]
        },
        {
            "empathy": "聽起來真的讓人困擾。",
            "questions": ["想請問大多是在什麼情況下發生呢？"]
        },
        {
            "empathy": "謝謝你願意分享。",
            "questions": ["能再說說具體的感受或影響嗎？"]
        }
    ]
    return random.choice(fallbacks)

    # return {
    #     "empathy": "我了解你目前的困擾。",
    #     "questions": ["可以再多說一點你的狀況嗎？"]
    # }


# @tool
# def check_availability(desired_date:DateModel, treatment_name:Literal['EECP', '腦波機', 'SIS', 'Emface', '蜂巢皮秒', 'Embody', 'NEO']) -> str:
#     """
#     查詢資料庫裡面是否有可以預約療程的時間。

#     參數：
#     desired_date (DateModel): 要查詢的日期和時間，格式為 "MM-DD-YYYY"。如果不提供，則查詢當天有空缺的時間。
#     treatment_name: 要查詢的療程名稱。

#     返回:
#     str: 可以預約的日期和時間。
#     """
#     print("Tool check availability")

#     base_dir = os.path.dirname(__file__)
#     parent_dir = os.path.dirname(base_dir)
#     csv_path = os.path.join(parent_dir, "data", "treatment_availability.csv")
#     df = pd.read_csv(csv_path, encoding="big5")    
#     df['date_slot_time'] = df['date_slot'].apply(lambda input: input.split(' ')[-1])
    
#     # print('desirteddddd', desired_date.date)
#     desired_day = desired_date.date
#     print('desired_day:', desired_day)

#     rows = list(
#         df[
#             (df['date_slot'].apply(lambda input: input.split(' ')[0]) == desired_day) &
#             (df['treatment'] == treatment_name) &
#             (df['is_available'] == True)
#         ]['date_slot_time']
#     )
#     print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr', rows)
#     if len(rows) == 0:
#         output = "一整天的預約都滿囉"
#     else:
#         output = f'可預約的日期 {desired_date.date}\n'
#         output += "可預約的時間: " + ', '.join(rows)

#     return output
# @tool
# def check_availability_by_specialization(desired_date:DateModel, specialization:Literal['EECP', '腦波機', 'SIS', 'Emface', '蜂巢皮秒', 'Embody', 'NEO']):
#     """
#     Checking the database if we have availability for the specific specialization.
#     The parameters should be mentioned by the user in the query
#     """
#     print("Tool check availability by specialization")

#     #Dummy data
#     base_dir = os.path.dirname(__file__)
#     parent_dir = os.path.dirname(base_dir)
#     csv_path = os.path.join(parent_dir, "data", "treatment_availability.csv")
    
#     df = pd.read_csv(csv_path, encoding="big5")
#     df['date_slot_time'] = df['date_slot'].apply(lambda input: input.split(' ')[-1])
#     rows = df[(df['date_slot'].apply(lambda input: input.split(' ')[0]) == desired_date.date) & (df['specialization'] == specialization) & (df['is_available'] == True)].groupby(['specialization', 'treatment'])['date_slot_time'].apply(list).reset_index(name='available_slots')

#     if len(rows) == 0:
#         output = "一整天的預約都滿囉"
#     else:
#         def convert_to_am_pm(time_str):
#             # Split the time string into hours and minutes
#             time_str = str(time_str)
#             hours, minutes = map(int, time_str.split(":"))
            
#             # Determine AM or PM
#             period = "AM" if hours < 12 else "PM"
            
#             # Convert hours to 12-hour format
#             hours = hours % 12 or 12
            
#             # Format the output
#             return f"{hours}:{minutes:02d} {period}"
#         output = f'This availability for {desired_date.date}\n'
#         for row in rows.values:
#             output += row[1] + ". Available slots: \n" + ', \n'.join([convert_to_am_pm(value)for value in row[2]])+'\n'

#     return output
# @tool
# def set_appointment(desired_date:DateTimeModel, id_number:IdentificationNumberModel, treatment_name:Literal['EECP', '腦波機', 'SIS', 'Emface', '蜂巢皮秒', 'Embody', 'NEO']) -> str:
#     """
#     幫客人預約療程。

#     參數：
#     desired_date (DateModel): 想要預約的日期和時間，格式為 "MM-DD-YYYY"。如果不提供，則查詢當天有空缺的時間。
#     id_number: 客人手機號碼
#     treatment_name: 要預約的療程名稱。

#     返回:
#     str: 預約成功的日期和時間。
#     """
#     print("set appointment")
#     try:
#         base_dir = os.path.dirname(__file__)
#         parent_dir = os.path.dirname(base_dir)
#         csv_path = os.path.join(parent_dir, "data", "treatment_availability.csv")
        
#         df = pd.read_csv(csv_path, encoding="big5")
#     except ValueError as ve:
#         print(f"pydantic 驗證錯誤: {ve}")
#     except Exception as e:
#         print(f"其他錯誤: {e}")
   
#     from datetime import datetime
#     def convert_datetime_format(dt_str):
#         dt = datetime.strptime(dt_str, "%m-%d-%Y %H:%M")
#         return dt.strftime("%m-%d-%Y %H:%M")

#     converted_date = convert_datetime_format(desired_date.date)
#     case = df[
#         (df['date_slot'] == converted_date) &
#         (df['treatment'] == treatment_name) &
#         (df['is_available'] == True)
#     ]

#     if len(case) == 0:
#         return "這個預約時段已滿喔"
#     else:
#         idx = df[
#             (df['date_slot'] == converted_date) &
#             (df['treatment'] == treatment_name) &
#             (df['is_available'] == True)
#         ].index

#         df.loc[idx, 'is_available'] = False
#         df.loc[idx, 'patient_to_attend'] = id_number.id

#         output_path = os.path.join(parent_dir, "data", "availability.csv")
#         df.to_csv(output_path, index=False, encoding="big5")

#         return f"預約成功！{treatment_name} 已預約在 {converted_date}"
# @tool
# def cancel_appointment(date:DateTimeModel, id_number:IdentificationNumberModel, treatment_name:Literal['EECP', '腦波機', 'SIS', 'Emface', '蜂巢皮秒', 'Embody', 'NEO']):
#     """
#     Canceling an appointment.
#     The parameters MUST be mentioned by the user in the query.
#     """
#     print("cancel appointment")

#     base_dir = os.path.dirname(__file__)
#     parent_dir = os.path.dirname(base_dir)
#     csv_path = os.path.join(parent_dir, "data", "treatment_availability.csv")
    
#     df = pd.read_csv(csv_path, encoding="big5")
#     case_to_remove = df[(df['date_slot'] == date.date)&(df['patient_to_attend'] == id_number.id)&(df['treatment'] == treatment_name)]
#     if len(case_to_remove) == 0:
#         return "You don´t have any appointment with that specifications"
#     else:
#         df.loc[(df['date_slot'] == date.date) & (df['patient_to_attend'] == id_number.id) & (df['treatment'] == treatment_name), ['is_available', 'patient_to_attend']] = [True, None]
#         df.to_csv(f'availability.csv', index = False)

#         return "Successfully cancelled"
# @tool
# def reschedule_appointment(old_date:DateTimeModel, new_date:DateTimeModel, id_number:IdentificationNumberModel, treatment_name:Literal['EECP', '腦波機', 'SIS', 'Emface', '蜂巢皮秒', 'Embody', 'NEO']):
#     """
#     Rescheduling an appointment.
#     The parameters MUST be mentioned by the user in the query.
#     """
#     print("reschedule appointment")

#     #Dummy data
#     base_dir = os.path.dirname(__file__)
#     parent_dir = os.path.dirname(base_dir)
#     csv_path = os.path.join(parent_dir, "data", "treatment_availability.csv")

#     df = pd.read_csv(csv_path, encoding="big5")
#     available_for_desired_date = df[(df['date_slot'] == new_date.date)&(df['is_available'] == True)&(df['treatment'] == treatment_name)]
#     if len(available_for_desired_date) == 0:
#         return "Not available slots in the desired period"
#     else:
#         cancel_appointment.invoke({'date':old_date, 'id_number':id_number, 'treatment':treatment_name})
#         set_appointment.invoke({'desired_date':new_date, 'id_number': id_number, 'treatment': treatment_name})
#         return "Successfully rescheduled for the desired time"
    


