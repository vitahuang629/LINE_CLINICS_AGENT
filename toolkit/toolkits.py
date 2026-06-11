import pandas as pd
from typing import Literal, Dict, Any
from contextvars import ContextVar
from langchain_core.tools import tool
from data_models.models import DateModel, DateTimeModel, IdentificationNumberModel
import os
from utils.ensemble_retriever import get_ensemble_retriever
from utils.qa_retriever import get_qa_retriever
from utils.consult_plan import get_consult_info
import random

print("toolkits.py is running")
qa_retriever = get_qa_retriever()
ensemble_retriever = get_ensemble_retriever()

# 每個 request 範圍的「合法療程」集合
# search_clinics_by_keyword 被呼叫時會把 retriever 回的療程名加進來
# 由 backend_agent_service 在 request 開始時 reset、結束時讀取
authorized_treatments_var: ContextVar = ContextVar("authorized_treatments", default=None)

# 療程同義詞 / 別名群組：每一組代表「同一個療程」的不同講法（正規名、學名、英文、暱稱）
# 用途：sanitize 檢查時，只要某一組裡有「正規名」這輪真的被 retriever 撈到（出現在
#       authorized_treatments_var），同組的其他別名就一起視為合法，避免把正確的學名/別名誤判為幻覺。
#       （例：腦波機這輪被檢索到 → 它的學名 DeepTMS / 深層經顱磁刺激 也放行）
# ⚠️ 維護原則：要和 agent.py 的「療程名稱白名單」（約 339-346 行）保持一致，新增療程兩邊都要改。
TREATMENT_SYNONYMS: list = [
    {"腦波機", "DeepTMS", "深層經顱磁刺激"},
    {"Emface", "菲斯波"},
    {"無限電波"},
    {"皮秒", "PicosurePRO"},
    {"NEO", "熱磁減脂"},
    {"EMBODY"},
    {"冷凍減脂", "冷脈衝"},
    {"SIS"},
    {"瘦瘦筆", "週纖達"},
    {"Alma Duo", "震波"},
    {"FemiLift"},
    {"G動椅", "Emsella"},
    {"EECP"},
    {"PBM", "紅光", "PBM紅光"},
    {"NightLase", "止鼾雷射", "NightLase無創雷射止鼾"},
]


# 診所基本資訊（地址 / 交通 / 看診時間 / 電話 / 停車）全擠在 clinics_qa「診所地點」那一筆。
# 短查詢（「地址」「怎麼去」）用模糊檢索常被別的 row 插隊甚至撈不到，
# 所以這類意圖直接回傳該筆，不賭向量 / BM25 排名。模組載入時抓一次。
def _load_clinic_basic_info():
    try:
        df = pd.read_csv("data/clinics_qa.csv", encoding="utf-8-sig")
    except Exception as e:
        print(f"[clinic_info] 載入 clinics_qa.csv 失敗：{e}")
        return None
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        cat = str(row.get("category", "")).strip()
        if q == "診所地點" or cat == "交通":
            return str(row.get("answer", "")).strip()
    print("[clinic_info] 找不到『診所地點』那筆（question=診所地點 / category=交通）")
    return None


CLINIC_BASIC_INFO = _load_clinic_basic_info()
print(f"[clinic_info] 診所基本資訊 {'已載入' if CLINIC_BASIC_INFO else '缺失'}")

# 命中以下任一意圖（出現在 agent 傳進來的 category）→ 直接回診所基本資訊那筆
CLINIC_INFO_INTENT = (
    "地址", "地點", "位置", "交通", "在哪", "在那",
    "怎麼去", "怎麼走", "怎麼到", "怎麼抵達", "搭車", "搭乘",
    "公車", "捷運", "開車", "停車", "電話", "聯絡",
    "看診", "營業", "幾點", "到達", "可以到嗎"
)


# 防呆用：症狀「分類標籤」→ 療程資料庫裡真實會出現的症狀詞。
# 這些標籤（皺紋類、皮膚其他…）是給 get_empathy_questions_by_symptom 用的分類桶，
# 標籤字串本身不在任何療程內文/關鍵字裡。若 LLM 誤把標籤傳進 search_clinics_by_keyword，
# 直接拿去檢索會撈不到正確療程（例：傳「皺紋類」會把 Emface 排掉），
# 故在檢索前先把標籤展開成真實症狀詞。詞彙對齊 clinics_introductions3.csv 的 keywords。
TAG_TO_KEYWORDS: dict = {
    "皺紋類": "法令紋 木偶紋 細紋 皺紋 紋路 抬頭紋 眼尾紋 嘴角細紋",
    "皮膚其他": "痘痘 斑點 膚色不均 肌膚暗沉 毛孔",
    "體態管理": "減脂 瘦身 局部雕塑 肚子 體態",
    "私密療程": "私密處 性功能 緊緻",
    "睡眠與神經": "失眠 睡不好 打呼 自律神經 壓力 心悸",
}


@tool
def set_appointment(symptom: str) -> dict:
    """
    第一階段：當客人「首次表達」預約意願時呼叫此工具，
    回傳一段要客人填寫的預約欄位（姓名、療程、時間、特殊需求、電話）。
    呼叫此工具**不會**觸發轉真人客服，只是顯示表單。
    """
    print(f"set_appointment called with: {symptom}")

    return {
        "appointment_info": (
            "好的！請提供以下資訊我幫您安排：\n"
            "1. 預約姓名：\n"
            "2. 想做的療程：\n"
            "3. 期望日期及時間：(mm/dd)：\n"
            "4. 特殊需求（怕痛、敏感膚質、懷孕等）：\n"
            "5. 聯絡電話："
        ),
        "should_terminate": True
    }


@tool
def confirm_booking(
    name: str,
    treatment: str,
    datetime_pref: str,
    contact: str,
    special_needs: str = "",
) -> dict:
    """
    第二階段：當客人**已經填好預約資訊**（提供姓名、療程、時間、聯絡電話等）時呼叫此工具，
    用於記錄預約意願並觸發轉接真人客服。

    參數：
    - name: 預約姓名
    - treatment: 想做的療程
    - datetime_pref: 期望日期及時間
    - contact: 聯絡電話
    - special_needs: 特殊需求（選填）

    呼叫此工具會觸發轉接真人客服（CallCS=2）。
    """
    print(
        f"confirm_booking called: name={name}, treatment={treatment}, "
        f"time={datetime_pref}, contact={contact}, special_needs={special_needs}"
    )

    return {
        "confirmation": "好的，已收到您的預約資訊，這邊幫您轉接專人進行後續安排，請稍候！",
        "should_terminate": True,
    }

@tool
def search_clinics_by_keyword(symptom: str) -> str:
    """
    根據用戶輸入的症狀或關鍵字，根據這個症狀合理詢問大約兩次並關心用戶，接著查詢診所內相關醫美療程，並提供介紹說明。
    若偵測到使用者在問對比照或是療程效果，需先確認療程項目再回答。
    """
    print(f"Tool called with: {symptom}")

    # 防呆：LLM 若誤傳分類標籤（如「皺紋類」），展開成資料庫真實症狀詞再檢索，
    # 避免標籤字串檢索不到正確療程。正常傳原始症狀詞（如「法令紋」）則不受影響。
    if symptom in TAG_TO_KEYWORDS:
        expanded = TAG_TO_KEYWORDS[symptom]
        print(f"[防呆] 偵測到分類標籤「{symptom}」→ 展開為「{expanded}」")
        symptom = expanded

    # retriever = get_ensemble_retriever()  # ← 呼叫函式，取得 EnsembleRetriever 實體
    docs = ensemble_retriever.get_relevant_documents(symptom)
    # print('dddddddddddd', docs)

    if not docs:
        return f"目前找不到與「{symptom}」相關的療程。"

    # 把這次 retriever 撈到的療程名收進 ContextVar（給後處理檢查用）
    current_set = authorized_treatments_var.get()
    if current_set is None:
        current_set = set()
    for doc in docs:
        name = doc.metadata.get("clinic", "")
        if name:
            current_set.add(name)
    authorized_treatments_var.set(current_set)
    print(f"[authorized treatments] Updated: {current_set}")

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
# def search_clinics_info(question: str, category: str = "費用") -> str:
def search_clinics_info(treatment_name: str, category: str = "費用") -> str:

    # """
    # 根據用戶問診所資訊（地址、電話、療程初診費用），
    # 若偵測到使用者在問費用、體驗價或初診，需先確認療程項目再回答。
    # """
    """
    查詢診所特定療程的資訊。
    參數:
    - treatment_name: 療程名稱 (例如: 瘦瘦筆, EMBODY, NEO, Emface)
    - category: 查詢類別 (費用, 地址, 電話)
    """
    print(f"Target Treatment: {treatment_name}, Category: {category}")

    # 初診 / 諮詢 → 走 consult_plan.csv 結構化查表（免費/收費是明確欄位，不用語意檢索）
    if "初診" in category or "諮詢" in category:
        consult = get_consult_info(treatment_name, TREATMENT_SYNONYMS)
        if consult:
            return consult
        print(f"[consult] 無對應，fallback 回 qa_retriever：{treatment_name}")

    # 地址 / 交通 / 電話 / 看診時間 / 停車 → 直接回「診所地點」那筆，不賭模糊檢索排名
    if any(k in category for k in CLINIC_INFO_INTENT):
        if CLINIC_BASIC_INFO:
            print(f"[clinic_info] category='{category}' 命中診所資訊意圖 → 直接回傳")
            return CLINIC_BASIC_INFO
        print("[clinic_info] 缺診所基本資訊，fallback 回 qa_retriever")

    boosted_query = f"{category} {category} {category}"
    print(f"Boosted Query: {boosted_query}")


    # --- Step 2. 呼叫 retriever ---
    docs = qa_retriever.get_relevant_documents(boosted_query)
    if not docs:
        print(f"[qa] 檢索無結果：{boosted_query}")
        return f"抱歉，我找不到「{boosted_query}」的資訊。"

    # --- 觀測：印出檢索到的前幾筆（排序、類別、內容開頭）---
    print(f"[qa] 檢索「{boosted_query}」回傳 {len(docs)} 筆：")
    for i, d in enumerate(docs):
        cat = d.metadata.get("category", "?")
        head = d.page_content.replace("\n", " ")[:80]
        print(f"  [{i}] category={cat} | {head}")

    # --- Step 3. 只回傳答案 ---
    content = docs[0].page_content
    print(f"[qa] 採用 [0] 回傳，長度 {len(content)}")
    return content



@tool
def get_empathy_questions_by_symptom(symptom_tag: str) -> dict:
    """
    根據用戶輸入的症狀，提供適當的同理心關懷語句 (1 句) 與需要追問的問題 (1 個)。
    若症狀越具體，則提問越聚焦。
    - 關懷語句要溫柔簡短。
    - 追問的問題要開放式，但避免超過 1 個。
    - 如果 symptom 太模糊，只要問 1 個關鍵問題即可。
    """
    print(f"Empathy tool called with: {symptom_tag}")

    symptom_map = {
        "皺紋類": {
            "empathy": "我很理解您對細紋或紋路的在意，這確實是許多人追求自信時最關注的細節。",
            "questions": ["過去有嘗試過相關的緊緻療程嗎？", "主要集中在哪一個部位？"]
        },
        "私密療程": {
            "empathy": "私密處的保養與健康確實非常重要，謝謝您願意信任並與我分享。 ",
            "questions": ["您主要是想了解功能改善，還是日常的美觀保養呢？"]
        },
        "睡眠與神經": {
            "empathy": "長期睡不好或壓力大對身心負擔真的很高，我們會陪您一起找回舒適的休息品質。",
            "questions": ["""方便進一步了解，請問目前有出現以下這些狀況嗎？/n
                          1️⃣  睡不好、淺眠易醒、入睡困難/n2️⃣  睡覺會打呼、有呼吸中止情況/n
                            3️⃣  長期依賴藥物，副作用明顯/n
                            4️⃣  情緒緊繃、緊張焦慮不安/n
                            5️⃣  心跳偏快、容易胸悶心悸/n
                            6️⃣  記憶力下降、注意力變差/n
                            7️⃣  頭痛、頭暈、耳鳴常發作/n
                            8️⃣  胃食道逆流、脹氣、消化不良/n
                            9️⃣  經常累沒精神、莫名身體痠痛""",
                        "這種狀況持續多久了？會想了解如何透過腦波偵測來找出原因嗎？"]
        },
        "體態管理": {
            "empathy": "體態調整需要耐心與科學方法，願意開始了解就是很棒的第一步。",
            "questions": ["了解您的情況了。為了更精確建議，請問您目前是偏向飲食習慣、運動缺乏，還是代謝問題比較困擾您呢？"]
        },
        "皮膚其他": {
            "empathy": "皮膚出現痘痘或斑點確實讓人困擾，我們會協助您找回肌膚的健康狀態。",
            "questions": ["目前的狀況大約持續多久了？有嘗試過什麼處理方式嗎？例如:保養品或曾做過其他醫美療程"]
        }
    }

    return symptom_map.get(symptom_tag, {
        "empathy": "謝謝您的分享，我非常理解您的困擾。",
        "questions": ["能再幫我多描述一下目前的狀況嗎？"]
    })

