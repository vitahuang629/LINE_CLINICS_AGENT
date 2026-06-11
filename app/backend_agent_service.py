"""
專門給後端工程師串接用的 Agent Service
使用 FB Messenger PSID 識別使用者
整合圖片處理邏輯
"""
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from .models import BackendUserQuery, BackendResponse
from agent import DoctorAppointmentAgent
# from utils.profile_db import get_user_profile_by_uuid
from typing import List
import requests
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

agent = DoctorAppointmentAgent()

def search_clinic_images(query: str, top_k: int = 1, min_score: float = 0.7) -> List[str]:
    """呼叫 embedding API 搜尋相關圖片"""
    try:
        response = requests.post(
            'https://ai.gastom.com.tw/clinics_image_embedding_api/api/search',
            json={"query": query},
            headers={'accept': 'application/json', 'Content-Type': 'application/json'},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            print(f"API 回傳原始資料 (query={query}):", data)  # 新增的 Debug
            images = []
            for res in data.get('results', [])[:top_k]:
                print(f"檢查圖片: {res['url']}, 分數: {res.get('score', 0)}") # 新增的 Debug
                if res.get('score', 0) >= min_score:
                    images.append(res['url'])
            return images
        else:
            print(f"API 狀態碼錯誤: {response.status_code}, 內容: {response.text}")
    except Exception as e:
        print(f"Image search API error: {e}")
    return []

def extract_image_urls(text: str) -> List[str]:
    """從 AI 回覆中提取圖片 URL"""
    url_pattern = r'https://[^\s)>\]\'"]+\.(?:jpg|jpeg|png|webp)'
    urls = re.findall(url_pattern, text)
    return urls

def clean_text_from_urls(text: str, urls: List[str]) -> str:
    """移除圖片引用：markdown 圖片語法、裸 URL，以及只剩標籤的殘骸行。"""
    clean = text
    # 1) markdown 圖片語法：![alt](url) / ![alt]() / ![alt]（含 URL 被抽走後的空括號）
    clean = re.sub(r'!\[[^\]]*\](?:\([^)]*\))?', '', clean)
    # 2) 裸圖片 URL（抽出來的那些 + 保險再掃一次殘留的）
    for url in urls:
        clean = clean.replace(url, '')
    clean = re.sub(r'https?://[^\s)>\]\'"]+\.(?:jpg|jpeg|png|webp)', '', clean)
    # 3) 清掉「只剩 <圖片網址N>: / 條列符號 / 標點」的殘骸行
    kept = []
    for line in clean.split('\n'):
        residue = re.sub(r'<?\s*圖片\s*網址\s*\d*\s*>?', '', line)
        residue = re.sub(r'[\-\*\d\.\s:：()（）<>「」\[\]!]', '', residue)
        if residue == '':
            continue
        kept.append(line.rstrip())
    clean = '\n'.join(kept)
    # 4) 收掉多餘空行
    clean = re.sub(r'\n\s*\n+', '\n', clean).strip()
    return clean


def strip_markdown(text: str) -> str:
    """把 markdown 記號轉成 Messenger/LINE 能好好顯示的純文字（這些通路不渲染 markdown）。

    - 標題 `#` → 去掉
    - 條列 `- / * / +` → 轉成「• 」（保留縮排），數字清單 `1.` 保留
    - 粗體/斜體 `**x** / __x__ / *x* / _x_`、行內 code `` `x` `` → 去掉記號保留文字
    """
    if not text:
        return text
    lines = []
    for line in text.split('\n'):
        line = re.sub(r'^\s{0,3}#{1,6}\s*', '', line)          # 標題 ###
        line = re.sub(r'^(\s*)[-*+]\s+', r'\1• ', line)        # 條列項 → •（數字清單不動）
        lines.append(line)
    text = '\n'.join(lines)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)             # **粗體**
    text = re.sub(r'__([^_]+)__', r'\1', text)                 # __粗體__
    text = re.sub(r'\*([^*\n]+)\*', r'\1', text)               # *斜體*
    text = re.sub(r'`([^`]+)`', r'\1', text)                   # `code`
    return text


def determine_additional_images(ai_reply: str, user_query: str, extracted_urls: List[str]) -> List[str]:
    """
    根據 AI 回覆內容和使用者問題，決定要額外加入哪些圖片
    
    Args:
        ai_reply: AI 的回覆內容
        user_query: 使用者的最新問題
        extracted_urls: 從 AI 回覆中提取的圖片 URL
        
    Returns:
        完整的圖片 URL 列表
    """
    additional_images = []
    print('starttttttttttt pick an image')
    
    # 1. 自律神經檢測相關
    if "兩種方案" in ai_reply or "自律神經檢測" in ai_reply:
        additional_images.extend([
            "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/treatment_procedure.jpg",
            "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/autonomic_fees.jpg"
        ])


    # 3. 停車場/地址相關
    elif any(kw in ai_reply for kw in ["地址", "地點", "位於"]) or \
         any(kw in user_query for kw in ["哪裡", "停車", "停車場", "開車"]):
        additional_images.append("https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/parking_lots.jpg")
        

    # 3. 療程相關（使用 API 取得對應圖片取代 if-elif）
    else:
        # 定義哪些療程關鍵字需要用 API 找圖
        treatment_keywords = ["Emface", "NEO", "SIS", "瘦瘦筆", "週纖達", "EMBODY", "體態檢測", "冷凍", "冷脈衝"]
        
        # 找出 AI 回覆中提及的療程名稱
        mentioned_treatments = [kw for kw in treatment_keywords if kw.upper() in ai_reply.upper()]
        
        # 將療程正規化 (冷凍轉冷脈衝、瘦瘦筆轉週纖達) 並去重，直接存回 mentioned_treatments
        def normalize_treatment(t):
            if t == "冷凍": return "冷脈衝"
            if t == "瘦瘦筆": return "週纖達"
            return t
            
        mentioned_treatments = list(set(normalize_treatment(t) for t in mentioned_treatments))
        print('mentioned_treatments', mentioned_treatments)
        
        for treatment in mentioned_treatments:
            # Emface 專屬過濾邏輯保留：有圖、提到電波、沒講到免費，都不出圖
            if treatment.upper() == "EMFACE":
                print('emfaceeeeeeeeeeee')
                # Emface 介紹時固定帶出皺紋示意圖（說明可改善的紋路，例如法令紋），與 API 介紹圖獨立
                additional_images.append(
                    "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/rinkle_emface.jpg"
                )
                if extracted_urls or "電波" in ai_reply:
                    continue
                    
            # 針對特定療程，單一提到才出圖，一次講多個療程不出介紹圖
            exclusive_treatments = ["NEO", "SIS", "週纖達", "EMBODY", "冷脈衝"]
            if treatment in exclusive_treatments:
                mentioned_exclusive_count = sum(1 for t in mentioned_treatments if t in exclusive_treatments)
                if mentioned_exclusive_count > 1:
                    continue
            
            # 呼叫 API 取圖 (發送的 payload 將會是 {"query": "SIS"} 或是 {"query": "NEO"} 等等)
            print('treatment', treatment)
            api_images = search_clinic_images(treatment, top_k=1, min_score=0.7)
            print('api_images', api_images)
            if api_images:
                additional_images.extend(api_images)
    
    # 合併：先放 AI 提取的圖片，再放額外的圖片
    all_images = extracted_urls + additional_images
    
    # 去重（保持順序）
    seen = set()
    unique_images = []
    for img in all_images:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)
    
    return unique_images

def ocr_image_with_llm(image_url: str) -> str:
    """
    使用獨立的 LLM 進行純文字辨識 (OCR)。
    使用「沒有醫療背景」的 Prompt 來避開 OpenAI 的醫療圖片安全審查 (Safety Filter)。
    若圖片無文字，會回傳特殊 sentinel "NO_TEXT_FOUND"，由呼叫端負責濾掉。
    """
    try:
        # 使用獨立的 LLM 實例，確保不帶入醫美的 System Prompt
        ocr_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        messages = [
            HumanMessage(content=[
                {"type": "text", "text": (
                    "請幫我辨識並輸出這張圖片裡面的「所有文字內容」。"
                    "請不要解釋或描述圖片，只需要純粹輸出圖片中包含的文字即可。"
                    "若圖片裡完全沒有任何文字，請回傳這個確切字串：NO_TEXT_FOUND"
                    "（不要回傳「無」、「沒有」、「空字串」這類中文描述）。"
                )},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
            ])
        ]
        response = ocr_llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
    
CS_KEYWORDS = [
    "轉真人", "真人客服", "找真人", "找專人", "找店長", "找老闆",
    "找客服", "人工客服", "不要 AI", "不要ai", "不要機器人", "我要找人",
    # 「轉/要 + 專人/真人」等變體（補上「轉專人」「真人服務」這類漏接的講法）
    "專人", "轉接", "轉人工", "真人服務", "要真人",
    "客訴", "投訴", "抱怨", "退費", "退錢", "退款",
    "醫療糾紛", "申訴",
]


# 療程關鍵字清單（給費用 pre-filter 用）
TREATMENT_KEYWORDS = [
    # 體雕
    "NEO", "EMBODY", "SIS", "冷凍", "冷脈衝", "瘦瘦筆", "週纖達",
    # 臉部
    "Emface", "菲斯波", "無限電波", "皮秒",
    # 自律神經 / 睡眠
    "腦波", "EECP", "紅光", "止鼾", "NightLase",
]


def identify_treatments_from_context(user_query: str, message_history: list) -> list:
    """
    用 LLM 從「本輪訊息 + 歷史對話」判斷客人在問哪些療程（處理代名詞「這個」「那個」）。
    回傳的療程名必須是 TREATMENT_KEYWORDS 裡的，方便後續 dict 比對。
    """
    # 先做快速 keyword 比對（如果本輪訊息直接命中就不用打 LLM）
    direct_hits = [kw for kw in TREATMENT_KEYWORDS if kw.lower() in user_query.lower()]
    if direct_hits:
        print(f"[fee filter] Direct keyword match: {direct_hits}")
        return direct_hits

    # 本輪沒直接命中 → 用 LLM 看上下文判斷
    if not message_history:
        return []

    recent = message_history[:8]  # 最近 8 則
    history_lines = []
    for m in reversed(recent):  # 反轉成「舊到新」讓 LLM 讀順
        role = "客人" if m.type == "human" else "AI"
        history_lines.append(f"{role}: {m.content}")
    history_text = "\n".join(history_lines)

    prompt = (
        "以下是醫美客服對話。請判斷客人「最新訊息」中提及或暗指的療程。\n"
        "若客人用代名詞（這個、那個、它）或序數（第一個、第二個、前者、後者）：\n"
        "→ 從歷史對話「由新往回找」，找到最近一則有列出療程（含編號清單）的 AI 訊息，\n"
        "  再依代名詞／序數對照該清單，找出對應的那一個療程"
        "（注意可能不是上一則，而是更前面的訊息）。\n\n"
        f"歷史對話：\n{history_text}\n\n"
        f"客人最新訊息：{user_query}\n\n"
        f"從以下清單挑出「客人這句話實際指向的療程」"
        f"（不是主題相關的全部，逗號分隔，沒有就回 NONE）：\n"
        f"{', '.join(TREATMENT_KEYWORDS)}"
    )
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        text = (response.content or "").strip()
        if not text or text.upper() == "NONE":
            return []
        parsed = [t.strip() for t in text.split(",") if t.strip()]
        # 過濾掉 LLM 亂造的（不在 TREATMENT_KEYWORDS 裡的）
        valid = [t for t in parsed if t in TREATMENT_KEYWORDS]
        print(f"[fee filter] LLM context match: {valid}")
        return valid
    except Exception as e:
        print(f"[fee filter] LLM extraction error: {e}")
        return []


def filter_fees_by_treatments(treatments: list, treatment_fees: list) -> list:
    """
    根據療程關鍵字清單，從 treatment_fees 過濾出 name 包含任一關鍵字的行。
    純 dict 比對、無幻覺空間。
    """
    if not treatments or not treatment_fees:
        return []
    return [
        f for f in treatment_fees
        if any(kw.lower() in f.name.lower() for kw in treatments)
    ]


# ─────────────────────────────────────────────────────────────
# 療程幻覺檢查：以 search_clinics_by_keyword 的 retriever 結果為合法依據
# AI 回覆裡提到的療程，只要不在合法清單就視為幻覺
# ─────────────────────────────────────────────────────────────
from toolkit.toolkits import authorized_treatments_var, TREATMENT_SYNONYMS


# 方案/規格用詞 —— 這些是「同一療程的選項」，不是療程本身，抽取時要排除
_OPTION_WORDS = ("單點", "雙點", "多點", "單次", "套裝", "組合", "方案", "堂", "部位", "療程方案")


def extract_mentioned_treatments(text: str) -> list:
    """用 gpt-4o-mini 抽出文字裡提到的所有療程名"""
    prompt = (
        "以下是醫美客服的回覆。請列出文中所有提到的「醫美療程名稱」"
        "（只要療程名，不是部位、不是症狀、不是描述、不是檢測項目，"
        "也**不是方案/規格**——例如「單點 / 雙點 / 單次 / 套裝 / 組合 / 堂數」是同一療程的選項，不要列）。"
        "用逗號分隔。文中沒有療程就回 NONE。\n\n"
        f"回覆：\n{text}"
    )
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        if not result or result.upper() == "NONE":
            return []
        candidates = [t.strip() for t in result.split(",") if t.strip()]
        # 確定性過濾：把含方案/規格用詞的（如「單點治療」「雙點治療」）剔掉，避免誤判成幻覺療程
        return [t for t in candidates if not any(w in t for w in _OPTION_WORDS)]
    except Exception as e:
        print(f"[sanitize] extract error: {e}")
        return []


def sanitize_ai_response(ai_text: str) -> tuple:
    """
    依 retriever 結果檢查 AI 回覆是否亂提療程。
    回傳 (清理過的文字, 是否要觸發 CallCS=1)
    """
    authorized = authorized_treatments_var.get()
    if not authorized:
        # 本輪沒呼叫 search_clinics_by_keyword（例如客人問費用 / 預約）→ 跳過檢查
        return ai_text, False

    mentioned = extract_mentioned_treatments(ai_text)
    if not mentioned:
        return ai_text, False

    # 比對：mentioned 裡有但 authorized 沒有的 = 幻覺
    def _match(x: str, y: str) -> bool:
        # 寬鬆比對：互為子字串就算命中（沿用原本行為，處理「腦波機療程」這種帶後綴的講法）
        x, y = x.lower().strip(), y.lower().strip()
        return bool(x) and bool(y) and (x in y or y in x)

    def is_authorized(t: str) -> bool:
        # 1) 直接比對這一輪 retriever 撈到的合法療程
        if any(_match(t, a) for a in authorized):
            return True
        # 2) 別名比對：t 命中某個同義詞群組，且該群組有成員這輪真的被檢索到（grounding）
        #    → 正確的學名/別名（如 腦波機↔DeepTMS）放行；AI 亂拼的名稱因為不在任何群組裡，照樣被擋
        for group in TREATMENT_SYNONYMS:
            if any(_match(t, s) for s in group) and any(
                _match(s, a) for s in group for a in authorized
            ):
                return True
        return False

    unauthorized = [t for t in mentioned if not is_authorized(t)]
    if not unauthorized:
        return ai_text, False

    print(f"⚠️ [sanitize] 偵測到不合規療程: {unauthorized}（合法清單: {authorized}）")

    # 用 LLM 改寫，移除違規療程
    rewrite_prompt = (
        f"以下醫美客服回覆提到了我們診所**沒有**提供的療程：{', '.join(unauthorized)}\n"
        f"我們**只有**這些療程可以推薦：{', '.join(sorted(authorized))}\n\n"
        f"請改寫這段回覆：\n"
        f"1. **完全移除**不存在的療程相關描述（連同其句子 / 段落一起刪）\n"
        f"2. 只保留我們有的療程內容\n"
        f"3. 結尾改成自然的引導句（例如「請問您對哪一個療程比較有興趣？」）\n"
        f"4. 不要解釋你做了什麼修改，只輸出改寫後的內容\n\n"
        f"原回覆：\n{ai_text}\n\n改寫後："
    )
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        cleaned = llm.invoke([HumanMessage(content=rewrite_prompt)]).content.strip()
        print(f"✅ [sanitize] 改寫完成")
        return cleaned, False
    except Exception as e:
        print(f"❌ [sanitize] 改寫失敗: {e} → 觸發 CallCS=1 轉真人")
        return ai_text, True  # 改寫失敗 → 觸發轉真人

def execute_backend_agent(user_input: BackendUserQuery) -> BackendResponse:
    """
    執行 Agent 並回傳結構化的回應

    Args:
        user_input: 包含 fb_account、本輪 content / image_url、以及 message_history

    Returns:
        BackendResponse: 包含 text、images、CallCS 的回應
    """
    # 印出進來的 request（帶 timestamp 偵測重複請求）
    from datetime import datetime
    print(
        f"📥 [{datetime.now().isoformat()}] Request from Backend "
        f"fb_account={user_input.fb_account} "
        f"content={user_input.content!r} "
        f"image_count={len(user_input.image_url)} "
        f"history_len={len(user_input.message_history)}"
    )

    # 重設本 request 的「合法療程」集合（由 search_clinics_by_keyword 動態累積）
    authorized_treatments_var.set(set())

    app_graph = agent.workflow()

    # 1. 把歷史對話一筆筆轉成 LangChain 物件
    #    後端傳進來是由新到舊（DB 預設排序），這裡 reversed 還原成由舊到新
    langchain_messages: List[BaseMessage] = []
    for msg in reversed(user_input.message_history):
        if msg.type == "human":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            langchain_messages.append(AIMessage(content=msg.content))

    # ─── 注入療程費用 SystemMessage（pre-filter 防幻覺）───
    # Step 1: LLM 看上下文找出客人問的療程
    # Step 2: dict 比對 treatment_fees 過濾出對應行
    # Step 3: 只把過濾後的行給 AI，AI 物理上看不到無關價格，無法 hallucinate
    if user_input.treatment_fees:
        user_query_for_filter = (user_input.content or "").strip()
        mentioned_treatments = identify_treatments_from_context(
            user_query_for_filter, user_input.message_history
        )
        relevant_fees = filter_fees_by_treatments(
            mentioned_treatments, user_input.treatment_fees
        )

        common_rules = (
            "重要規則：\n"
            "1. 此區塊**只有療程體驗價**（實際做療程一次的費用）。\n"
            "   「初診費 / 諮詢檢測費」請走 search_clinics_info 工具查詢，不要從這裡取。\n"
            "2. 客人沒明確問價時，不要主動報價；先了解需求、推薦合適療程。\n"
            "3. 報體驗價時，**必須同時呼叫 search_clinics_info(treatment_name, \"初診\")** "
            "查詢該療程的初診詳情，整合回客人。\n"
            "4. 組合療程（name 含 \"+\"）整筆對應一個總價，直接報整筆即可。\n"
            "5. 檔期活動有時效性，講價格時可順帶提及但不硬推。\n"
        )

        if mentioned_treatments and not relevant_fees:
            # 客人問了具體療程，但費用表完全沒對應行 → 明確告訴 AI 「沒有」
            fee_note = (
                f"[費用資訊] 客人本輪詢問的療程（{', '.join(mentioned_treatments)}）"
                f"**目前沒有提供體驗價方案**。\n"
                f"請直接回覆：「目前 {' / '.join(mentioned_treatments)} 沒有提供體驗價方案，"
                f"可以先預約諮詢評估，由專人為您安排。」\n"
                f"（注意：**不要主動說「免費」**。客人若問諮詢是否要錢，請呼叫 search_clinics_info 查初診/諮詢費再回答。）\n"
                f"**嚴禁**自己編造價格，也**嚴禁**拿其他療程的價格套用上去。\n\n"
                + common_rules
            )
            print(f"[fee filter] 客人問 {mentioned_treatments} 但 fees 無對應 → 告訴 AI 沒有方案")
        elif relevant_fees:
            # 有對應行 → 只注入這幾行。標註單做/組合，方便 AI 不要把組合拆開。
            fee_lines = [
                f"- {f.name}：NT$ {f.price:,}"
                + ("（組合方案，整套價）" if "+" in f.name else "（單做價）")
                for f in relevant_fees
            ]
            # 防呆：只要這次撈到的行裡有組合（name 含「+」），就補一條硬性禁止拆解的規則，
            # 避免 AI 把「A + B = 總價」拆成 A、B 各自的單項價（會捏造數字）。
            has_combo = any("+" in f.name for f in relevant_fees)
            combo_guard = (
                "\n\n🚨 **組合方案防呆（嚴格遵守）**：\n"
                "- 上面標「組合方案」的那幾行，price 是**整套一起做**的總價，"
                "**嚴禁**拆成單一療程或單一部位的價格。\n"
                "- 例：「NEO + 冷凍單點 = 15,999」**不代表** NEO 單做 15,999，也**不代表**冷凍單點 15,999。\n"
                "- 只能照上面**完整的 name 原樣報價**；上面沒列出的單做價就是**沒有**，"
                "請回覆「該方案目前沒有單獨體驗價」，**嚴禁**自己換算或編造數字。\n"
            ) if has_combo else ""
            fee_note = (
                "[費用資訊] 以下是客人本輪詢問療程的相關體驗價：\n"
                + "\n".join(fee_lines)
                + combo_guard
                + "\n\n"
                + common_rules
            )
            print(f"[fee filter] 注入 {len(relevant_fees)} 行相關費用"
                  f"{'（含組合，已加防呆）' if has_combo else ''}")
        else:
            # 客人沒指定具體療程（例如問「有什麼優惠」「現在有什麼活動」）
            # → 不要丟全表，直接請 AI 反問客人有興趣的方向
            fee_note = (
                "[費用資訊] 客人本輪問費用/優惠/活動，但**沒有指定特定療程**。\n"
                "請反問客人：「我們有針對不同需求的療程方案，請問您比較想了解哪一塊？"
                "例如體態雕塑、臉部緊緻、私密保養、睡眠改善等。」\n"
                "**不要**主動列出全部療程或價格，等客人指定再給對應資訊。\n\n"
                + common_rules
            )
            print(f"[fee filter] 客人未指定療程 → 請 AI 反問")

        langchain_messages.insert(0, SystemMessage(content=fee_note))

    # 首次對話（無歷史）且有廣告來源 → 注入廣告 context 給 AI 當開場提示（放最前面）
    if not user_input.message_history and user_input.ad_referral:
        referral_note = (
            f"[系統提示] 客人是從「{user_input.ad_referral}」相關的 Meta 廣告進來的，"
            f"請主動詢問或介紹此療程，但語氣自然不要硬推。"
        )
        langchain_messages.insert(0, SystemMessage(content=referral_note))

    # 2. 處理本次客人的訊息 (含 OCR)
    #    content 和 image_url 互斥：客人傳文字 → content 有值、image_url 為空 list
    #                              客人傳圖片 → content 為 None、image_url 為一或多張
    # LLM 偶爾會把「無文字」描述成 "空字串"、"無" 等中文，這裡一併濾掉
    NO_TEXT_MARKERS = {"NO_TEXT_FOUND", "空字串", "無", "沒有", "沒有文字", "無文字", ""}
    ocr_texts: List[str] = []
    for url in user_input.image_url:
        print(f"執行 OCR 辨識圖片: {url}")
        text = ocr_image_with_llm(url)
        cleaned = (text or "").strip()
        if cleaned and cleaned not in NO_TEXT_MARKERS:
            ocr_texts.append(cleaned)
        else:
            print(f"  圖片無文字內容（OCR 回傳: {cleaned!r}）")

    # 短路：客人只傳圖、文字 OCR 抓不到任何內容（例如純臉照 / 純物件照）
    # → 不跑 LLM，直接回傳通用反問，避免 AI 亂猜或誤觸 safety filter
    has_image = bool(user_input.image_url)
    has_text = bool((user_input.content or "").strip())
    if has_image and not has_text and not ocr_texts:
        print("圖片無文字內容，回傳通用反問")
        return BackendResponse(
            text="請問您是想諮詢哪個部位呢？",
            images=[],
            CallCS=0,
        )

    final_text = (user_input.content or "").strip()
    if ocr_texts:
        joined = "\n---\n".join(ocr_texts)
        final_text += f"\n\n[客人上傳的圖片內容：]\n{joined}" if final_text else f"[客人上傳的圖片內容：]\n{joined}"
    if not final_text:
        final_text = "(沒有提供內容)"

    langchain_messages.append(
        HumanMessage(content=[{"type": "text", "text": final_text}])
    )

    # 準備查詢資料
    query_data = {
        "messages": langchain_messages,
        "fb_account": user_input.fb_account,   # FB Messenger PSID
        "next": "",
        "query": "",
        "current_reasoning": "",
    }

    # 執行 Agent（不使用 checkpointer，歷史由後端在 messages 內傳入）
    config = {
        "configurable": {
            "recursion_limit": 20
        }
    }
    response = app_graph.invoke(query_data, config=config)

    # 提取最終的 AI 回應
    final_ai_message_content = "抱歉，目前無法回覆。"
    if "messages" in response and response["messages"]:
        for msg in reversed(response["messages"]):
            if isinstance(msg, AIMessage):
                final_ai_message_content = msg.content
                break

    # 🛡️ 療程幻覺檢查 + 改寫（只在本輪呼叫過 search_clinics_by_keyword 時生效）
    final_ai_message_content, sanitize_force_handoff = sanitize_ai_response(final_ai_message_content)

    # 提取 AI 回覆中的圖片 URL
    extracted_urls = extract_image_urls(final_ai_message_content)
    
    # 取得使用者最新的問題（直接用 top-level content）
    user_query = user_input.content or ""
    # CallCS 三種值：0 = 正常 / 1 = 客人主動找真人客服（清空 text）/ 2 = 預約流程（保留 text）
    call_cs = 0

    # 療程幻覺改寫失敗 → 直接轉真人客服（最保守）
    if sanitize_force_handoff:
        call_cs = 1
    user_query_lower = user_query.lower()
    if any(kw.lower() in user_query_lower for kw in CS_KEYWORDS):
        call_cs = 1   # 最高優先：客人想轉真人 → 後端應清空 AI 回覆並通知客服

    # 預約流程：booking_node 偵測到本輪有呼叫 set_appointment → state["booking_completed"] = True
    # （tool_calls 被 booking_node 包起來不會出現在 response["messages"]，必須靠 state flag）
    if call_cs == 0 and response.get("booking_completed"):
        call_cs = 2   # 預約流程：保留 AI 預約引導文字，後端先發 text、再通知客服

    # 根據內容決定要加入哪些圖片
    all_images = determine_additional_images(final_ai_message_content, user_query, extracted_urls)

    # 去重：過濾掉「過去 AI / 真人客服已經發過」的圖片，避免同一張圖反覆推送
    already_sent_urls = set()
    for msg in user_input.message_history:
        if msg.type == "ai":
            already_sent_urls.update(extract_image_urls(msg.content))
    all_images = [img for img in all_images if img not in already_sent_urls]

    # 清理文字（移除圖片 URL）
    clean_text = clean_text_from_urls(final_ai_message_content, extracted_urls)
    clean_text = strip_markdown(clean_text)   # 去 markdown 記號，讓 Messenger/LINE 顯示乾淨

    # 【當確定有回傳停車圖片時，強制加上停車資訊】
    parking_image = "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/LINE_PHOTOS/parking_lots.jpg"
    if parking_image in all_images:
        if "春光公園" not in clean_text:
            clean_text = "🅿️ 停車資訊：可以到走5分鐘的永春停車場或是走8分鐘即可抵達的春光公園地下停車場。"

    # CallCS=1（客人主動找真人）→ 清空 text/images，純通知後端轉接
    # CallCS=2（預約流程）→ 保留 AI 的預約引導文字、清空 images，後端先發文字再通知客服
    # CallCS=0 → 正常回覆
    if call_cs == 1:
        clean_text = ""
        all_images = []
    elif call_cs == 2:
        all_images = []

    response_body = BackendResponse(
        text=clean_text,
        images=all_images,
        CallCS=call_cs,
    )

    # 印出回傳給後端的完整 JSON（debug 用）
    import json
    print(
        "📤 [Response to Backend]:\n"
        + json.dumps(response_body.model_dump(), ensure_ascii=False, indent=2)
    )

    return response_body

    