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
from time import perf_counter
import requests
import re
# token 用量統計：contextvar 型的 callback，已實測會穿透 LangGraph 節點與巢狀子圖
# （booking 的 create_react_agent），所以 graph 內外所有 OpenAI 呼叫都算得到。
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# 逾時設定集中在 utils/llms.py（那裡有為什麼一定要設的完整說明）。
# 本檔下面三處是繞過 LLMModel、直接建 ChatOpenAI 的呼叫，同樣要帶上，否則會是無限等待。
from utils.llms import LLM_TIMEOUT, LLM_MAX_RETRIES

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


    # 停車 / 地址圖：不再寫死單一圖。兩間分店（台北 / 竹北）的停車圖已分別嵌在 clinic_qa
    # 答案裡（「圖片: https://...png」），由上游 extract_image_urls 從回覆文字自動抓出，
    # 這裡不再處理，避免貼到錯的分店或過期的圖。

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
        ocr_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )
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
    # 取消 / 改約 → 交給真人客服處理（用較完整片語，避免「取消」單字誤判其他句子）
    "取消預約", "取消約", "取消約診", "取消我的預約",
    "改約", "改預約", "改時間", "改期", "更改預約", "更改時間", "重新預約",
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


def _treatment_tokens_from_fees(treatment_fees: list) -> list:
    """
    從 DB 費用 name 拆出可比對的療程詞，讓**新增到費用 DB 的療程不必手動補進
    TREATMENT_KEYWORDS** 也能被費用 pre-filter 命中（混合策略：DB 為主、清單補同義詞）。

    - 拆「+」組合（「NEO + 冷凍單點」→ NEO、冷凍單點）
    - 先去括號規格（「冷凍減脂(60分鐘-單點)」→「冷凍減脂」），再去方案/規格用詞，還原成原子療程詞
    - 過濾單字元，避免拆出太短的詞造成誤命中

    ⚠️ 拆出的詞會被 filter_fees_by_treatments 拿去對 fee name 做**子字串**比對，
       所以必須保證是原始 name 的乾淨子字串：只去括號規格與頭尾標點，
       **不可**清掉詞內連字（如「NEO-熱磁減脂」），否則反而對不回原始 name。
       （曾出過 bug：只去 option 詞、沒去括號 → 拆出「冷凍減脂(60分鐘-)」對不到任何 fee name → relevant_fees 空掉）
    """
    tokens = set()
    for f in treatment_fees or []:
        name = (getattr(f, "name", "") or "").strip()
        if not name:
            continue
        for part in name.split("+"):
            t = part.strip()
            t = re.sub(r"[（(][^）)]*[）)]", "", t)   # 先整段去掉括號內規格（如「(60分鐘-單點)」）
            for w in _OPTION_WORDS:
                t = t.replace(w, "")
            t = t.strip(" -–—~、,，.。/·")             # 只清頭尾標點/破折號，保留詞內連字
            if len(t) >= 2:
                tokens.add(t)
    return sorted(tokens)


def identify_treatments_from_context(
    user_query: str, message_history: list, treatment_fees: list = None
) -> list:
    """
    用 LLM 從「本輪訊息 + 歷史對話」判斷客人在問哪些療程（處理代名詞「這個」「那個」）。
    回傳的療程名必須在「比對詞庫」內，方便後續 dict 比對。

    比對詞庫 = 手寫 TREATMENT_KEYWORDS（含同義詞/中英對照/組合拆詞）
              ∪ DB 費用名稱拆出的療程詞（新增療程自動納入，不必改程式碼）。
    """
    # 比對詞庫：手寫清單 ∪ DB 費用名稱拆出的詞（去重、保序）
    keyword_pool = list(dict.fromkeys(TREATMENT_KEYWORDS + _treatment_tokens_from_fees(treatment_fees)))

    # 先做快速 keyword 比對（如果本輪訊息直接命中就不用打 LLM）
    direct_hits = [kw for kw in keyword_pool if kw.lower() in user_query.lower()]
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
        f"{', '.join(keyword_pool)}"
    )
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        text = (response.content or "").strip()
        if not text or text.upper() == "NONE":
            return []
        parsed = [t.strip() for t in text.split(",") if t.strip()]
        # 過濾掉 LLM 亂造的（不在比對詞庫裡的）
        valid = [t for t in parsed if t in keyword_pool]
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
# 輸出端「價格守門」：pre-filter 只洗乾淨 [費用資訊]，洗不掉對話歷史裡別的療程殘留價，
# AI 仍可能把歷史中「別的療程的真實價」套到本療程頭上（跨療程張冠李戴）。
# 這裡對「最終回覆」再驗一次：抓出所有價格 → 若某價是「別的療程的真實價、卻不在本療程合法價內」
# → 判定捏造，讓 AI 帶著正確價重寫一次；重寫後仍錯（或無法回答）才轉真人。
# ─────────────────────────────────────────────────────────────

# 只抓「像價格」的數字：NT$ 前綴、或帶千分位逗號（如 16,800）。
# 「30分鐘」「6/25」「0912345678」不帶逗號也無 NT$ 前綴，不會被誤抓。
_PRICE_RE = re.compile(r"(?:NT\$?|＄|\$)\s*([\d][\d,]*)|(\b\d{1,3}(?:,\d{3})+\b)")


def _extract_prices(text: str) -> set:
    """從文字抽出所有價格數字（去逗號後轉 int）。"""
    out = set()
    for m in _PRICE_RE.finditer(text or ""):
        raw = (m.group(1) or m.group(2) or "").replace(",", "")
        if raw.isdigit():
            out.add(int(raw))
    return out


def _resolve_reply_treatments(text: str) -> list:
    """從 AI 回覆本身掃出提到的療程（用 TREATMENT_SYNONYMS），回傳可供 filter_fees 比對的別名清單。
    命中某別名就把整組別名都放進來（fee 表 name 可能用組內另一種寫法），去重保序。

    用途：fee filter 對序數／代名詞（如「第一個療程」）解析失敗、relevant_fees 空掉時，
    改以「AI 回覆實際在講的療程」回頭查真表，讓價格守門的偵測與重寫都有正確依據。
    """
    low = (text or "").lower()
    hits = []
    for group in TREATMENT_SYNONYMS:
        if any(alias.lower() in low for alias in group):
            hits.extend(group)
    return list(dict.fromkeys(hits))


def _format_fee_lines(fees: list) -> str:
    return "\n".join(
        f"- {f.name}：NT$ {f.price:,}"
        + ("（組合方案整套價）" if "+" in f.name else "（單做價）")
        for f in fees
    )


def _rewrite_price_reply(draft: str, treatments: list, relevant_fees: list, leaked: set) -> str:
    """帶著本療程正確價，讓 AI 重寫一次。無法在不編價下回答 → 回 [[HANDOFF]]。"""
    allowed_txt = _format_fee_lines(relevant_fees) if relevant_fees else "（本療程目前無任何體驗價方案）"
    t = " / ".join(treatments) if treatments else "客人詢問的療程"
    sys = (
        "你剛才給客人的回覆裡，報了『不屬於本療程』的價格——很可能是把對話歷史中「其他療程」的價格套了過來。\n"
        f"本輪客人問的療程是：{t}。\n"
        f"本療程真正、且唯一可用的體驗價如下：\n{allowed_txt}\n\n"
        f"被判定錯誤的價格：{', '.join(f'NT$ {p:,}' for p in sorted(leaked))}\n\n"
        "請重寫要給客人的回覆，規則：\n"
        "1. 療程『體驗價』只能用上面列出的；被判定錯誤的價格必須移除或改正，**絕不可保留**。\n"
        "2. 若上面顯示本療程沒有體驗價方案，就誠實說明目前沒有體驗價、可先預約諮詢評估，不要報任何體驗價。\n"
        "3. 除了上面列出的體驗價，**不要生出任何新的療程價格數字**（初診/諮詢費等非體驗價資訊可照舊保留）。\n"
        "4. 其餘關懷、推薦、預約引導、網址／圖片連結、emoji、換行一律原樣保留。\n"
        "5. 若你無法在不編造價格的情況下回答客人本來的問題，**只輸出這一行**：[[HANDOFF]]（前後不要有任何其他字）。\n"
        "只輸出最終要給客人的純文字，不要任何解釋。"
    )
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )
        resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=draft)])
        return (resp.content or "").strip() or draft
    except Exception as e:
        print(f"[price guard] 重寫失敗: {e} → 維持原文、交由上層轉真人")
        return "[[HANDOFF]]"


# ─────────────────────────────────────────────────────────────
# 療程幻覺檢查（faithfulness）：實際核對已移到 graph 的 moderator_node，
# 以本輪 retriever 撈到的「原始內容」為唯一事實來源。
# 這裡只負責在每個 request 開始時重設下列 context var（供 moderator 讀取）。
# ─────────────────────────────────────────────────────────────
from toolkit.toolkits import (
    authorized_treatments_var,
    grounded_content_var,
    treatment_fees_var,
    TREATMENT_SYNONYMS,
    register_grounded_content,
)


# 方案/規格用詞 —— 這些是「同一療程的選項」，不是療程本身，抽取時要排除
_OPTION_WORDS = ("單點", "雙點", "多點", "單次", "套裝", "組合", "方案", "堂", "部位", "療程方案")


def _execute_backend_agent_inner(user_input: BackendUserQuery) -> BackendResponse:
    """
    執行 Agent 並回傳結構化的回應

    ⚠️ 這是內層實作，外部請呼叫 execute_backend_agent（它會在 trace 補上
       usage / elapsed_ms / grounding_* 三組觀測欄位，並印出 📤 log）。

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
    # 重設本 request 的「事實來源」清單（由各 retriever 動態累積成有序、去重的 chunk list，
    # 供 sanitize faithfulness 核對）
    grounded_content_var.set([])
    # 把「整張費用表」原封不動存入（不預先 filter）；由 booking 的 get_treatment_fee 工具
    # 在認出療程的當下才 filter。取代舊的 graph 前弱解析預注入，避免代名詞/序數解析失敗導致編價。
    treatment_fees_var.set(user_input.treatment_fees or [])

    app_graph = agent.workflow()

    # 1. 把歷史對話一筆筆轉成 LangChain 物件
    #    後端傳進來是由新到舊（DB 預設排序），這裡 reversed 還原成由舊到新
    langchain_messages: List[BaseMessage] = []
    for msg in reversed(user_input.message_history):
        if msg.type == "human":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            langchain_messages.append(AIMessage(content=msg.content))

    # ─── 療程體驗價：已改為「工具化」，不再於 graph 前預注入 [費用資訊] ───
    # 整張費用表已存入 treatment_fees_var（見上方）；booking 的 get_treatment_fee 工具會在
    # 「認出療程的當下」才 filter，用 booking 的可靠解析取代舊的 graph 前弱解析預注入
    # （舊做法對代名詞/序數會解析失敗、relevant_fees 空掉 → AI 沒價可用而編價）。
    # 下面兩個變數保留空初始化，供輸出端「價格守門」引用（守門主要靠 reply_fees 從回覆回推真價）。
    mentioned_treatments: list = []
    relevant_fees: list = []
    if not user_input.treatment_fees:
        print("[fee filter] ⚠️ 本次 request 未帶 treatment_fees → get_treatment_fee 查不到任何體驗價")

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
        "trace": {},   # 各節點累積寫入的評估軌跡（LLM-as-judge 用）
    }

    # 執行 Agent（不使用 checkpointer，歷史由後端在 messages 內傳入）
    config = {
        # recursion_limit 算的是「單次 invoke 內 graph 走了幾個 super-step」，
        # 跟對話輪數／message_history 長度無關（本服務 stateless，每次 /chat 都是全新 invoke）。
        #
        # 外層 graph 是無環 DAG（start→guard→supervisor→information/booking→moderator→END），
        # 最長 5 步，根本碰不到上限。真正會迴圈的是 booking_node 裡的 create_react_agent
        # （model→tools→model→…，每輪 tool call 約吃 2 步），這個值實際上是在管它。
        # 內層子圖會透過 contextvar 繼承這裡的值，即使 booking_agent.invoke() 沒傳 config。
        #
        # ⚠️ 一定要放**頂層**。放進 "configurable" 會被當成一般 configurable 值忽略，
        #    實際套用的是預設 25 —— 舊寫法就是這樣，設的 20 完全沒生效。
        #
        # 取 12（≈5~6 輪 tool call）而非預設 25 的理由：LLM 呼叫已設 60 秒 timeout
        # （見 utils/llms.py），25 步最壞情況會讓單一 request 拖到十幾分鐘、
        # 期間一直佔著一條 threadpool thread，而客人早就走了。
        # 撞到上限本來就代表這通已經歪了，早點丟 GraphRecursionError 讓下面的 🛡️ 兜底接手
        # （回罐頭 → 再崩才轉真人）反而比較好，也省 token。
        "recursion_limit": 12,

        # ─── LangSmith 觀測用 ───────────────────────────────────────────
        # 沒有 metadata 就只能一筆一筆點開看 trace，無法分群比較。
        # 這幾個維度是用來回答「什麼樣的請求特別貴／特別慢」——
        # 實務上 token 大戶不是問題本身（客人一句話能多長），而是
        # ①history_len 累積的對話 ②檢索塞進 prompt 的 chunk 量，所以優先記這些。
        # ⚠️ 不要放 content 本文或任何個資：trace 的 inputs 本來就存得到，
        #    metadata 是拿來當篩選維度的，重複塞進去只是讓列表變難讀。
        "run_name": "clinic_chat",
        "metadata": {
            "history_len": len(user_input.message_history),
            "has_image": bool(user_input.image_url),
            "image_count": len(user_input.image_url),
            "ocr_text_len": sum(len(t) for t in ocr_texts),
            "ad_referral": user_input.ad_referral or "",
            "is_first_turn": not user_input.message_history,
            "fee_table_size": len(user_input.treatment_fees or []),
        },
    }
    # 🛡️ 兜底：graph 執行期間任何未預期例外（工具參數驗證失敗、OpenAI 抽風/逾時、
    #    recursion limit…）都在此接住，避免整個 request 變成 HTTP 500、客人已讀不回。
    #    回覆內容都在 graph 裡生成、崩潰風險幾乎都在這；後處理唯二會炸的外部呼叫
    #    （價格守門 LLM 重寫、圖片 embedding API）本身已各自有 try/except。
    #    兩段式降級（stateless，靠後端傳來的 message_history 判斷）：
    #      首次崩潰          → 回罐頭請客人重述（CallCS=0，不轉真人）
    #      客人重述後又崩潰  → 轉真人（CallCS=1），避免卡在「請再說一次」鬼打牆
    RETRY_CANNED = "不好意思，麻煩您再訴說一次您的問題，謝謝！"
    try:
        response = app_graph.invoke(query_data, config=config)
    except Exception as e:
        import traceback
        print(
            f"❌ [{datetime.now().isoformat()}] graph 崩潰 "
            f"fb_account={user_input.fb_account}: {e}"
        )
        traceback.print_exc()   # 完整堆疊留在後端 log，方便事後抓真正的 bug
        # message_history 由後端傳來、順序「由新到舊」→ 第一個 type=="ai" 即上一輪 AI 回覆。
        last_ai = next(
            (m.content for m in user_input.message_history if m.type == "ai"),
            None,
        )
        if last_ai and RETRY_CANNED in last_ai:
            # 上一輪已回過罐頭、客人重述後又炸 → 可能換句話說也繞不過 → 轉真人
            return BackendResponse(
                text="",
                images=[],
                CallCS=1,
                trace={"fatal_error": str(e), "handoff_reason": "fatal_error_repeat"},
            )
        return BackendResponse(
            text=RETRY_CANNED,
            images=[],
            CallCS=0,
            trace={"fatal_error": str(e)},
        )

    # 提取最終的 AI 回應
    final_ai_message_content = "抱歉，目前無法回覆。"
    if "messages" in response and response["messages"]:
        for msg in reversed(response["messages"]):
            if isinstance(msg, AIMessage):
                final_ai_message_content = msg.content
                break

    # 🛡️ 療程事實核對（faithfulness）已併入 graph 的 moderator_node：
    #    moderator 本輪有檢索到療程內容時會用 gpt-4o 核對事實、修正無依據的描述；
    #    核對失敗時回傳 force_handoff=True，這裡讀出來決定是否轉真人。
    sanitize_force_handoff = bool(response.get("force_handoff", False))

    # 🛡️ 輸出端價格守門：抓回覆裡的價格，若出現「別的療程的真實價、卻不在本療程合法價內」
    #    （典型：把對話歷史裡別療程的價格套過來）→ 讓 AI 帶正確價重寫一次；第二次仍錯才轉真人。
    price_guard_handoff = False
    price_guard_trace = None
    all_table_prices = {f.price for f in (user_input.treatment_fees or [])}
    reply_prices = _extract_prices(final_ai_message_content)
    # 合法價集合 = fee filter 注入的 relevant_fees ∪「AI 回覆實際提到的療程」回頭查表的真實 fees。
    # 後者補救 fee filter 對序數／代名詞（如「第一個療程」）解析失敗、relevant_fees 空掉的情況：
    # 直接從 AI 回覆認出療程（如冷凍減脂）查它的真實價 → 偵測不誤殺正確價、重寫也拿得到正確價。
    reply_treatments = _resolve_reply_treatments(final_ai_message_content)
    reply_fees = filter_fees_by_treatments(reply_treatments, user_input.treatment_fees)
    guard_fees = []
    _seen_fee = set()
    for f in (relevant_fees + reply_fees):
        key = (f.name, f.price)
        if key not in _seen_fee:
            _seen_fee.add(key)
            guard_fees.append(f)
    allowed_prices = {f.price for f in guard_fees}
    leaked = (reply_prices & all_table_prices) - allowed_prices   # 屬別療程的真實價、非本療程合法價
    if leaked:
        print(f"[price guard] 偵測到不屬本療程的價格 {sorted(leaked)}"
              f"（回覆療程={reply_treatments}｜合法價={sorted(allowed_prices)}）→ 讓 AI 重寫一次")
        rewritten = _rewrite_price_reply(
            final_ai_message_content,
            (mentioned_treatments or reply_treatments),
            guard_fees,
            leaked,
        )
        leaked2 = (_extract_prices(rewritten) & all_table_prices) - allowed_prices
        if "[[HANDOFF]]" in rewritten or leaked2:
            print(f"[price guard] 第二次仍有問題（leaked={sorted(leaked2)}）→ 轉真人")
            price_guard_handoff = True   # 維持原文，call_cs=1 時上層會清空 text
        else:
            print("[price guard] 重寫後價格正確 → 採用新回覆")
            final_ai_message_content = rewritten
        price_guard_trace = {
            "leaked": sorted(leaked),
            "allowed": sorted(allowed_prices),
            "reply_treatments": reply_treatments,
            "retried": True,
            "resolved": (not price_guard_handoff),
            "handoff": price_guard_handoff,
        }

    # 提取 AI 回覆中的圖片 URL
    extracted_urls = extract_image_urls(final_ai_message_content)
    
    # 取得使用者最新的問題（直接用 top-level content）
    user_query = user_input.content or ""
    # CallCS 三種值：0 = 正常 / 1 = 客人主動找真人客服（清空 text）/ 2 = 預約流程（保留 text）
    call_cs = 0

    # 療程幻覺改寫失敗 → 直接轉真人客服（最保守）
    if sanitize_force_handoff:
        call_cs = 1
    # 價格守門重寫兩次仍錯 → 轉真人
    if price_guard_handoff:
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

    # 依後端契約，message_history 裡「真人客服」講的話都帶 "[真人客服] " 前綴（見 main_webhook 說明），
    # 那是給 AI 辨識講者用的內部標記。當一通對話真人介入很多次時，AI 會「模仿」這個格式、
    # 把前綴一起寫進自己的回覆開頭 → 客人就會看到「[真人客服] 您好…」。
    # 只在「回覆真的帶了前綴」時才移除（正常回覆完全不動），並印 log 以便統計實際發生頻率。
    # 只處理開頭（可能連續多個），不動內文，避免誤傷「轉接真人客服」這類正常語句。
    if clean_text.lstrip().startswith("[真人客服]"):
        print(
            f"⚠️ AI 回覆誤帶 [真人客服] 前綴，已移除 "
            f"fb_account={user_input.fb_account}"
        )
        clean_text = re.sub(r"^\s*(?:\[真人客服\]\s*)+", "", clean_text)

    # CallCS=1（客人主動找真人）→ 清空 text/images，純通知後端轉接
    # CallCS=2（預約流程）→ 保留 AI 的預約引導文字、清空 images，後端先發文字再通知客服
    # CallCS=0 → 正常回覆
    if call_cs == 1:
        clean_text = ""
        all_images = []
    elif call_cs == 2:
        all_images = []

    # ─── 組裝 trace（LLM-as-judge 評估用）───
    # graph 內各節點已寫入 route / guard / draft / grounding / final / moderator；
    # 這裡補上只有 backend 才知道的兩塊：OCR 增補後的 user_input、以及最終 handoff_reason。
    # fb_account / timestamp / call_cs 不放 trace（後端存 DB 時本來就有，避免重複）。
    trace = dict(response.get("trace") or {})
    trace["user_input"] = final_text   # 含 OCR 圖片文字的完整輸入（後端手上只有原始 content）
    if price_guard_trace is not None:
        trace["price_guard"] = price_guard_trace   # 價格守門：是否偵測到跨療程殘留價、重寫是否救回
    if any(kw.lower() in user_query_lower for kw in CS_KEYWORDS):
        trace["handoff_reason"] = "customer_keyword"   # 客人主動要真人（最優先，對應 call_cs=1）
    elif sanitize_force_handoff:
        trace["handoff_reason"] = "fact_check"          # AI 幻覺被 moderator 攔下
    elif price_guard_handoff:
        trace["handoff_reason"] = "price_fabrication"   # 價格守門重寫兩次仍錯 → 轉真人
    elif call_cs == 2:
        trace["handoff_reason"] = "booking"             # 預約流程觸發
    else:
        trace["handoff_reason"] = None

    response_body = BackendResponse(
        text=clean_text,
        images=all_images,
        CallCS=call_cs,
        trace=trace,
    )

    # 📤 log 已移到外層 execute_backend_agent —— 那裡才拿得到 usage / elapsed_ms，
    # 而且四條 return 路徑都會印到（以前只有這條正常回覆會印）。
    return response_body


def _flatten_strings(value) -> List[str]:
    """把 grounding 攤平成字串清單。

    形狀不固定：moderator 寫入時可能是 `sorted(grounded)` 的扁平字串清單，
    也可能是 `[grounded]` 這種把整個 list 再包一層的巢狀結構（見 agent.py:1382）。
    遞迴攤平、只收字串，避免統計時因為形狀不同而少算。
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for v in value:
            out.extend(_flatten_strings(v))
        return out
    return []


def execute_backend_agent(user_input: BackendUserQuery) -> BackendResponse:
    """
    對外入口：跑 agent，並在 trace 補上「成本與效能」三組觀測欄位。

    為什麼包一層、而不是寫在 _execute_backend_agent_inner 裡面：
    那支函式有四個 return 出口（純圖片短路、graph 崩潰回罐頭、崩潰後轉真人、正常回覆），
    在外層量一次才能保證每條路徑都涵蓋到。尤其「純圖片短路」那條也跑了 OCR、
    是有 token 成本的，漏掉它成本統計就會系統性偏低。

    這三個欄位是**成本與效能**維度，跟既有的品質維度（handoff_reason / moderator.fact_check /
    draft vs final）互補；真正有用的分析多半在兩者交叉的地方。
    """
    t0 = perf_counter()
    with get_openai_callback() as usage:
        response_body = _execute_backend_agent_inner(user_input)

    trace = dict(response_body.trace or {})

    # ① 成本：input / output 分開記。本專案的成本幾乎全在 input
    #    （system prompt + 最近 GEN_HISTORY_MSGS=30 則歷史 + grounding 原文），
    #    output 只有幾百字 —— 要省錢得從「餵進去多少」下手，所以兩者要分開才有意義。
    trace["usage"] = {
        "total_tokens": usage.total_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        # 一輪打了幾次 LLM。booking 的 tool 迴圈會把這個數字墊高，
        # 異常大的值 = 那通在 react agent 裡繞圈（對照 recursion_limit=12）。
        "llm_calls": usage.successful_requests,
        # 依 langchain 內建價目表估算；表上沒收錄的新模型會算成 0，別當帳單用。
        "cost_usd": round(usage.total_cost, 6),
    }

    # ② 效能：伺服器端端到端耗時。
    #    log 裡的 ⏱️ 在 Docker、trace 在後端 DB，兩邊沒有共同 key 可以 join；
    #    要跟 route / usage 做交叉分析就得放進 trace。
    trace["elapsed_ms"] = round((perf_counter() - t0) * 1000)

    # ③ 原因變數：這輪塞了多少檢索原文進 prompt。
    #    ① 和 ② 是「結果」，這個是用來分辨「貴是因為對話歷史長，還是檢索撈太多」——
    #    兩者要動的旋鈕不同（agent.py 的 GEN_HISTORY_MSGS vs retriever 的 k）。
    chunks = _flatten_strings(trace.get("grounding"))
    trace["grounding_chunks"] = len(chunks)
    trace["grounding_chars"] = sum(len(c) for c in chunks)

    response_body.trace = trace

    # 印出「真正回傳給後端的完整內容」（含 trace 全部）供 Portainer log 對照。
    # 摘要放在第一行，這樣掃 log 時不用展開整包 JSON 就看得到成本與耗時。
    import json
    from datetime import datetime
    print(
        f"📤 [{datetime.now().isoformat()}] Response to Backend "
        f"fb_account={user_input.fb_account} "
        f"tokens={usage.total_tokens} calls={usage.successful_requests} "
        f"elapsed_ms={trace['elapsed_ms']} grounding_chars={trace['grounding_chars']}\n"
        + json.dumps(response_body.model_dump(), ensure_ascii=False, indent=2)
    )

    return response_body

    