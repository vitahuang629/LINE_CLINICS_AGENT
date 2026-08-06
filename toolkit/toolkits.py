from typing import Literal, Dict, Any
from contextvars import ContextVar
from langchain_core.tools import tool
from data_models.models import DateModel, DateTimeModel, IdentificationNumberModel
import os
import csv
import pandas as pd
from utils.ensemble_retriever import get_ensemble_retriever
from utils.qa_retriever import get_qa_retriever
from utils.consult_plan import get_consult_info
import random

print("toolkits.py is running")
# 兩份獨立 QA 索引（共用 builder、各自快取，避免交叉污染）：
#   clinic_qa     → 診所交易/層級問答（付款、預約流程、保險、初診…），由 booking_node 的 search_clinics_info 使用
#   treatment_qa  → 療程內容問答（效果、修復期、會不會痛…，category=療程名），由 information_node 使用
#
# ⚠️ 分店交通/停車（台北交通、竹北停車…）已搬到 CLINIC_BRANCH_CSV，**刻意不建進任何索引**：
#    兩間分店的答案長得幾乎一樣，語意檢索本來就容易撈錯筆，一律走 CLINIC_INFO_ROWS 查表。
TREATMENT_QA_CSV = "data/treatment_qa.csv"
CLINIC_QA_CSV = "data/clinics_qa.csv"
CLINIC_BRANCH_CSV = "data/clinic_branch_info.csv"
clinic_qa_retriever = get_qa_retriever(
    CLINIC_QA_CSV, "./chroma_clinic_qa", "./bm25_clinic_qa.pkl")
treatment_qa_retriever = get_qa_retriever(
    TREATMENT_QA_CSV, "./chroma_treatment_qa", "./bm25_treatment_qa.pkl", k=6)
ensemble_retriever = get_ensemble_retriever()

# treatment_qa 實際存在的 category 值 —— planner 的 qa_treatment 唯一合法值域。
# 直接由 CSV 產生：新增療程只要在 CSV 加列，planner 的可選清單就跟著長出來，
# 不必再為了「客人可能怎麼稱呼它」去手動補別名（那是列舉不完的）。
#
# ⚠️ 與 TREATMENT_SYNONYMS 是兩件事，別混用：
#    TREATMENT_SYNONYMS → 「這是我方還是外院療程」的廣義白名單，含沒有 QA 資料的療程
#                          （無限電波、EMBODY、FemiLift、G動椅…），也供 sanitize 展開別名。
#    這份              → 「QA 檢索查得到答案」的療程，拿前者當 qa_treatment 值域會讓
#                          planner 吐出必然撈空的療程名。
TREATMENT_QA_CATEGORIES: list = sorted(
    {
        str(c).strip()
        for c in pd.read_csv(TREATMENT_QA_CSV, encoding="utf-8-sig")["category"].dropna()
        if str(c).strip()
    }
)

# 每個 request 範圍的「合法療程」集合
# search_clinics_by_keyword 被呼叫時會把 retriever 回的療程名加進來
# 由 backend_agent_service 在 request 開始時 reset、結束時讀取
authorized_treatments_var: ContextVar = ContextVar("authorized_treatments", default=None)

# 每個 request 範圍的「事實來源」集合：本輪所有 retriever 撈到的原始內容（療程介紹、療程問答…）。
# sanitize 用它做 faithfulness 核對——AI 回覆裡關於療程的事實（名稱、英文全名、技術原理、
# 數據、機器/品牌名）必須能在這些內容裡找到依據，找不到的就視為編造、要依資料重講。
# 由 backend_agent_service 在 request 開始時 reset。
grounded_content_var: ContextVar = ContextVar("grounded_content", default=None)

# 每個 request 範圍的「療程費用表」：後端把整張表原封不動存進來（不預先 filter），
# 由 get_treatment_fee 工具在 booking 認出療程的當下才 filter——用 booking 的可靠解析，
# 取代原本 graph 前那個弱解析的預注入（會對代名詞/序數失敗、導致 AI 沒價可用而編價）。
treatment_fees_var: ContextVar = ContextVar("treatment_fees", default=None)


def register_grounded_content(text: str) -> None:
    """
    把本輪 retriever 撈到的原始內容登錄為「事實來源」，供 sanitize 的 faithfulness 核對。

    以「有序、去重的 chunk list」保存：傳入文字先按 retriever 慣用的 "\n\n---\n\n"
    分隔切成 chunk，逐塊登錄，只有「首次出現」的 chunk 會留下（chunk 級去重）。
    這讓 moderator 事實核對能對「單一小 chunk」逐字/語意比對，又快又準；
    也自然消掉「同一段原文經不同管道以不同包裝重複塞入」的重疊
    （set 只能去掉整段相同，抓不到這種近似重複）。

    ⚠️ 與 authorized_treatments_var 同樣的理由：必須**原地 mutate 同一個 list 物件**，
    不可換新物件再 set 回去——node 可能跑在 copy_context 裡，在 copy 內重建物件不會傳回
    父 context，但原地 .append() 因共用同一參照，父 context（sanitize）讀得到。
    """
    if not text or not str(text).strip():
        return
    cur = grounded_content_var.get()
    if cur is None:
        cur = []
    for piece in str(text).split("\n\n---\n\n"):
        piece = piece.strip()
        if piece and piece not in cur:
            cur.append(piece)
    grounded_content_var.set(cur)

# ── 療程清單單一真相來源：data/clinics_introductions3.csv ────────────────────
# name / keywords / category 三欄同時產生「別名表」與「Composer 白名單」。
# 以前這兩份分開手維護、靠人工同步 → 已經漂掉過（CSV 早有「野馬波」「瘦瘦針」，
# 別名表卻還是舊寫法，害療程對不上、QA 檢索整段被跳過）。改由 CSV 產生，新增療程只要加一列。
TREATMENT_INTRO_CSV = "data/clinics_introductions3.csv"


def _load_treatment_rows() -> list:
    """讀療程介紹 CSV（big5；Excel 另存時別存成 utf-8，程式會讀不到）。"""
    try:
        with open(TREATMENT_INTRO_CSV, encoding="big5", newline="") as f:
            rows = [{k: (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
        return [r for r in rows if r.get("name")]
    except Exception as e:
        print(f"[treatment] 載入療程清單失敗：{e}")
        return []


TREATMENT_ROWS: list = _load_treatment_rows()


def _family(name: str) -> str:
    """『冷凍減脂 + MitoX粒線體』歸到『冷凍減脂』底下，變體不另立門戶。"""
    for r in TREATMENT_ROWS:
        if r["name"] != name and name.startswith(r["name"] + " +"):
            return r["name"]
    return name


def _kw(text: str) -> list:
    return [w.strip() for w in str(text or "").replace("，", ",").split(",") if w.strip()]


# 跨療程共用的 keyword（如「熱減脂」同時掛 NEO 和 EMBODY）不能當別名——無法唯一決定療程。
# 它們仍留在 CSV 供 ensemble_retriever 檢索使用（那邊多撈幾個療程本來就是要推薦），
# 所以 CSV 裡撞詞不必手動清理，這裡會自動略過。
_owners: dict = {}
for _r in TREATMENT_ROWS:
    for _w in _kw(_r.get("keywords")):
        _owners.setdefault(_w.lower(), set()).add(_family(_r["name"]))

_groups: dict = {}
for _r in TREATMENT_ROWS:
    _g = _groups.setdefault(_family(_r["name"]), set())
    _g.add(_r["name"])                                    # 正式名一律是合法別名
    _g.update(w for w in _kw(_r.get("keywords")) if len(_owners[w.lower()]) == 1)

TREATMENT_SYNONYMS: list = list(_groups.values())

# Composer prompt 的「本診所只提供這些療程」白名單：依 category 分行、每個療程附上別名
_lines: list = []
for _cat in dict.fromkeys(r.get("category") or "其他" for r in TREATMENT_ROWS):
    _parts = []
    # 變體（冷凍減脂 + MitoX粒線體）不另列一行，會由基底那筆的別名帶出來
    for _n in [r["name"] for r in TREATMENT_ROWS
               if (r.get("category") or "其他") == _cat and _family(r["name"]) == r["name"]]:
        _alias = sorted(a for a in _groups[_family(_n)] if a != _n)
        _parts.append(f"{_n}（{'／'.join(_alias)}）" if _alias else _n)
    _lines.append(f"【{_cat}】" + "、".join(_parts))
TREATMENT_WHITELIST_LINES: str = "\n            ".join(_lines)

print(f"[treatment] 療程 {len(TREATMENT_ROWS)} 筆、別名群組 {len(TREATMENT_SYNONYMS)} 組")


@tool
def get_treatment_fee(treatment_name: str) -> str:
    """查詢某療程的「體驗價」（療程單次 / 組合方案的體驗價格）。

    參數:
    - treatment_name: 療程名稱（例如: Emface, NEO, 冷凍減脂, 瘦瘦筆, SIS…）。
      **必須帶明確療程名**，不可傳空字串、「費用」、「體驗價」這類非療程名。

    回傳該療程在費用表裡的所有體驗價方案（單做 + 組合都會列出）；查無方案時明確告知。
    ⚠️ 只查「體驗價」；初診 / 諮詢費請改用 search_clinics_info(treatment_name, "初診")。
    """
    fees = treatment_fees_var.get() or []
    name = (treatment_name or "").strip()
    if not name or name in ("費用", "體驗價", "價格", "初診"):
        return "（未指定明確療程，無法查體驗價。請先確認客人問的是哪個療程，再帶療程名查詢。）"

    # 用別名群組擴充比對詞（Emface↔菲斯波、NEO↔熱磁減脂…），對 fee name 做子字串比對
    keywords = {name}
    low = name.lower()
    for group in TREATMENT_SYNONYMS:
        if any(a.lower() in low or low in a.lower() for a in group):
            keywords |= set(group)
            break

    matched = [
        f for f in fees
        if any(k.lower() in (getattr(f, "name", "") or "").lower() for k in keywords)
    ]
    if not matched:
        return (
            f"[體驗價] {name} 目前在費用表裡沒有對應的體驗價方案。\n"
            f"請回覆客人「目前 {name} 沒有提供體驗價方案，可以先預約諮詢評估，由專人為您安排」，"
            f"**嚴禁**自己編造價格，也**嚴禁**拿其他療程的價格套用。"
        )

    lines = [
        f"- {getattr(f, 'name', '')}：NT$ {getattr(f, 'price', 0):,}"
        + ("（組合方案，整套價）" if "+" in (getattr(f, "name", "") or "") else "（單做價）")
        for f in matched
    ]
    has_combo = any("+" in (getattr(f, "name", "") or "") for f in matched)
    combo_guard = (
        "\n🚨 組合方案：標「組合方案」的那幾行是**整套一起做**的總價，"
        "嚴禁拆成單一療程 / 單一部位的價格，只能照完整方案名原樣報價。"
    ) if has_combo else ""

    content = "[體驗價] 以下是該療程的體驗價方案：\n" + "\n".join(lines) + combo_guard
    # 體驗價來自費用表（確定性來源）→ 登錄成 grounding，讓 moderator 事實核對認得它是依據
    register_grounded_content(content)
    print(f"[get_treatment_fee] {name!r} → 命中 {len(matched)} 筆體驗價")
    return content


# 診所基本資訊（地址 / 停車 / 看診時間 / 電話）已改為「分店別」多筆存在 clinics_qa，
# 由 search_clinics_info 依 category（如「台北停車」「竹北地址」）檢索對應分店那筆，
# 不再走單店短路。原 _load_clinic_basic_info / CLINIC_BASIC_INFO / CLINIC_INFO_INTENT 已移除。

# 分店靜態資訊查表：category（「台北交通」「竹北交通」「台北停車」「竹北停車」…）→ 乾淨答案文字。
# 供 booking 對「地址 / 停車 / 看診時間 / 電話」原文直出，讓地址文字 100% 來自 CSV、不經 LLM 生成，
# 徹底杜絕地址幻覺（模組載入時抓一次，直讀 CSV 不走模糊檢索，確保撈到正確分店那筆）。
#
# 吃兩份 CSV：交通/停車那 4 筆已搬到 clinic_branch_info.csv（搬出 RAG），其餘診所 FAQ 仍在 clinics_qa.csv。
# 兩份欄位格式與編碼相同，合併成同一張表，key（category）維持原樣 → 呼叫端不用改。
def _load_clinic_info_rows(*csv_paths: str) -> dict:
    table = {}
    for path in csv_paths:
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    cat = (row.get("category") or "").strip()
                    ans = (row.get("answer") or "").strip()
                    if cat and ans and cat not in table:
                        table[cat] = ans
        except Exception as e:
            print(f"[clinic_info] 載入 {path} 失敗：{e}")
    return table


CLINIC_INFO_ROWS: dict = _load_clinic_info_rows(CLINIC_QA_CSV, CLINIC_BRANCH_CSV)
print(f"[clinic_info] 分店靜態資訊查表已載入 {len(CLINIC_INFO_ROWS)} 筆 category")

# 啟動自檢：這 4 筆是「原文直出」唯一的資料來源。少了不會拋錯，只會安靜地掉回 react agent
# 去檢索（撈到付款/初診那類最接近的其他筆硬答），連停車圖也會一起消失 —— 所以在載入時就喊出來。
_REQUIRED_BRANCH_CATS = [f"{b}{t}" for b in ("台北", "竹北") for t in ("交通", "停車")]
_missing_cats = [k for k in _REQUIRED_BRANCH_CATS if k not in CLINIC_INFO_ROWS]
if _missing_cats:
    print(f"[clinic_info] ⚠️ 查表缺少必要 category：{_missing_cats} → 請檢查 {CLINIC_BRANCH_CSV}")


# ── 分店主題 category 的收斂：react agent 傳來的 category 五花八門（「台北停車」「竹北地址」
#    「台北看診時間」「信義店電話」…），統一對到 CLINIC_INFO_ROWS 的 4 個 key。
#    對照表刻意跟 agent.py 的 _CLINIC_PARKING_KW / _CLINIC_LOCATION_KW 分開維護：
#    那邊判斷的是「客人原話」，這邊判斷的是「LLM 填的參數」，兩者詞彙分佈不同。
_BRANCH_ALIAS = {"台北": "台北", "臺北": "台北", "信義": "台北", "竹北": "竹北", "新竹": "竹北"}
_PARKING_WORDS = ("停車", "車位", "開車", "好停")
_TRAFFIC_WORDS = ("交通", "地址", "位置", "地點", "門牌", "在哪", "哪裡", "怎麼",
                  "看診", "門診", "營業", "時間", "電話",
                  # 客人問路的講法（關鍵字表列不完，這裡只求常見的先接住）
                  "捷運", "公車", "高鐵", "火車", "客運", "出口", "幾站", "路線",
                  "導航", "地圖", "走路", "步行", "搭車", "坐車", "過去", "到達")

# ⚠️ 與 agent.py 的 _CLINIC_BRANCH_ASK 組出來的反問句必須一字不差
#    （agent.py 靠這個句子的指紋辨識「客人正在回答哪一間」），改一邊要改兩邊。
BRANCH_ASK_REPLY = "我們有台北信義店和竹北店，請問您想了解哪一間呢？😊"


def lookup_branch_info(category: str):
    """category（「台北停車」「竹北地址」…）→ 分店靜態資訊原文。

    - 分店 + 主題都對得到 → 回原文
    - 有主題但沒帶分店（「停車」「地址」）→ 回固定反問句，不讓它去檢索亂撈
    - 不是分店主題（付款、初診、保險…）→ 回 None，呼叫端照原本流程走
    """
    cat = (category or "").strip()
    if not cat:
        return None
    branch = next((v for k, v in _BRANCH_ALIAS.items() if k in cat), "")
    if any(w in cat for w in _PARKING_WORDS):
        topic = "停車"
    elif any(w in cat for w in _TRAFFIC_WORDS):
        topic = "交通"
    else:
        return None
    if not branch:
        print(f"[clinic_info] category={cat!r} 有主題但沒分店 → 反問客人")
        return BRANCH_ASK_REPLY
    return CLINIC_INFO_ROWS.get(f"{branch}{topic}")


# 療程介紹「原文直出」：療程名 → clinics_introductions3.csv 的 introduction 原文（big5）。
# 供 information_node 對「單一指名療程的介紹請求」一字不改直出，杜絕 AI 自編英文全名/縮寫/原理。
TREATMENT_INTRO_ROWS: dict = {
    r["name"]: r["introduction"] for r in TREATMENT_ROWS if r.get("introduction")
}
print(f"[treatment_intro] 療程介紹原文已載入 {len(TREATMENT_INTRO_ROWS)} 筆")


def _match_intro_name(name: str) -> str:
    """把客人／planner 講的名字對到 CSV 的正式療程名；對不上回 ""。"""
    low = name.lower()
    for k in TREATMENT_INTRO_ROWS:                        # 1) 直接命中（不分大小寫）
        if k.lower() == low:
            return k
    for group in TREATMENT_SYNONYMS:                      # 2) 別名群組（Emface↔菲斯波、SIS↔Sis…）
        if any(a.lower() in low or low in a.lower() for a in group):
            for k in TREATMENT_INTRO_ROWS:
                kl = k.lower()
                if any(a.lower() in kl or kl in a.lower() for a in group):
                    return k
            break
    for k in TREATMENT_INTRO_ROWS:                        # 3) 寬鬆子字串
        if name in k or k in name:
            return k
    return ""


def get_treatment_intro(treatment_name: str):
    """回傳某療程的 CSV 原文介紹；找不到回 None。

    若該療程有「加強版」變體（冷凍減脂 → 冷凍減脂 + MitoX粒線體），結尾只附一句帶過，
    不把第二段介紹整個貼上——兩段都貼，一則 LINE 訊息會太長。
    """
    name = (treatment_name or "").strip()
    matched = _match_intro_name(name) if name else ""
    if not matched:
        return None
    intro = TREATMENT_INTRO_ROWS[matched]
    variants = [r["name"] for r in TREATMENT_ROWS
                if r["name"] != matched and _family(r["name"]) == matched]
    if variants:
        intro += f"\n\n另外我們也有「{'」「'.join(variants)}」的加強方案，需要的話可以幫您介紹 💕"
    return intro


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
    # ⚠️ 必須「原地 mutate」這個 set，不可換成新物件再 set 回去。
    #    backend_agent_service 在 request 開始時先 set 一個空 set，graph 的 node 可能跑在
    #    copy_context 裡——在 copy 裡做的 .set() 不會傳回父 context，但原地 .add() 因為共用
    #    同一個物件參照，父 context（sanitize_ai_response）讀得到。改成 `set(old | new)` 之類
    #    建新物件的寫法會斷鏈，導致幻覺檢查靜默失效。
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

    # 把療程介紹原始內容登錄為事實來源，供 sanitize faithfulness 核對（機器/品牌/技術名都在這裡）
    register_grounded_content(combined_results)

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
    查詢診所資訊（分店交通/停車、初診費、付款、健保保險、預約流程…），只回資料庫固定答案。
    參數:
    - treatment_name: 療程名稱 (例如: 瘦瘦筆, EMBODY, NEO, Emface, 冷凍減脂, 腦波機, SIS)；
      查分店交通/停車這類與療程無關的資訊時填「診所」。
    - category: 查詢類別
      · 分店交通 / 停車 → 必須寫成「<分店><主題>」，例：「台北停車」「竹北地址」「台北看診時間」。
        分店只有「台北」(信義店) 與「竹北」兩間；地址 / 看診時間 / 電話同屬「交通」那筆。
        ⚠️ 客人沒指明哪一間時**不要呼叫**，請先反問是台北還是竹北。
      · 其他 → 費用、初診、諮詢、付款、健保、保險、預約、分店（問共有幾間店用這個）。
    """
    print(f"Target Treatment: {treatment_name}, Category: {category}")

    # 初診 / 諮詢 → 走 consult_plan.csv 結構化查表（免費/收費是明確欄位，不用語意檢索）
    if "初診" in category or "諮詢" in category:
        consult = get_consult_info(treatment_name, TREATMENT_SYNONYMS)
        if consult:
            # 初診/諮詢內容來自 consult_plan 查表（確定性來源）→ 登錄成 grounding，
            # 否則 moderator 事實核對會因為在檢索內容裡找不到依據而把它當幻覺砍掉。
            register_grounded_content(consult)
            return consult
        print(f"[consult] 無對應，fallback 回 qa_retriever：{treatment_name}")

    # 分店交通 / 停車（category 帶「台北停車」「竹北地址」「信義店看診時間」…）→ 直接查表回原文，不進檢索。
    # 這 4 筆已搬出 clinics_qa、不在 clinic_qa 索引裡，若還走語意檢索只會撈到付款/初診那類
    # 「最接近的其他筆」，然後模型拿著不相干的內容硬答 —— 回錯的比回「查無」更糟。
    branch_answer = lookup_branch_info(category)
    if branch_answer:
        # 查表是確定性來源 → 登錄成 grounding，否則 moderator 事實核對會在檢索內容裡
        # 找不到依據，把正確的地址當幻覺砍掉。
        register_grounded_content(branch_answer)
        print(f"[clinic_info] search_clinics_info 查表直出：{category}")
        return branch_answer

    # 走到這裡 = 診所交易/層級問答（付款、預約流程、保險、初診…），用 category 主題詞檢索 clinic_qa。
    # 療程內容問答（效果/修復期/會不會痛）已改由 information_node 走 treatment_qa，不在此處理。
    boosted_query = f"{category} {category} {category}"
    print(f"Boosted Query: {boosted_query}")


    # --- Step 2. 呼叫 retriever ---
    docs = clinic_qa_retriever.get_relevant_documents(boosted_query)
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
    register_grounded_content(content)
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
            "empathy": """老化不是單純皮膚問題，底層的肌肉也在慢慢「無力」，才是臉型往下走的真正原因喔。
                        我們這邊特別的地方是從最底層的結構「肌肉層 + 筋膜層 + 真皮層」這三個層次同時處理，讓效果更全面更持久✨
                        歡迎與我們聊聊💬了解您的需求～""",
            "questions": ["過去有嘗試過相關的療程嗎？", "請問目前年齡大約在哪個區間呢？ 想加強或改善哪些臉部狀況呢？"]
        },
        "私密療程": {
            "empathy": "私密處的保養與健康確實非常重要，謝謝您願意信任並與我分享。 ",
            "questions": ["您主要是想了解功能改善，還是日常的美觀保養呢？"]
        },
        "睡眠與神經": {
            "empathy": '很多人睡不好、失眠？但其實背後常伴隨壓力、緊繃、焦慮甚至心悸的狀況，更常見的是不舒服很久，卻一直找不到原因！\n\n我們結合複合式非侵入治療與現代 AI 智能數據檢測，累積萬筆治療成功案例，讓治療有依據更安心💕',
            "questions": ['方便進一步了解，請問目前有出現以下這些狀況嗎？\n1. 睡不好、淺眠易醒、入睡困難\n2. 睡覺會打呼、有呼吸中止情況\n3. 長期依賴藥物，副作用明顯\n4. 情緒緊繃、緊張焦慮不安\n5. 心跳偏快、容易胸悶心悸\n6. 記憶力下降、注意力變差\n7. 頭痛、頭暈、耳鳴常發作\n8. 胃食道逆流、脹氣、消化不良\n9. 經常累沒精神、莫名身體痠痛\n\n（只需回數字即可，方便快速回覆唷）']
        },
        "體態管理": {
            "empathy": '有產生這些問題，往往不是單一因素造成的唷！\n\n我們不只幫您解決單一問題，而是從源頭出發進行整合性的調整，由內到外讓您美麗與健康同時兼具，幫助您維持更長久的良好狀態。',
            "questions": ['歡迎和我們聊聊，好讓我們了解一下您的狀況唷\n1.【體重嚴重卡關 / 瘦不下來】\n2.【嘗試減肥失敗 / 容易復胖】\n3.【作息睡眠品質 / 情緒壓力】\n4.【飲食難以控制／暴飲暴食】\n5.【長期久站久坐／缺乏運動】\n6.【產後無法維持／身材走樣】\n7.【代謝逐漸下降／年紀漸長】\n\n（只需回數字即可，方便快速回覆唷）']
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

