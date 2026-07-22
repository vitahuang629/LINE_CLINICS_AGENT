# -*- coding: utf-8 -*-
"""
測試共用核心：組 request → 呼叫 AI → 比對「答對的特徵」。
replay.py（建題庫）與 test_regression.py（跑回歸）都 import 這裡，
保證兩邊用同一套組裝與判定邏輯。
"""
import os
import sys

# 讓 tests/ 底下的檔案 import 得到專案根目錄的 app / agent
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.models import BackendUserQuery, HistoryMessage, TreatmentFee, BackendResponse
from app.backend_agent_service import execute_backend_agent
from tests.fees import FEES

# ── 簡體字偵測（確定性、免錢）：純簡體字集合，命中即代表回覆夾了簡體 ──
_SIMPLIFIED_CHARS = set(
    "们这那会无时间个为请问变报价疗门单双脑检测对说护纤优严关离见认达谈觉让帮现"
    "书买卖东车轮软标题类页预约丽经历显员开闭构营养线应纹脸颜龄医术妆华语汉图团"
    "网际实业务运动机说话让们对时钟决达认谈显宝贝题总样"
)


def find_simplified(text: str) -> list:
    """回傳文字裡出現的簡體字（去重、保序）；空 list = 全繁體。"""
    seen, out = set(), []
    for ch in (text or ""):
        if ch in _SIMPLIFIED_CHARS and ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


def fees_text() -> str:
    """把費用表轉成給評審看的文字。"""
    return "\n".join(f"- {f['name']}：NT$ {f['price']:,}" for f in FEES)


_JUDGE_SCHEMA = {
    "title": "Verdict",
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong", "na"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

_JUDGE_SYSTEM = (
    "你是醫美客服回覆的評審。評分的**核心是：AI 有沒有正確回應『客人這一句實際問的問題』**。\n"
    "『真人參考答案』只是參考，**不是必須逐項複述的清單**——AI 不需要複述參考答案裡"
    "客人這句沒問到的額外資訊。『費用表』是價格的事實依據。只輸出結構化結果。\n\n"
    "重要原則：\n"
    "- **客人沒問的，AI 沒講，不扣分**（例：客人在回答年齡/部位、或只是要療程介紹時，AI 沒報價 → 不扣分）。\n"
    "- 以『客人的核心問題有沒有被正確回答』為主；次要的語氣/贅字瑕疵不足以判 wrong。\n"
    "- 價格類：只要 AI 報的價與費用表一致，就算價格正確；沒報客人沒問的其他療程價不扣分。\n\n"
    "判定標準：\n"
    "- correct：正確回應了客人這句實際問的問題；有講到的可查核事實（價/地址/次數）正確；無捏造價、無療效誇大。\n"
    "- partial：客人一句問了多件事，AI 只答到一部分；或核心答對但漏掉客人確實有問的某一項。\n"
    "- wrong：關鍵事實答錯（報錯價/錯地址）、捏造價格數字、療效誇大（保證一定瘦/治癒）、"
    "或完全沒回應客人問的核心問題（答非所問）。\n"
    "- na：無法公平評分——真人用了 AI 拿不到的資訊（查會員/查預約系統/「幫您查看」），"
    "或客人這句只是純寒暄/開場、無可查核的對錯。\n"
)


def judge_answer(question: str, reference: str, ai_answer: str) -> dict:
    """reference-based LLM 評審，回傳 {'verdict': ..., 'reason': ...}。"""
    from langchain_openai import ChatOpenAI

    user = (
        f"【客人問題】\n{question}\n\n"
        f"【真人參考答案】\n{reference or '（此題真人沒有回覆／對話結束，請僅依費用表與常識判斷 AI 回覆是否合理）'}\n\n"
        f"【診所費用表】\n{fees_text()}\n\n"
        f"【AI 的回覆】\n{ai_answer or '（空）'}"
    )
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(_JUDGE_SCHEMA)
        out = llm.invoke([
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ])
        return {"verdict": out.get("verdict", "na"), "reason": out.get("reason", "")}
    except Exception as e:
        return {"verdict": "na", "reason": f"評審失敗：{e}"}


def load_fees():
    return [TreatmentFee(**f) for f in FEES]


def build_query(content: str, history_chrono=None, ad_referral=None) -> BackendUserQuery:
    """
    組一個 request。

    history_chrono: 由「舊到新」的對話（自然閱讀順序），元素為
                    {"type": "human"|"ai", "content": "..."}。
                    正式後端是「由新到舊」，這裡自動幫你反轉，題庫用自然順序寫即可。
    ad_referral:    廣告來源關鍵字（僅在首次對話、history 為空時對 AI 生效）。
    """
    history_chrono = history_chrono or []
    # 反轉成後端契約要的「由新到舊」
    msg_history = [HistoryMessage(**m) for m in reversed(history_chrono)]
    return BackendUserQuery(
        fb_account="regression_test",
        content=content,
        image_url=[],
        message_history=msg_history,
        ad_referral=ad_referral,
        treatment_fees=load_fees(),
    )


def run_agent(query: BackendUserQuery) -> BackendResponse:
    return execute_backend_agent(query)


def _price_in(text: str, price: int) -> bool:
    """價格是否出現（同時比對 5999 與 5,999 兩種寫法）。"""
    return str(price) in (text or "") or f"{price:,}" in (text or "")


def check_expectations(resp: BackendResponse, expect: dict) -> list:
    """
    只驗 expect 裡「有寫」的欄位，回傳一串失敗訊息（空 list = 全過）。

    支援欄位：
      call_cs            int      —— CallCS 必須等於
      handoff_reason     str|null —— trace.handoff_reason 必須等於
      prices_present     [int]    —— 這些價格都要出現（自動比對有無逗號）
      prices_absent      [int]    —— 這些價格都不可出現
      text_contains      [str]    —— 這些字串「都」要出現
      text_contains_any  [str]    —— 這些字串「至少一個」要出現（同義詞用，如 單次/一次）
      text_not_contains  [str]    —— 這些字串都不可出現
      images_include_any [str]    —— images 至少要有一張「包含」清單中任一子字串
      images_empty       bool     —— images 必須為空
    """
    fails = []
    text = resp.text or ""
    imgs = resp.images or []
    trace = resp.trace or {}

    if "call_cs" in expect and resp.CallCS != expect["call_cs"]:
        fails.append(f"call_cs 期望 {expect['call_cs']}，實際 {resp.CallCS}")

    if "handoff_reason" in expect:
        hr = trace.get("handoff_reason")
        if hr != expect["handoff_reason"]:
            fails.append(f"handoff_reason 期望 {expect['handoff_reason']!r}，實際 {hr!r}")

    for p in expect.get("prices_present", []):
        if not _price_in(text, p):
            fails.append(f"應出現價格 {p:,} 但沒有")

    for p in expect.get("prices_absent", []):
        if _price_in(text, p):
            fails.append(f"不應出現價格 {p:,} 卻出現了")

    for s in expect.get("text_contains", []):
        if s not in text:
            fails.append(f"文字應包含 {s!r}")

    any_list = expect.get("text_contains_any")
    if any_list and not any(s in text for s in any_list):
        fails.append(f"文字應至少包含其一 {any_list}")

    for s in expect.get("text_not_contains", []):
        if s in text:
            fails.append(f"文字不應包含 {s!r}")

    inc = expect.get("images_include_any")
    if inc and not any(sub in img for sub in inc for img in imgs):
        fails.append(f"圖片應包含任一 {inc}，實際 {imgs}")

    if expect.get("images_empty") and imgs:
        fails.append(f"圖片應為空，實際 {imgs}")

    return fails
