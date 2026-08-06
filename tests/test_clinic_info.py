# -*- coding: utf-8 -*-
"""分店靜態資訊（地址／停車／看診／電話）的離線回歸測試。

跟 test_regression.py 不同：**這裡完全不呼叫 OpenAI**，跑完只要幾秒、不花錢。
因為這條路徑刻意設計成「確定性查表」——判斷用關鍵字表、答案用 CSV 原文，
中間沒有 LLM，所以行為可以百分之百重現，適合用一般 pytest 鎖住。

守的是三件事：
  1. 該接管的問題有直出，而且撈到**正確分店**那筆（地址錯了比答不出來更糟）
  2. **不該接管的問題不要被劫走**——關鍵字表加太寬時，療程問題會被誤判成問地點
     （典型：「台北有腦波機療程嗎」「做完走路會不會痛」）
  3. 查表資料還在（CSV 被搬走時，直出會安靜地失效，連停車圖也一起消失）

跑法（專案根目錄）：
    pytest tests/test_clinic_info.py -v
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import clinic_info_direct_answer
from toolkit.toolkits import BRANCH_ASK_REPLY, CLINIC_INFO_ROWS, lookup_branch_info

# 反問句的指紋（agent.py 用它辨識「客人正在回答哪一間」）
_ASK = "請問您想了解哪一間"


def _h(text):
    """客人訊息（backend 實際傳進來的是 list-of-dict 格式，這裡照抄）。"""
    return HumanMessage(content=[{"type": "text", "text": text}])


def _a(text):
    return AIMessage(content=text)


def _label(reply):
    """把回覆歸成一個好比對的標籤：撈到哪一筆 / 反問 / 沒接管。

    用答案裡的專屬字串認人（春光公園只在台北停車、莊敬南路只在竹北交通…），
    比對到「哪一間」的層級，避免「有回答但回錯分店」被當成通過。
    """
    if reply is None:
        return "none"
    if _ASK in reply:
        return "ask"
    for marker, label in (
        ("春光公園", "台北停車"),
        ("P12", "竹北停車"),
        ("忠孝東路", "台北交通"),
        ("莊敬南路", "竹北交通"),
    ):
        if marker in reply:
            return label
    return f"未知內容：{reply[:40]!r}"


# (題目, 對話, 期望標籤)
CASES = [
    # ── 該接管：分店 + 主題都判斷得出 → 原文直出 ──
    ("台北問停車",        [_h("台北店怎麼停車？")],       "台北停車"),
    ("竹北問停車",        [_h("竹北好停車嗎")],           "竹北停車"),
    ("問車位",            [_h("台北那間有車位嗎")],       "台北停車"),
    ("竹北問地點",        [_h("竹北在哪裡")],             "竹北交通"),
    ("問捷運哪一站",      [_h("台北的捷運坐到哪一站呢")], "台北交通"),
    ("問交通方便嗎",      [_h("竹北那邊交通方便嗎")],     "竹北交通"),
    ("問幾號出口",        [_h("台北店是幾號出口")],       "台北交通"),

    # ── 有主題、沒指明分店 → 固定句反問，不要自己猜一間 ──
    ("沒指明分店問地點",  [_h("你們在哪裡？")],           "ask"),
    ("沒指明分店問停車",  [_h("你們有停車場嗎")],         "ask"),

    # ── 上下文：客人回答分店 / 沿用稍早講過的分店 ──
    ("反問後客人回台北",
     [_h("你們在哪裡？"), _a(f"我們有台北信義店和竹北店，{_ASK}呢？😊"), _h("台北")],
     "台北交通"),
    ("先問竹北地址再問停車",
     [_h("竹北在哪"), _a("(竹北地址原文)"), _h("那有停車嗎")],
     "竹北停車"),

    # ── 不該接管：這些要交回 react agent，被劫走就是 bug ──
    # 句中有分店名，但問的是療程 → 回地址是答非所問
    ("問某分店有沒有某療程", [_h("台北有腦波機療程嗎")],   "none"),
    # 「走路」若被當成問路關鍵字，這題就會被誤劫（所以關鍵字表只放「走過去」）
    ("問術後走路會不會痛",   [_h("做完走路會不會痛")],     "none"),
    ("問療程會不會痛",       [_h("Emface 會不會痛")],      "none"),
    # 分店總覽（有幾間店）不是單店地址，答案在 clinics_qa，走 react agent
    ("問共有幾間店",         [_h("你們有幾間店")],         "none"),
    # 客人只是嫌遠、沒有在問地點；此時該由 react agent 用兩間店的資訊安撫
    ("客人抱怨距離遠",
     [_h("你們在哪？"), _a("(竹北地址原文)"), _h("因為我在台南有點遠欸")],
     "none"),
]


@pytest.mark.parametrize("name,messages,expected", CASES, ids=[c[0] for c in CASES])
def test_clinic_info_direct_answer(name, messages, expected):
    got = _label(clinic_info_direct_answer(messages))
    assert got == expected, (
        f"\n題目：{name}"
        f"\n客人最後一句：{messages[-1].content}"
        f"\n期望：{expected}｜實際：{got}"
    )


# react agent 那條路：category 由 LLM 填，寫法五花八門，一律收斂到同 4 筆查表
LOOKUP_CASES = [
    ("台北停車",     "台北停車"),
    ("竹北停車",     "竹北停車"),
    ("台北地址",     "台北交通"),
    ("竹北怎麼去",   "竹北交通"),
    ("台北看診時間", "台北交通"),
    ("信義店電話",   "台北交通"),
    ("停車",         "ask"),      # 有主題沒分店 → 反問，不要讓它去檢索亂撈
    ("地址",         "ask"),
    ("付款",         "none"),     # 非分店主題 → 交回 clinic_qa 檢索
    ("初診",         "none"),
    ("分店",         "none"),
]


@pytest.mark.parametrize("category,expected", LOOKUP_CASES, ids=[c[0] for c in LOOKUP_CASES])
def test_lookup_branch_info(category, expected):
    got = _label(lookup_branch_info(category))
    assert got == expected, f"category={category!r} 期望 {expected}、實際 {got}"


@pytest.mark.parametrize("cat", ["台北交通", "竹北交通", "台北停車", "竹北停車"])
def test_required_categories_loaded(cat):
    """4 筆分店資料是直出唯一的來源；CSV 被搬走時這裡要先紅，而不是線上安靜地失效。"""
    assert cat in CLINIC_INFO_ROWS, (
        f"查表缺少 {cat} → 檢查 data/clinic_branch_info.csv 是否存在／欄位是否正確"
    )


@pytest.mark.parametrize("cat", ["台北停車", "竹北停車"])
def test_parking_answer_keeps_image_url(cat):
    """停車圖只靠答案裡這行文字被 backend 抓出來（extract_image_urls），刪掉圖就沒了。"""
    assert "圖片: https://" in CLINIC_INFO_ROWS[cat], f"{cat} 的答案裡沒有圖片 URL"


def test_branch_ask_reply_matches_agent_fingerprint():
    """toolkits 的反問句與 agent.py 用來辨識「客人在回答哪一間」的指紋必須一致。

    兩邊各寫一份（工具端／短路端），字串一旦漂掉，客人回「台北」就接不回原本的主題。
    """
    assert _ASK in BRANCH_ASK_REPLY
