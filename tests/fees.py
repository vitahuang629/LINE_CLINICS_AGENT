# -*- coding: utf-8 -*-
"""
測試用費用表 —— 你只要改這裡的數字（對照官方最新體驗價）。
replay 與回歸測試每次呼叫 AI 都會把「整張表」一起送進去（跟正式後端一樣），
藉此順便測 AI 有沒有挑對療程、報對價。
"""

# 療程體驗價（name 貼近正式後端格式：體驗時間包在括號裡，如「SIS(15分鐘)」；price 為純數字）
# ⚠️ 時長就從 name 括號裡的「X分鐘」讀出來，故 name 一定要保留分鐘資訊。
FEES = [
    {"name": "SIS 科技深層精雕(15分鐘)",                     "price": 1888},
    {"name": "冷凍減脂(60分鐘-單點)",                        "price": 5999},
    {"name": "冷凍減脂(60分鐘-雙點-打卡贈紅光)",             "price": 8999},
    {"name": "NEO-熱磁減脂(30分鐘) + SIS(15分鐘)",           "price": 9999},
    {"name": "EMFACE 菲斯波(20分鐘-全臉多層次拉提)",         "price": 16800},
    {"name": "NEO-熱磁減脂(30分鐘) + 冷凍雙點(60分鐘)",      "price": 18999},
]

# 廣告主題 → ad_referral 關鍵字（首次對話注入給 AI 當開場提示）。
# 掃廣告文字，命中哪組 keyword 就用對應 referral（由上到下先命中先贏）。
AD_REFERRAL_RULES = [
    (["SIS", "深層精雕", "超磁能"],                     "SIS 科技深層精雕"),
    (["EMFACE", "菲斯波", "拉提", "法令紋", "雙效"],     "EMFACE 菲斯波"),
    (["熱磁", "NEO"],                                    "NEO 熱磁減脂"),
    (["冷凍", "下腹", "小腹", "Z LIPO", "塑酷", "5999"], "冷凍減脂"),
]


def resolve_ad_referral(ad_text: str):
    """從廣告文字推出 ad_referral 關鍵字；推不出回 None。"""
    t = (ad_text or "").upper()
    for keywords, referral in AD_REFERRAL_RULES:
        if any(kw.upper() in t for kw in keywords):
            return referral
    return None
