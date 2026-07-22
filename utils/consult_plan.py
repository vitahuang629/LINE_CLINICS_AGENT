"""
初診 / 諮詢費結構化查表。

consult_plan.csv 欄位：
  - treatment    療程名（通常含「初診」後綴，如「腦波機初診」）
  - consult_free 諮詢是否免費（是 / 否）
  - consult_fee  收費方案說明文字（免費療程此欄為空）
  - plans        初診流程細節 / 免費療程的完整說明

用「療程名」精確查表，而非語意檢索 —— 免費 / 收費是明確欄位，
AI 只需照抄結論，不會再自己腦補「免費」。

TREATMENT_SYNONYMS 由呼叫端（toolkit.toolkits）以參數傳入，
避免與 toolkits 互相 import 造成循環。
"""
import pandas as pd

_CSV_PATH = "data/consult_plan.csv"


def _synonym_set(name: str, synonyms: list) -> set:
    """回傳 name 所屬的同義詞群組（小寫）；找不到就回傳 {name 本身}。"""
    n = name.lower().strip()
    for group in synonyms:
        if any(n == g.lower() or n in g.lower() or g.lower() in n for g in group):
            return {g.lower() for g in group}
    return {n}


def _load_consult_plans() -> list:
    try:
        df = pd.read_csv(_CSV_PATH, encoding="utf-8-sig")
    except Exception as e:
        print(f"[consult] 載入 {_CSV_PATH} 失敗：{e}")
        return []
    records = []
    for _, row in df.iterrows():
        treatment = str(row["treatment"]).strip()
        base = treatment[:-2] if treatment.endswith("初診") else treatment
        free = str(row.get("consult_free", "")).strip() in ("是", "Y", "y", "True", "1")
        fee = "" if pd.isna(row.get("consult_fee")) else str(row["consult_fee"]).strip()
        plans = "" if pd.isna(row.get("plans")) else str(row["plans"]).strip()
        records.append({
            "treatment": treatment, "base": base,
            "free": free, "fee": fee, "plans": plans,
        })
    return records


# 模組載入時讀一次
CONSULT_PLANS = _load_consult_plans()
print(f"[consult] 載入 {len(CONSULT_PLANS)} 筆初診方案")


def _format_consult(rec: dict) -> str:
    """組成回給客人的固定文字（免費療程 fee 為空只回 plans，收費療程 fee+plans 都帶）。"""
    parts = [p for p in (rec["fee"], rec["plans"]) if p]
    return "\n\n".join(parts)


def get_consult_info(treatment_name: str, synonyms: list) -> str | None:
    """
    依療程名查 consult_plan.csv，回傳組好的初診/諮詢說明文字；查不到回 None。
    比對優先序：同義詞群組命中 > 子字串命中。
    """
    q = (treatment_name or "").lower().strip()
    if not q or not CONSULT_PLANS:
        return None

    q_syns = _synonym_set(q, synonyms)
    fallback = None
    for rec in CONSULT_PLANS:
        rec_syns = _synonym_set(rec["base"], synonyms)
        if q_syns & rec_syns:
            print(f"[consult] 同義詞命中：{treatment_name} → {rec['treatment']} (free={rec['free']})")
            return _format_consult(rec)
        base = rec["base"].lower()
        if fallback is None and (q in base or base in q):
            fallback = rec

    if fallback:
        print(f"[consult] 子字串命中：{treatment_name} → {fallback['treatment']} (free={fallback['free']})")
        return _format_consult(fallback)

    print(f"[consult] 查無對應療程：{treatment_name}")
    return None


