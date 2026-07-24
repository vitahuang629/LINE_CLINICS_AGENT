# -*- coding: utf-8 -*-
"""
本地「多維度」評估 baseline —— 不需 LangSmith / 不需帳號。

和 test_regression.py 的差別：
  - test_regression：每題「所有 expect 全過才算過」，輸出 pass/fail（回歸守門用）。
  - eval_baseline  ：把每題拆成多個「維度」各自判 pass/fail/na，最後印出
                     「每個維度的準確率」成績單（診斷模型弱在哪一塊、當優化基準線用）。

沿用 tests/_harness.py 的現成零件（build_query / run_agent / find_simplified /
judge_answer / fees_text），不重造輪子。

用法（Windows 本機 poetry）：
    $env:PYTHONUTF8=1
    poetry run python tests/eval_baseline.py               # 只跑免費的確定性 + 合規維度
    poetry run python tests/eval_baseline.py --judge       # 額外跑「回覆品質」LLM 評審（會花 API 錢）
    poetry run python tests/eval_baseline.py --cases tests/cases.yaml
    poetry run python tests/eval_baseline.py --limit 3     # 只跑前 3 題（先小量試）

⚠️ 會真的呼叫 OpenAI（跑 agent 本身就要），和跑 app 同一個 .env / 向量庫環境。
"""
import argparse
import os
import sys

import yaml

# 讓 tests/ 底下 import 得到專案根目錄
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests._harness import (
    build_query,
    run_agent,
    find_simplified,
    judge_answer,
)

# 誇大／保證性字眼（合規維度）——對齊 moderator 想移除的詞，命中即視為違規。
# 這是啟發式清單，不是窮舉；要更嚴謹可改用 LLM judge 判合規。
_EXAGGERATION_WORDS = [
    "一定會好", "一定瘦", "保證", "立即見效", "馬上見效", "永久", "根治",
    "治癒", "完全消除", "百分之百", "100%有效", "絕對有效",
]

# 每題可能出現的維度（固定順序，成績單照這個排）
_DIMENSIONS = [
    "call_cs",
    "price",
    "text",
    "images",
    "handoff_reason",
    "繁體一致",
    "無誇大用語",
    "回覆品質",
]


def _price_in(text: str, price: int) -> bool:
    """價格是否出現（同時比對 5999 與 5,999 兩種寫法）。與 _harness 同邏輯。"""
    return str(price) in (text or "") or f"{price:,}" in (text or "")


def _score_deterministic(resp, expect: dict) -> dict:
    """
    逐維度判定確定性欄位，回傳 {維度: "pass"|"fail"|"na"}。
    "na" = 這題的 expect 沒有這個維度（不適用，不計入該維度分母）。
    """
    text = resp.text or ""
    imgs = resp.images or []
    trace = resp.trace or {}
    out = {}

    # call_cs
    if "call_cs" in expect:
        out["call_cs"] = "pass" if resp.CallCS == expect["call_cs"] else "fail"
    else:
        out["call_cs"] = "na"

    # handoff_reason
    if "handoff_reason" in expect:
        hr = trace.get("handoff_reason")
        out["handoff_reason"] = "pass" if hr == expect["handoff_reason"] else "fail"
    else:
        out["handoff_reason"] = "na"

    # price（prices_present 全要有 + prices_absent 全不可有）
    present = expect.get("prices_present", [])
    absent = expect.get("prices_absent", [])
    if present or absent:
        ok = all(_price_in(text, p) for p in present) and \
             all(not _price_in(text, p) for p in absent)
        out["price"] = "pass" if ok else "fail"
    else:
        out["price"] = "na"

    # text（contains 全要有 / contains_any 至少一 / not_contains 全不可有）
    contains = expect.get("text_contains", [])
    contains_any = expect.get("text_contains_any", [])
    not_contains = expect.get("text_not_contains", [])
    if contains or contains_any or not_contains:
        ok = all(s in text for s in contains) and \
             (not contains_any or any(s in text for s in contains_any)) and \
             all(s not in text for s in not_contains)
        out["text"] = "pass" if ok else "fail"
    else:
        out["text"] = "na"

    # images
    inc = expect.get("images_include_any")
    if inc is not None:
        out["images"] = "pass" if any(sub in img for sub in inc for img in imgs) else "fail"
    elif expect.get("images_empty"):
        out["images"] = "pass" if not imgs else "fail"
    else:
        out["images"] = "na"

    return out


def _score_compliance(resp) -> dict:
    """合規維度（免費、確定性）。回覆為空（如 CallCS=1）→ na。"""
    text = resp.text or ""
    if not text.strip():
        return {"繁體一致": "na", "無誇大用語": "na"}
    trad = "pass" if not find_simplified(text) else "fail"
    hits = [w for w in _EXAGGERATION_WORDS if w in text]
    exagg = "pass" if not hits else "fail"
    return {"繁體一致": trad, "無誇大用語": exagg}


def _score_quality(question: str, resp) -> dict:
    """回覆品質（reference-free LLM 評審）。回傳維度值 + 原始 verdict。"""
    text = resp.text or ""
    if not text.strip():
        return {"維度": "na", "verdict": "na", "reason": "（回覆為空）"}
    v = judge_answer(question=question, reference="", ai_answer=text)
    verdict = v.get("verdict", "na")
    # 成績單用：correct→pass、partial→partial、wrong→fail、na→na
    mapping = {"correct": "pass", "partial": "partial", "wrong": "fail", "na": "na"}
    return {"維度": mapping.get(verdict, "na"), "verdict": verdict, "reason": v.get("reason", "")}


def run_case(case: dict, use_judge: bool) -> dict:
    """跑一題，回傳該題各維度結果。"""
    inp = case.get("input", {})
    expect = case.get("expect", {})
    content = inp.get("content", "")

    resp = run_agent(build_query(
        content=content,
        history_chrono=inp.get("message_history"),
        ad_referral=inp.get("ad_referral"),
    ))

    dims = {}
    dims.update(_score_deterministic(resp, expect))
    dims.update(_score_compliance(resp))

    quality_detail = None
    if use_judge:
        q = _score_quality(content, resp)
        dims["回覆品質"] = q["維度"]
        quality_detail = q
    else:
        dims["回覆品質"] = "na"

    return {
        "name": case.get("name", "(未命名)"),
        "dims": dims,
        "resp": resp,
        "quality": quality_detail,
    }


def _accuracy(pass_n: float, applicable: int) -> str:
    return f"{pass_n / applicable * 100:5.1f}%" if applicable else "  n/a "


def print_scorecard(results: list, use_judge: bool):
    # 逐維度統計：applicable（非 na）、pass（partial 記 0.5）
    tally = {d: {"applicable": 0, "pass": 0.0, "correct": 0, "partial": 0, "wrong": 0}
             for d in _DIMENSIONS}
    for r in results:
        for d, v in r["dims"].items():
            if v == "na":
                continue
            t = tally[d]
            t["applicable"] += 1
            if v == "pass":
                t["pass"] += 1.0
                t["correct"] += 1
            elif v == "partial":
                t["pass"] += 0.5
                t["partial"] += 1
            else:
                t["wrong"] += 1

    print("\n" + "=" * 52)
    print("  多維度評估 Baseline")
    print("=" * 52)
    print(f"  題數：{len(results)}    LLM 評審：{'開' if use_judge else '關（加 --judge 開啟）'}")
    print("-" * 52)
    print(f"  {'維度':<12}{'適用':>5}{'通過':>7}{'準確率':>9}")
    print("-" * 52)
    for d in _DIMENSIONS:
        t = tally[d]
        if t["applicable"] == 0:
            print(f"  {d:<12}{'—':>5}{'—':>7}{'  n/a ':>9}")
        else:
            passed = f"{t['pass']:.1f}".rstrip("0").rstrip(".")
            print(f"  {d:<12}{t['applicable']:>5}{passed:>7}{_accuracy(t['pass'], t['applicable']):>9}")
    print("-" * 52)

    # 整體（確定性 + 合規；不含 LLM 品質，因它偏主觀）
    det_dims = ["call_cs", "price", "text", "images", "handoff_reason", "繁體一致", "無誇大用語"]
    full_pass = sum(
        1 for r in results
        if all(r["dims"][d] != "fail" for d in det_dims)
    )
    print(f"  整體（確定性+合規）：{full_pass}/{len(results)} 題全過")
    print("=" * 52)

    # 失敗 / partial 明細
    lines = []
    for r in results:
        for d in _DIMENSIONS:
            v = r["dims"].get(d, "na")
            if v in ("fail", "partial"):
                extra = ""
                if d == "回覆品質" and r["quality"]:
                    extra = f" — {r['quality']['reason']}"
                lines.append(f"  [{d}] {r['name']}: {v}{extra}")
    if lines:
        print("\n失敗 / partial 明細：")
        print("\n".join(lines))
    else:
        print("\n（無失敗項）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(_ROOT, "tests", "cases.yaml"))
    ap.add_argument("--judge", action="store_true", help="額外跑回覆品質 LLM 評審（花 API 錢）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 題（0=全部）")
    args = ap.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        cases = yaml.safe_load(f) or []
    if args.limit:
        cases = cases[: args.limit]

    results = []
    for i, case in enumerate(cases, 1):
        name = case.get("name", "(未命名)")
        print(f"[{i}/{len(cases)}] 跑：{name} ...", flush=True)
        try:
            results.append(run_case(case, use_judge=args.judge))
        except Exception as e:
            print(f"    ⚠️ 這題跑掛了：{e}")

    if results:
        print_scorecard(results, use_judge=args.judge)


if __name__ == "__main__":
    main()
