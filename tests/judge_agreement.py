# -*- coding: utf-8 -*-
"""LLM-as-judge 的信度檢查：算「AI 評審」與「你的人工標註」的一致率。

為什麼需要：judge 的分數目前沒人驗證過。如果 judge 跟你的判斷對不起來，
那 eval_baseline 的「回覆品質」欄位就只是一個看起來很專業的亂數，
拿它當優化基準線會把模型往錯的方向調。

資料來源：tests/replay_result.md（49 題，已含客人問／真人答／AI答／judge 判定）。
不必重跑 agent，所以整個流程**不花 API 錢**。

用法：
    python tests/judge_agreement.py --make-sheet     # 產生盲標註表 → tests/annotation_sheet.md
    （你填完 annotation_sheet.md 的「判定：」欄位）
    python tests/judge_agreement.py --score          # 算一致率 + κ + 混淆矩陣 + 歧異清單

⚠️ 標註表刻意**不顯示 judge 的判定**。看得到 judge 答案再標，等於抄答案，
   算出來的一致率會虛高、沒有參考價值。
"""
import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 讓 `python tests/judge_agreement.py` 也 import 得到 tests.replay
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPLAY_MD = os.path.join(_HERE, "replay_result.md")
_SHEET_MD = os.path.join(_HERE, "annotation_sheet.md")

_ICON2KEY = {"✅正確": "correct", "🟡部分": "partial", "❌錯誤": "wrong", "⚪不列入": "na"}
_LABELS = ["correct", "partial", "wrong", "na"]

# 人工可以填的寫法（大小寫不拘）
_ALIASES = {
    "correct": "correct", "c": "correct", "正確": "correct", "對": "correct",
    "partial": "partial", "p": "partial", "部分": "partial",
    "wrong": "wrong", "w": "wrong", "錯誤": "wrong", "錯": "wrong",
    "na": "na", "n": "na", "不列入": "na", "無法判斷": "na",
}

# 標註原則直接抄 _harness._JUDGE_SYSTEM 的判定標準 —— 人跟 judge 必須用同一把尺，
# 尺不同的話算出來的是「兩套標準的差異」，不是「judge 準不準」。
_RUBRIC = """\
判定標準（跟 AI 評審用同一把尺，請照這個標）：

- correct：正確回應了客人這句**實際問的問題**；有講到的可查核事實（價／地址／次數）正確；
           無捏造價、無療效誇大。
           ※ 客人沒問的，AI 沒講，**不扣分**（例：客人在回答年齡／部位時，AI 沒報價 → 不扣分）
- partial：客人一句問了多件事，AI 只答到一部分；或核心答對但漏掉客人確實有問的某一項。
- wrong  ：關鍵事實答錯（報錯價／錯地址）、捏造價格數字、療效誇大（保證一定瘦／治癒），
           或完全沒回應客人問的核心問題（答非所問）。
- na     ：無法公平評分 —— 真人用了 AI 拿不到的資訊（查會員／查預約系統／「幫您查看」），
           或客人這句只是純寒暄／開場、沒有可查核的對錯。

「真人答」只是**參考**，不是必須逐項複述的清單。AI 沒複述客人沒問到的額外資訊，不扣分。
"""


def _load_histories():
    """從 test_reply.txt 還原每個測試點的對話歷史 → {「對話3-1」: [{type, content}, …]}。

    replay_result.md 只記「N 則歷史」沒記內容，但 agent 當初是看著完整歷史回答的，
    標註的人（和 judge）看不到就沒辦法判斷「這句有沒有接住上文」。
    這裡用 replay.py 同一套切分邏輯重建，純文字處理、不呼叫 API。
    """
    from tests.replay import SRC, build_test_points, parse_conversations

    out = {}
    for conv in parse_conversations(SRC):
        for idx, pt in enumerate(build_test_points(conv), 1):
            if not pt["content"]:            # 與 replay.py 一致：空 content 跳過但 idx 照跑
                continue
            out[f"對話{conv['no']}-{idx}"] = pt["history"]
    return out


def _parse_replay(path=_REPLAY_MD):
    """把 replay_result.md 拆成 [{id, question, reference, ai, verdict, reason}]。"""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    items, cur, field = [], None, None
    for ln in lines:
        if ln.startswith("### "):
            if cur:
                items.append(cur)
            cur = {"id": ln[4:].strip().rstrip("　"),
                   "question": "", "reference": "", "ai": "", "verdict": "", "reason": ""}
            field = None
            continue
        if cur is None:
            continue

        m = re.match(r"- \*\*(.+?)\*\*：(.*)", ln)
        if m:
            name, val = m.group(1), m.group(2)
            if name == "客人問":
                field = "question"
            elif name.startswith("真人答"):
                field = "reference"
            # 一題可能有多個 AI答（帶／不帶 ad_referral）；judge 只評第一個，這裡也只取第一個
            elif name.startswith("AI答"):
                field = "ai" if not cur["ai"] else None
            elif name == "評審":
                icon = val.split(" ")[0]
                cur["verdict"] = _ICON2KEY.get(icon, "")
                cur["reason"] = val.split("—— ", 1)[-1] if "—— " in val else ""
                field = None
            else:                      # 上下文 / 你的判定 …
                field = None
            if field:
                cur[field] = val
            continue

        if ln.startswith("    - ") or ln.startswith("## ") or ln.startswith("# "):
            field = None               # CallCS／自動檢查那幾行，不屬於任何欄位
            continue
        if field:                      # 多行欄位的續行
            cur[field] += "\n" + ln

    if cur:
        items.append(cur)
    return [i for i in items if i["id"]]


def _history_block(history):
    """把歷史排成好讀的逐字稿；沒有歷史就標明是對話開頭。"""
    if not history:
        return "（無，這是對話第一句）"
    lines = []
    for h in history:
        who = "客人" if h.get("type") == "human" else "客服"
        text = (h.get("content") or "").replace("[真人客服] ", "").replace("\n", " ")
        lines.append(f"> **{who}**：{text}")
    return "\n".join(lines)


def make_sheet(items):
    histories = _load_histories()
    old_labels = _parse_sheet(_SHEET_MD) if os.path.exists(_SHEET_MD) else {}
    if old_labels:
        print(f"偵測到你已經標了 {len(old_labels)} 題 → 會原封不動保留")

    out = [
        "# 人工標註表（LLM-as-judge 一致率用）",
        "",
        f"共 {len(items)} 題。請在每題的「判定：」後面填一個值：",
        "`correct` / `partial` / `wrong` / `na`（也可以簡寫 c / p / w / n，或中文 正確/部分/錯誤/不列入）。",
        "",
        "⚠️ 這張表**刻意不顯示 AI 評審的判定**——先看到答案再標，算出來的一致率沒有意義。",
        "填完存檔，再跑 `python tests/judge_agreement.py --score`。",
        "",
        "```",
        _RUBRIC,
        "```",
        "",
        "---",
        "",
    ]
    for n, it in enumerate(items, 1):
        short = re.match(r"對話\d+-\d+", it["id"])
        hist = histories.get(short.group(0) if short else "", [])
        out += [
            f"### [{n:02d}] {it['id']}",
            "",
            "**先前對話**（舊 → 新，AI 回答時看得到這些）：",
            _history_block(hist),
            "",
            f"**客人這句**：{it['question'].strip()}",
            "",
            f"**真人答（參考）**：{it['reference'].strip() or '（無）'}",
            "",
            f"**AI 答**：{it['ai'].strip() or '（空）'}",
            "",
            f"判定：{old_labels.get(n, '')}",
            "",
            "---",
            "",
        ]
    with open(_SHEET_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"已產生 {_SHEET_MD}（{len(items)} 題）")
    print("填完「判定：」欄位後，跑：python tests/judge_agreement.py --score")


def _parse_sheet(path=_SHEET_MD):
    """讀回人工標註 → {題號: label}；沒填的略過。"""
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}，請先跑 --make-sheet")
    labels, cur_no = {}, None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            m = re.match(r"### \[(\d+)\]", ln)
            if m:
                cur_no = int(m.group(1))
                continue
            m = re.match(r"判定：\s*(.*)", ln.strip())
            if m and cur_no is not None:
                raw = m.group(1).strip().lower()
                if raw:
                    key = _ALIASES.get(raw)
                    if key is None:
                        print(f"⚠️ 第 {cur_no} 題填了看不懂的值：{raw!r}（略過）")
                    else:
                        labels[cur_no] = key
                cur_no = None
    return labels


def rejudge(items):
    """拿 replay_result.md 裡現成的（問題／真人答／AI答）重跑當前 judge，一題兩版：

      nohist —— 不給上下文（重現 replay_result.md 的評分條件）
      hist   —— 給完整上下文（AI 當初看得到什麼，judge 就看得到什麼）

    同一批題目、同一套程式路徑跑兩版，差異才能歸因到「上下文」而不是別的變因。
    不重跑 agent，只重跑評審 → 49 題 × 2 版 = 98 次呼叫。
    """
    import json

    from tests._harness import judge_answer

    histories = _load_histories()
    out = {}
    for n, it in enumerate(items, 1):
        short = re.match(r"對話\d+-\d+", it["id"])
        hist = histories.get(short.group(0) if short else "", [])
        q, ref, ai = it["question"].strip(), it["reference"].strip(), it["ai"].strip()
        print(f"[{n:02d}/{len(items)}] {it['id']} …", flush=True)
        a = judge_answer(question=q, reference=ref, ai_answer=ai)
        b = judge_answer(question=q, reference=ref, ai_answer=ai, history=hist)
        out[str(n)] = {
            "id": it["id"], "n_history": len(hist),
            "nohist": a["verdict"], "nohist_reason": a["reason"],
            "hist": b["verdict"], "hist_reason": b["reason"],
        }
    with open(_REJUDGE_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n已寫入 {_REJUDGE_JSON}　接著跑：python tests/judge_agreement.py --score")


def _load_rejudge():
    import json
    if not os.path.exists(_REJUDGE_JSON):
        return {}
    with open(_REJUDGE_JSON, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def _kappa(pairs):
    """Cohen's κ：扣掉「隨機也會猜對」的部分，比生一致率誠實。"""
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(1 for h, j in pairs if h == j) / n
    pe = 0.0
    for lab in _LABELS:
        pe += (sum(1 for h, _ in pairs if h == lab) / n) * (sum(1 for _, j in pairs if j == lab) / n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _report(title, pairs):
    n = len(pairs)
    agree = sum(1 for h, j in pairs if h == j)
    bin_agree = sum(1 for h, j in pairs
                    if (h == "correct") == (j == "correct"))
    print(f"  {title:<22}{agree:>3}/{n:<4}{agree / n * 100:>7.1f}%"
          f"{_kappa(pairs):>9.3f}{bin_agree / n * 100:>10.1f}%")


def score(items):
    human = _parse_sheet()
    if not human:
        raise SystemExit("annotation_sheet.md 裡一題都還沒填")
    rej = _load_rejudge()

    # 三個版本的 judge：存檔的（7/23 那次，無上下文）、重跑無上下文、重跑有上下文
    variants = {"存檔（無上下文）": [], "重跑（無上下文）": [], "重跑（有上下文）": []}
    rows = []
    for n, it in enumerate(items, 1):
        if n not in human:
            continue
        h = human[n]
        variants["存檔（無上下文）"].append((h, it["verdict"]))
        if n in rej:
            variants["重跑（無上下文）"].append((h, rej[n]["nohist"]))
            variants["重跑（有上下文）"].append((h, rej[n]["hist"]))
        rows.append((n, it, h))

    print("\n" + "=" * 62)
    print("  LLM-as-judge vs 人工標註")
    print("=" * 62)
    print(f"  已標註：{len(rows)} / {len(items)} 題"
          + ("" if rej else "　（尚未跑 --rejudge，只比得了存檔那版）"))
    print("-" * 62)
    print(f"  {'judge 版本':<22}{'一致':>7}{'四類':>8}{'κ':>9}{'二分':>10}")
    for name, pairs in variants.items():
        if pairs:
            _report(name, pairs)
    print("=" * 62)

    main_pairs = variants["重跑（有上下文）"] or variants["存檔（無上下文）"]
    print("  混淆矩陣（列＝你，欄＝judge"
          + ("・有上下文版" if variants["重跑（有上下文）"] else "・存檔版") + "）")
    print(f"  {'':<10}" + "".join(f"{l:>9}" for l in _LABELS))
    for h in _LABELS:
        cells = [sum(1 for hh, jj in main_pairs if hh == h and jj == j) for j in _LABELS]
        print(f"  {h:<10}" + "".join(f"{c:>9}" for c in cells))
    print("=" * 62)

    # 歧異清單：以「有上下文」那版為主（沒跑 rejudge 就退回存檔版）
    print()
    shown = 0
    for num, it, h in rows:
        j = rej[num]["hist"] if num in rej else it["verdict"]
        reason = rej[num]["hist_reason"] if num in rej else it["reason"]
        if h == j:
            continue
        shown += 1
        extra = f"（無上下文版判 {rej[num]['nohist']}）" if num in rej else ""
        print(f"[{num:02d}] {it['id']}　你={h}　judge={j} {extra}")
        print(f"     客人問：{it['question'].strip()[:60]}")
        print(f"     judge 理由：{reason[:110]}")
        print()
    if not shown:
        print("（完全一致）\n")

    print("判讀：κ ≥ 0.6 算堪用、≥ 0.8 算好。")
    print("      比較「有／無上下文」兩列 → 補上下文到底幫了多少。")
    print("      歧異若集中在 correct↔partial，通常是「客人沒問的算不算漏答」界線不同")
    print("      → 改 tests/_harness.py 的 _JUDGE_SYSTEM。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sheet", action="store_true", help="產生盲標註表（會保留已填的判定）")
    ap.add_argument("--rejudge", action="store_true",
                    help="用當前 judge 重跑那 49 題的『有／無上下文』兩版（98 次 API 呼叫）")
    ap.add_argument("--score", action="store_true", help="算一致率")
    args = ap.parse_args()

    items = _parse_replay()
    if args.make_sheet:
        make_sheet(items)
    elif args.rejudge:
        rejudge(items)
    elif args.score:
        score(items)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
