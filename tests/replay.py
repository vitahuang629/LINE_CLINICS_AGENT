# -*- coding: utf-8 -*-
"""
replay：把真實對話（test_reply.txt）的「每一句客人問話」丟給 AI 答一遍，
        跟真人當初的回覆並排，並附自動檢查，方便你標「對/錯」建題庫。

用法（在專案根目錄）：
    python tests/replay.py                 # 跑全部
    python tests/replay.py --only 4        # 只跑第 4 段對話
    python tests/replay.py --limit 5       # 只跑前 5 個測試點（先小量試跑）

輸出：
    tests/replay_result.md      —— 人看的對照表（真人 vs AI + 自動檢查）
    tests/suggested_cases.yaml  —— 幫你生好的題庫草稿；你把「AI 答對」的挑進 cases.yaml

注意：會真的呼叫 OpenAI、需要 .env 與向量庫，建議在能跑起 app 的環境執行
      （本機 poetry，或 docker exec 進容器）。每個測試點約 10~40 秒。
"""
import argparse
import os
import re
import sys

# 讓 `python tests/replay.py` 從根目錄執行時 import 得到 tests 套件
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注意：_harness 會拉起整個 agent（重、且需要 .env/向量庫），故延遲到 main() 才匯入，
# 讓下面的 parser（parse_conversations / build_test_points）能被單獨匯入測試。
from tests.fees import FEES, resolve_ad_referral

SRC = "test_reply.txt"
OUT_MD = "tests/replay_result.md"
OUT_CASES = "tests/suggested_cases.yaml"

_ROLE_RE = re.compile(r"^\s*(客人|客服|廣告|傳送者)\s*[:：;；]?\s*(.*)$")
_CONV_RE = re.compile(r"^\s*#\s*對話\s*(\d+)\s*(.*)$")
_BUTTON_RE = re.compile(r"^\s*(👉|想了解|想預約)")


def parse_conversations(path):
    """把 test_reply.txt 拆成 [{no, ad_text, turns:[(role,text)]}]。role ∈ customer/agent。"""
    convs = []
    cur = None
    role = None          # 目前累積中的 turn 角色：customer/agent/ad/sender/None
    buf = []

    def flush():
        nonlocal role, buf
        text = "\n".join(buf).strip()
        if cur is not None and text:
            if role == "ad":
                cur["ad_text"] += ("\n" + text) if cur["ad_text"] else text
            elif role in ("customer", "agent"):
                cur["turns"].append((role, text))
            # sender（傳送者）視為雜訊，丟棄
        buf = []

    lines = open(path, encoding="utf-8").read().splitlines()
    for line in lines:
        m_conv = _CONV_RE.match(line)
        if m_conv:
            flush()
            if cur is not None:
                convs.append(cur)
            # 標題可能自帶「廣告: XXX」
            header_rest = m_conv.group(2)
            ad_seed = ""
            if "廣告" in header_rest:
                ad_seed = re.split(r"廣告\s*[:：]?", header_rest, 1)[-1].strip()
            cur = {"no": int(m_conv.group(1)), "ad_text": ad_seed, "turns": []}
            role, buf = None, []
            continue

        m_role = _ROLE_RE.match(line)
        if m_role:
            flush()
            tag, inline = m_role.group(1), m_role.group(2)
            role = {"客人": "customer", "客服": "agent", "廣告": "ad", "傳送者": "sender"}[tag]
            buf = [inline] if inline.strip() else []
        else:
            buf.append(line)

    flush()
    if cur is not None:
        convs.append(cur)
    return convs


def build_test_points(conv):
    """
    從一段對話產生測試點：把「連續的客人訊息」併成一個 block，
    參考答案 = 該 block 後面第一則客服訊息。回傳 [{content, reference, history, is_first, weak}]。
    history 為「舊到新」，客服訊息加上 [真人客服] 前綴。
    """
    points, history = [], []
    turns = conv["turns"]
    i = 0
    while i < len(turns):
        rolle, text = turns[i]
        if rolle == "customer":
            block = []
            while i < len(turns) and turns[i][0] == "customer":
                block.append(turns[i][1])
                i += 1
            reference = turns[i][1] if i < len(turns) and turns[i][0] == "agent" else None
            content = "\n".join(block).strip()
            weak = bool(_BUTTON_RE.match(content)) and len(content) <= 18
            points.append({
                "content": content,
                "reference": reference,
                "history": list(history),
                "is_first": len(history) == 0,
                "weak": weak,
            })
            for t in block:
                history.append({"type": "human", "content": t})
        else:  # agent
            history.append({"type": "ai", "content": "[真人客服] " + text})
            i += 1
    return points


def prices_in_reference(reference):
    """參考答案裡出現、且在費用表內的價格（當作『AI 應該報的價』提示）。"""
    if not reference:
        return []
    hit = []
    for f in FEES:
        p = f["price"]
        if str(p) in reference or f"{p:,}" in reference:
            hit.append(p)
    return sorted(set(hit))


def yaml_str(s):
    """簡單把字串包成單行 yaml 值。"""
    return '"' + (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip() + '"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, help="只跑指定對話編號")
    ap.add_argument("--limit", type=int, help="只跑前 N 個測試點")
    args = ap.parse_args()

    # 延遲匯入：跑實測才需要 agent（重、需 .env/向量庫）
    from tests._harness import build_query, run_agent, judge_answer, find_simplified

    convs = parse_conversations(SRC)
    md = ["# Replay 對照結果（真人 vs AI）", "", "<!--SUMMARY-->", ""]  # SUMMARY 佔位，最後回填
    cases = ["# 自動生成的題庫草稿 —— 把『AI 答對』的整段挑進 tests/cases.yaml，再微調 expect", ""]
    count = 0
    # 準確率統計（以每個測試點的「代表回覆」= 第一個 run 計分）
    verdicts = {"correct": 0, "partial": 0, "wrong": 0, "na": 0}
    simplified_points = []   # 夾了簡體字的測試點

    for conv in convs:
        if args.only and conv["no"] != args.only:
            continue
        ad_referral = resolve_ad_referral(conv["ad_text"])
        md.append(f"\n## 對話 {conv['no']}　(廣告來源推定: {ad_referral or '無'})\n")

        for idx, pt in enumerate(build_test_points(conv), 1):
            if args.limit and count >= args.limit:
                break
            if not pt["content"]:
                continue
            count += 1

            # 首次對話且有廣告 → 跑「帶 / 不帶 ad_referral」兩種
            runs = []
            if pt["is_first"] and ad_referral:
                runs = [("帶ad_referral", ad_referral), ("不帶ad_referral", None)]
            else:
                runs = [("一般", ad_referral if not pt["history"] else None)]

            hint_prices = prices_in_reference(pt["reference"])
            md.append(f"### 對話{conv['no']}-{idx}　{'（弱：按鈕點擊，可略）' if pt['weak'] else ''}")
            md.append(f"- **上下文**：{len(pt['history'])} 則歷史")
            md.append(f"- **客人問**：{pt['content']}")
            md.append(f"- **真人答（參考）**：{pt['reference'] or '（此題真人沒有回覆／對話結束）'}")

            primary_text = None   # 代表回覆（第一個 run），拿去給評審計分
            for run_i, (label, adref) in enumerate(runs):
                q = build_query(pt["content"], pt["history"], adref)
                try:
                    resp = run_agent(q)
                    ai_text = (resp.text or "").replace("\n", " ")
                    if run_i == 0:
                        primary_text = resp.text or ""
                    checks = []
                    for p in hint_prices:
                        ok = str(p) in (resp.text or "") or f"{p:,}" in (resp.text or "")
                        checks.append(f"{'✅' if ok else '❌'}價{p:,}")
                    # 其他費用表價格若冒出來 → 疑似報錯療程
                    others = [f["price"] for f in FEES if f["price"] not in hint_prices
                              and (str(f["price"]) in (resp.text or "") or f"{f['price']:,}" in (resp.text or ""))]
                    # 簡體字偵測（確定性）
                    simp = find_simplified(resp.text or "")
                    check_line = " ".join(checks) + (f"　⚠️另出現價:{others}" if others else "")
                    if simp:
                        check_line += f"　🈲簡體字:{''.join(simp)}"
                        if run_i == 0:
                            simplified_points.append(f"對話{conv['no']}-{idx}")
                    md.append(f"- **AI答（{label}）**：{ai_text}")
                    md.append(f"    - CallCS={resp.CallCS}　handoff={resp.trace.get('handoff_reason') if resp.trace else None}　images={resp.images}")
                    md.append(f"    - 自動檢查：{check_line or '（此題無價格提示）'}")
                except Exception as e:
                    md.append(f"- **AI答（{label}）**：❌ 執行錯誤：{e}")

            # ── LLM 評審（以代表回覆計分）──
            v = judge_answer(pt["content"], pt["reference"], primary_text or "")
            verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
            icon = {"correct": "✅正確", "partial": "🟡部分", "wrong": "❌錯誤", "na": "⚪不列入"}.get(v["verdict"], v["verdict"])
            md.append(f"- **評審**：{icon} —— {v['reason']}")
            md.append("- **你的判定**： 對 / 錯 　（對→挑進 cases.yaml）")
            md.append("")

            # 產生一則題庫草稿（call_cs 先留白，你確認 AI 對了再填實際值）
            cases.append(f"- name: 對話{conv['no']}_{idx}")
            cases.append(f"  input:")
            if pt["is_first"] and ad_referral:
                cases.append(f"    ad_referral: {yaml_str(ad_referral)}")
            if pt["history"]:
                cases.append(f"    message_history:   # 由舊到新（自然順序）")
                for h in pt["history"]:
                    cases.append(f"      - {{ type: {h['type']}, content: {yaml_str(h['content'])} }}")
            cases.append(f"    content: {yaml_str(pt['content'])}")
            cases.append(f"  expect:")
            cases.append(f"    # 真人答提到的價格（請確認 AI 也該報這些）：")
            if hint_prices:
                cases.append(f"    prices_present: {hint_prices}")
            cases.append(f"    # call_cs: 0")
            cases.append(f"    # prices_absent: []")
            cases.append("")

        if args.limit and count >= args.limit:
            break

    # ── 準確率彙總（correct=1、partial=0.5、wrong=0；na 不列入分母）──
    c, p, w, na = verdicts["correct"], verdicts["partial"], verdicts["wrong"], verdicts["na"]
    scored = c + p + w
    accuracy = (c + 0.5 * p) / scored * 100 if scored else 0.0
    summary = [
        "## 📊 準確率彙總",
        "",
        f"- **準確率：{accuracy:.1f}%**　（correct=1、partial=0.5、wrong=0；na 不列入分母）",
        f"- ✅ 正確 {c}　🟡 部分 {p}　❌ 錯誤 {w}　⚪ 不列入 {na}　（共 {count} 題）",
        f"- 🈲 夾簡體字的回覆：{len(simplified_points)} 題" + (f"（{', '.join(simplified_points)}）" if simplified_points else ""),
        "",
    ]
    full = "\n".join(md).replace("<!--SUMMARY-->", "\n".join(summary))
    open(OUT_MD, "w", encoding="utf-8").write(full)
    open(OUT_CASES, "w", encoding="utf-8").write("\n".join(cases))
    print(f"✅ 完成：跑了 {count} 個測試點")
    print(f"📊 準確率：{accuracy:.1f}%　(正確{c} 部分{p} 錯誤{w} 不列入{na})")
    print(f"🈲 夾簡體字：{len(simplified_points)} 題")
    print(f"   對照表 → {OUT_MD}")
    print(f"   題庫草稿 → {OUT_CASES}")


if __name__ == "__main__":
    main()
