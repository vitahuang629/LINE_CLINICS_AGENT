# -*- coding: utf-8 -*-
"""
回歸測試：讀 cases.yaml，每題丟給 AI 答一遍，驗「答對的特徵」。

⭐ 抗 LLM 隨機性：每題最多跑 RUNS 次，**至少 NEEDED 次通過**才算綠（多數決），
   避免偶發抽風造成「假性變紅」。含早停：已達多數就不再跑，省 API。

跑法（專案根目錄）：
    pytest tests/test_regression.py -v            # 全部（預設 3 次取 2）
    REGRESSION_RUNS=1 pytest tests/test_regression.py -v   # 快速模式：每題只跑 1 次

紅的那題 = 這次改動把它弄壞了（或該題 expect 需要更新）。
會真的呼叫 OpenAI，需要 .env 與向量庫。
"""
import os

import pytest
import yaml

from tests._harness import build_query, run_agent, check_expectations

_CASES_PATH = os.path.join(os.path.dirname(__file__), "cases.yaml")

# 每題最多跑幾次、至少要過幾次（多數決）
RUNS = int(os.getenv("REGRESSION_RUNS", "3"))
NEEDED = RUNS // 2 + 1


def _load_cases():
    with open(_CASES_PATH, encoding="utf-8") as f:
        cases = yaml.safe_load(f) or []
    return [(c["name"], c) for c in cases]


@pytest.mark.parametrize("name,case", _load_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_case(name, case):
    inp = case["input"]
    expect = case.get("expect", {})

    passes = 0
    attempts = 0
    failed_runs = []          # [(第幾次, resp, fails)]

    for i in range(RUNS):
        attempts += 1
        resp = run_agent(build_query(
            content=inp["content"],
            history_chrono=inp.get("message_history"),
            ad_referral=inp.get("ad_referral"),
        ))
        fails = check_expectations(resp, expect)
        if fails:
            failed_runs.append((i + 1, resp, fails))
        else:
            passes += 1

        # 早停 1：已達多數 → 綠，不必再跑
        if passes >= NEEDED:
            return
        # 早停 2：剩下的次數已不足以湊到多數 → 紅，不必再跑
        if (attempts - passes) > (RUNS - NEEDED):
            break

    detail = "\n".join(
        f"  [第 {n} 次] AI回覆={resp.text!r}\n"
        f"            CallCS={resp.CallCS} images={resp.images}\n"
        f"            不符：{'; '.join(fails)}"
        for n, resp, fails in failed_runs
    )
    pytest.fail(
        f"\n題目：{name}"
        f"\n客人問：{inp['content']}"
        f"\n結果：{attempts} 次中只過 {passes} 次（需 {NEEDED}/{RUNS}）\n{detail}",
        pytrace=False,
    )
