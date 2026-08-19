#!/usr/bin/env python3
"""
V2 Eval — 31 条用例自动评测。
用法: cd ai-portfolio && python eval/eval.py
"""

import json, sys, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import db
from agent_v2 import chat


def normalize_intent(intent_result: dict) -> list:
    """从 chat 返回的 intent_result 中提取意图字符串列表。"""
    if "chat_intent" in intent_result:
        return [intent_result["chat_intent"]]
    intents = intent_result.get("intents", [])
    result = []
    for i in intents:
        op = i.get("operation", "?")
        dim = i.get("dimension") or ""
        metric = i.get("metric") or ""
        # recommend 操作：metric 固定为 neg_rate，归一化统一格式
        if op == "recommend":
            result.append("recommend×category×neg_rate")
        elif dim and metric:
            result.append(f"{op}×{dim}×{metric}")
        elif dim:
            result.append(f"{op}×{dim}")
        else:
            result.append(op)
    return result


def run_case(tc: dict) -> dict:
    """运行单条用例，返回 {pass, failures}。"""
    failures = []
    turns = tc.get("turns")
    q = tc.get("q")

    # ── 执行对话 ──
    if turns:
        # 多轮：逐轮调用，维护 history
        history = []
        for i, turn_q in enumerate(turns):
            result = chat(turn_q, history or None)
            intent_result, entities, all_data, answer, trace = result
            if i < len(turns) - 1:
                history.append(f"用户：{turn_q}")
                history.append(f"助手：{answer[:100]}")
        # 最后一轮的结果用于判分
    else:
        result = chat(q)
        intent_result, entities, all_data, answer, trace = result

    actual_intents = normalize_intent(intent_result)
    answer_text = answer if answer else ""

    # ── 1. 意图匹配 ──
    expected = tc.get("expected_intents") or []
    if expected:
        for exp in expected:
            if exp == "引导":
                # 特殊：期望引导（缺参/不支持），不匹配具体 intent
                # 检查回答是否包含引导性文字
                guide_keywords = ["告诉", "提供", "补充", "需要", "至少", "再"]
                if not any(kw in answer_text for kw in guide_keywords):
                    failures.append(f"意图: 期望引导, 实际intents={actual_intents}")
            elif exp == "capability":
                if "capability" not in actual_intents:
                    failures.append(f"意图: 期望capability, 实际={actual_intents}")
            elif exp == "methodology":
                if "methodology" not in actual_intents:
                    failures.append(f"意图: 期望methodology, 实际={actual_intents}")
            elif exp == "unsupported":
                if "unsupported" not in actual_intents:
                    failures.append(f"意图: 期望unsupported, 实际={actual_intents}")
            elif exp == "other":
                if "other" not in actual_intents:
                    failures.append(f"意图: 期望other, 实际={actual_intents}")
            else:
                # 业务意图：精确匹配
                if exp not in actual_intents:
                    failures.append(f"意图: 期望{exp}, 实际={actual_intents}")

    # ── 2. 禁止词 ──
    banned = tc.get("banned") or []
    for word in banned:
        if word in answer_text:
            failures.append(f"禁止词: '{word}' 出现在回答中")

    # ── 3. 禁止回答包含某意图关键词 ──
    banned_intent = tc.get("banned_answer_contains_intent") or []
    # 该字段历史上有两种类型：bool（True/False）或 数组（关键词列表）。bool 跳过。
    if isinstance(banned_intent, list):
        for word in banned_intent:
            if word in answer_text:
                failures.append(f"禁止回答含: '{word}'")

    # ── 4. 必须词（任一匹配即可）──
    required_any = tc.get("required_any") or []
    if required_any:
        if not any(word in answer_text for word in required_any):
            failures.append(f"必须词: 缺少 {required_any} 中的任何一个")

    # ── 5. 双层评审（Eval 用例也评审，打通质量防线）──
    review = None
    try:
        from agent_v2 import _run_dual_review
        review = _run_dual_review(q, answer, all_data or [], user="eval")
        if review:
            mj = review.get("model_judge") or {}
            overall = mj.get("overall") or 0
            if overall < 6:  # 评审分低于 6 判失败
                failures.append(f"评审低分: {overall}/10 — {mj.get('comment', '')[:40]}")
    except Exception as e:
        failures.append(f"评审异常: {str(e)[:50]}")

    return {
        "pass": len(failures) == 0,
        "failures": failures,
        "actual_intents": actual_intents,
        "answer_preview": answer_text[:80],
        "review": review,
    }


def _run_case_timed(idx_tc):
    """线程池 worker：运行单条用例并计时"""
    idx, tc = idx_tc
    t0 = time.time()
    r = run_case(tc)
    elapsed = time.time() - t0
    return idx, tc, r, elapsed


def main():
    cases = db.get_cases()
    workers = min(6, len(cases))  # 并发数 6，避免限流

    print(f"\n{'═' * 60}")
    print(f"  V2 Eval — {len(cases)} 条用例（并发×{workers}，{'Supabase' if db.is_using_supabase() else '本地 JSONL'}）")
    print(f"{'═' * 60}\n")

    t_start = time.time()

    # ── 并发执行 ──
    raw_results = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_case_timed, (i, tc)): i for i, tc in enumerate(cases)}
        done_count = 0
        for fut in as_completed(futures):
            idx, tc, r, elapsed = fut.result()
            raw_results[idx] = (tc, r, elapsed)
            done_count += 1
            tc_id = tc.get("id", idx + 1)
            desc = tc.get("q") or " → ".join(tc.get("turns") or [])
            status = "✅" if r["pass"] else "❌"
            print(f"[{done_count:2d}/{len(cases)}] {status} [{tc_id:2d}] {desc[:50]} ({elapsed:.1f}s)", flush=True)

    t_total = time.time() - t_start

    # ── 按原始顺序排列 ──
    results = [(tc, r) for tc, r, _ in raw_results]
    timings = [elapsed for _, _, elapsed in raw_results]

    # ── 汇总 ──
    total = len(results)
    passed = sum(1 for _, r in results if r["pass"])
    failed = total - passed

    # 按类别统计
    sections = {}
    for tc, r in results:
        s = tc.get("section", "其他")
        if s not in sections:
            sections[s] = {"total": 0, "pass": 0}
        sections[s]["total"] += 1
        if r["pass"]:
            sections[s]["pass"] += 1

    # 意图准确率（仅计算有 expected_intents 的用例）
    intent_cases = [(tc, r) for tc, r in results if tc.get("expected_intents")]
    intent_pass = sum(1 for tc, r in intent_cases if r["pass"] or not any("意图:" in f for f in r["failures"]))
    intent_total = len(intent_cases)

    # 防幻觉通过率（有 banned 的用例）
    halluc_cases = [(tc, r) for tc, r in results if tc.get("banned")]
    halluc_pass = sum(1 for tc, r in halluc_cases if not any("禁止词:" in f for f in r["failures"]))
    halluc_total = len(halluc_cases)

    # 双层评审通过率（有评审结果的用例，分 >= 6 通过）
    review_cases = [(tc, r) for tc, r in results if r.get("review")]
    review_pass = sum(1 for tc, r in review_cases
                      if not any("评审" in f for f in r["failures"]))
    review_total = len(review_cases)

    print(f"\n{'═' * 60}")
    print(f"  📊 Eval 结果")
    print(f"{'═' * 60}")
    print(f"  总用例:      {total}")
    print(f"  通过:        {passed}")
    print(f"  失败:        {failed}")
    print(f"  总通过率:    {passed/total*100:.1f}%")
    print(f"  总耗时:      {t_total:.1f}s（串行预估 {sum(timings):.0f}s，提速 {sum(timings)/max(t_total,0.1):.1f}×）")
    print()
    for s, v in sections.items():
        print(f"  [{s}] {v['pass']}/{v['total']} 通过")
    print()
    print(f"  意图准确率:  {intent_pass}/{intent_total} ({intent_pass/max(intent_total,1)*100:.1f}%)")
    print(f"  防幻觉通过:  {halluc_pass}/{halluc_total} ({halluc_pass/max(halluc_total,1)*100:.1f}%)")
    print(f"  评审通过:    {review_pass}/{review_total} ({review_pass/max(review_total,1)*100:.1f}%)")

    if failed > 0:
        print(f"\n{'─' * 60}")
        print(f"  ❌ 失败明细")
        print(f"{'─' * 60}")
        for tc, r in results:
            if not r["pass"]:
                desc = tc.get("q") or " → ".join(tc.get("turns") or [])
                print(f"\n  [{tc.get('id', '?')}] {desc}")
                for f in r["failures"]:
                    print(f"    ✗ {f}")
                print(f"    意图: {r['actual_intents']}")
                print(f"    回答: {r['answer_preview']}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
