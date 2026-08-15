#!/usr/bin/env python3
"""
运营后台 — 履约 AI 助手 V2.0
5 大模块：指标看板 / 模型管理 / 测评管理 / Agent 管理 / （Eval 评测中心保留）
"""

import streamlit as st
import json
import os
import re
import subprocess
import sys
import yaml
from datetime import datetime, timedelta
from collections import Counter

st.set_page_config(page_title="运营后台 · V2.0", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # pages/ 的上一级 = 项目根
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LOG_DIR = os.path.join(BASE_DIR, "logs")
EVAL_DIR = os.path.join(BASE_DIR, "eval")
MODEL_FILE = os.path.join(CONFIG_DIR, "model.yaml")
PROMPTS_FILE = os.path.join(CONFIG_DIR, "prompts.yaml")
FILTERS_FILE = os.path.join(CONFIG_DIR, "filters.yaml")
TOKEN_LOG = os.path.join(LOG_DIR, "token_usage.jsonl")
CONV_LOG = os.path.join(LOG_DIR, "conversations.jsonl")
EVAL_LOG = os.path.join(LOG_DIR, "eval_results.jsonl")
TOGGLES_FILE = os.path.join(LOG_DIR, "toggles.json")
CASES_FILE = os.path.join(EVAL_DIR, "cases.jsonl")
PROMPT_VERSIONS_FILE = os.path.join(LOG_DIR, "prompt_versions.jsonl")
EVALUATIONS_LOG = os.path.join(LOG_DIR, "evaluations.jsonl")
BUG_FEEDBACK_LOG = os.path.join(LOG_DIR, "bug_feedback.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# ── 模型单价（$/1M tokens，可编辑）──
MODEL_PRICES = {
    "deepseek-v4-flash": {"input": 0.27, "output": 1.10},
    "deepseek-v4-pro":   {"input": 2.19, "output": 8.76},
}


# ── 全局样式 ──
st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Inter", "Noto Sans SC", sans-serif; }
.kpi-card {
    background: #fff; border: 1px solid #e8e8e8; border-radius: 12px;
    padding: 20px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
.kpi-label { font-size: 0.85rem; color: #888; margin-top: 4px; }
.intent-chip {
    display: inline-block; padding: 2px 10px; margin: 2px;
    border-radius: 12px; font-size: 0.78rem; font-weight: 500;
    background: #e8eaf6; color: #3949ab;
}
</style>
""", unsafe_allow_html=True)


# ── 工具函数 ──
def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def save_jsonl(path: str, rows: list):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_toggles() -> dict:
    if os.path.exists(TOGGLES_FILE):
        with open(TOGGLES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_toggles(toggles: dict):
    with open(TOGGLES_FILE, "w", encoding="utf-8") as f:
        json.dump(toggles, f, ensure_ascii=False, indent=2)


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """估算单次 API 调用成本（美元）"""
    price = MODEL_PRICES.get(model, MODEL_PRICES["deepseek-v4-flash"])
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def reload_agent_config():
    """热重载 agent 配置"""
    try:
        sys.path.insert(0, BASE_DIR)
        from agent_v2 import reload_config
        reload_config()
    except Exception:
        pass


# ── 侧边栏导航 ──
with st.sidebar:
    st.markdown("### 📊 运营后台")
    st.markdown("---")

PAGES = {
    "① 指标看板": "dashboard",
    "② 模型管理": "model",
    "③ Eval 评测中心": "eval",
    "④ 测评管理": "eval_mgmt",
    "⑤ Agent 管理": "agent",
}

page = st.sidebar.radio("导航", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
if st.sidebar.button("← 返回助手", use_container_width=True):
    st.switch_page("app_v2.py")


# ════════════════════════════════════════
# 模块 ① 指标看板
# ════════════════════════════════════════
def page_dashboard():
    st.title("📈 指标看板")
    st.caption("Agent 层 + 模型层合一的运营视图")

    convs = load_jsonl(CONV_LOG)
    tokens = load_jsonl(TOKEN_LOG)
    evals = load_jsonl(EVAL_LOG)

    # ── KPI 卡片 ──
    total_convs = len(convs)
    total_calls = len(tokens)
    total_prompt = sum(t.get("prompt_tokens", 0) for t in tokens)
    total_completion = sum(t.get("completion_tokens", 0) for t in tokens)
    total_tokens = sum(t.get("total_tokens", 0) for t in tokens)
    total_cost = sum(calc_cost(t.get("model", ""), t.get("prompt_tokens", 0), t.get("completion_tokens", 0)) for t in tokens)

    last_eval_pass = "—"
    if evals:
        last_eval_pass = f"{evals[-1].get('pass_rate', 0):.0%}"

    # 统计坏例数
    cases = load_jsonl(CASES_FILE)
    bad_count = sum(1 for c in cases if c.get("section") == "坏例审核")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{total_convs}</div><div class="kpi-label">总对话数</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{total_calls}</div><div class="kpi-label">模型调用数</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{total_tokens:,}</div><div class="kpi-label">总 Token</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">${total_cost:.4f}</div><div class="kpi-label">预估成本</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{last_eval_pass}</div><div class="kpi-label">Eval 通过率</div></div>""", unsafe_allow_html=True)
    with col6:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{bad_count}</div><div class="kpi-label">坏例数</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 每日调用量趋势 ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📅 最近 7 天调用量")
        if tokens:
            date_counts = Counter()
            for t in tokens:
                d = t.get("ts", "")[:10]
                if d:
                    date_counts[d] += 1
            today_dt = datetime.now()
            dates = [(today_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
            counts = [date_counts.get(d, 0) for d in dates]
            st.bar_chart({"日期": dates, "调用数": counts}, x="日期", y="调用数")
        else:
            st.info("暂无调用记录")

    with col_b:
        st.subheader("🎯 Eval 通过率走势")
        if evals:
            eval_dates = [e.get("ts", "")[:10] for e in evals[-14:]]
            eval_rates = [round(e.get("pass_rate", 0) * 100, 1) for e in evals[-14:]]
            st.line_chart({"日期": eval_dates, "通过率%": eval_rates}, x="日期", y="通过率%")
        else:
            st.info("暂无 Eval 记录")

    st.markdown("---")

    # ── 操作×维度×指标 命中分布 ──
    st.subheader("🔗 意图命中分布")
    if convs:
        op_counter = Counter()
        dim_counter = Counter()
        metric_counter = Counter()
        intent_counter = Counter()

        for c in convs:
            intent = c.get("intent", "unknown")
            intent_counter[intent] += 1
            for part in intent.split("|"):
                segments = part.split("×")
                if len(segments) >= 3:
                    op_counter[segments[0]] += 1
                    dim_counter[segments[1]] += 1
                    metric_counter[segments[2]] += 1

        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.markdown("**操作 (operation)**")
            if op_counter:
                st.bar_chart(dict(op_counter.most_common()))
        with col_y:
            st.markdown("**维度 (dimension)**")
            if dim_counter:
                st.bar_chart(dict(dim_counter.most_common()))
        with col_z:
            st.markdown("**指标 (metric)**")
            if metric_counter:
                st.bar_chart(dict(metric_counter.most_common()))

        st.markdown("---")
        st.subheader("🔗 完整三元组 Top 15")
        for intent_str, count in intent_counter.most_common(15):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f'<span class="intent-chip">{intent_str}</span>', unsafe_allow_html=True)
            with cols[1]:
                st.write(f"**{count}** 次")
    else:
        st.info("暂无对话记录")


# ════════════════════════════════════════
# 模块 ② 模型管理
# ════════════════════════════════════════
def page_model():
    st.title("🤖 模型管理")
    st.caption("管理主模型、裁判模型、参数配置，保存后 Agent 下次对话自动生效")

    cfg = load_yaml(MODEL_FILE)
    tokens = load_jsonl(TOKEN_LOG)

    # ── 当前配置预览 ──
    st.subheader("📋 当前配置")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("主模型", cfg.get("model", "deepseek-v4-flash"))
    with col2:
        st.metric("裁判模型", cfg.get("judge_model", "deepseek-v4-pro"))
    with col3:
        st.metric("意图温度", cfg.get("temperature_intent", 0))
    with col4:
        st.metric("回答温度", cfg.get("temperature_answer", 0.7))

    st.markdown("---")

    # ── 模型选择 + 参数配置 ──
    st.subheader("✏️ 修改配置")
    MODEL_OPTIONS = ["deepseek-v4-flash", "deepseek-v4-pro"]

    with st.form("model_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            current_model = cfg.get("model", "deepseek-v4-flash")
            model_idx = MODEL_OPTIONS.index(current_model) if current_model in MODEL_OPTIONS else 0
            model_name = st.selectbox("主模型", MODEL_OPTIONS, index=model_idx,
                                       help="回答用户问题的模型")

            temp_intent = st.slider(
                "意图识别温度", 0.0, 2.0,
                value=float(cfg.get("temperature_intent", 0)),
                step=0.1, help="越低越确定，意图识别建议 0",
            )
            max_tokens_intent = st.number_input(
                "意图识别 max_tokens", 100, 2000,
                value=int(cfg.get("max_tokens_intent", 400)), step=50,
            )

        with col_b:
            current_judge = cfg.get("judge_model", "deepseek-v4-pro")
            judge_idx = MODEL_OPTIONS.index(current_judge) if current_judge in MODEL_OPTIONS else 1
            judge_model = st.selectbox("裁判模型", MODEL_OPTIONS, index=judge_idx,
                                        help="评审主模型回答质量的模型")

            temp_answer = st.slider(
                "回答生成温度", 0.0, 2.0,
                value=float(cfg.get("temperature_answer", 0.7)),
                step=0.1, help="越高越发散，回答建议 0.5~0.8",
            )
            max_tokens_answer = st.number_input(
                "回答生成 max_tokens", 100, 2000,
                value=int(cfg.get("max_tokens_answer", 500)), step=50,
            )

        max_tokens_multi = st.number_input(
            "多意图回答 max_tokens", 100, 2000,
            value=int(cfg.get("max_tokens_multi_answer", 600)), step=50,
        )

        submitted = st.form_submit_button("💾 保存配置", type="primary", use_container_width=True)
        if submitted:
            new_cfg = {
                "model": model_name,
                "judge_model": judge_model,
                "judge_temperature": cfg.get("judge_temperature", 0.3),
                "judge_max_tokens": cfg.get("judge_max_tokens", 500),
                "temperature_intent": temp_intent,
                "temperature_answer": temp_answer,
                "max_tokens_intent": max_tokens_intent,
                "max_tokens_answer": max_tokens_answer,
                "max_tokens_multi_answer": max_tokens_multi,
            }
            save_yaml(MODEL_FILE, new_cfg)
            reload_agent_config()
            st.success("✅ 已保存到 config/model.yaml，Agent 下次对话自动生效")
            st.rerun()

    st.markdown("---")

    # ── Token 统计 ──
    st.subheader("📊 Token 统计")
    if tokens:
        # 按模型统计
        model_stats = {}
        for t in tokens:
            m = t.get("model", "unknown")
            if m not in model_stats:
                model_stats[m] = {"calls": 0, "prompt": 0, "completion": 0, "total": 0, "cost": 0.0}
            model_stats[m]["calls"] += 1
            model_stats[m]["prompt"] += t.get("prompt_tokens", 0)
            model_stats[m]["completion"] += t.get("completion_tokens", 0)
            model_stats[m]["total"] += t.get("total_tokens", 0)
            model_stats[m]["cost"] += calc_cost(m, t.get("prompt_tokens", 0), t.get("completion_tokens", 0))

        for m, stats in model_stats.items():
            with st.expander(f"**{m}** — {stats['calls']} 次调用"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("调用次数", stats["calls"])
                c2.metric("Prompt Tokens", f"{stats['prompt']:,}")
                c3.metric("Completion Tokens", f"{stats['completion']:,}")
                c4.metric("预估成本", f"${stats['cost']:.4f}")

        # ── 单价表（可编辑）──
        st.markdown("---")
        st.subheader("💰 模型单价（$/1M tokens）")
        price_data = []
        for m, p in MODEL_PRICES.items():
            price_data.append({"模型": m, "输入单价": p["input"], "输出单价": p["output"]})
        st.dataframe(price_data, use_container_width=True, hide_index=True)

        # ── Token 明细表 ──
        st.markdown("---")
        st.subheader(f"📋 Token 明细（最近 50 条，共 {len(tokens)} 条）")
        recent = tokens[-50:]
        display_rows = []
        for t in reversed(recent):
            m = t.get("model", "?")
            cost = calc_cost(m, t.get("prompt_tokens", 0), t.get("completion_tokens", 0))
            display_rows.append({
                "时间": t.get("ts", "")[:19].replace("T", " "),
                "调用方": t.get("caller", "?"),
                "模型": m,
                "Prompt": t.get("prompt_tokens", 0),
                "Completion": t.get("completion_tokens", 0),
                "Total": t.get("total_tokens", 0),
                "成本": f"${cost:.6f}",
            })
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
    else:
        st.info("暂无 Token 使用记录。Agent 运行后会自动写入 logs/token_usage.jsonl")


# ════════════════════════════════════════
# 模块 ③ Eval 评测中心
# ════════════════════════════════════════
def page_eval():
    st.title("🧪 Eval 评测中心")
    st.caption("运行 Eval 测试套件，查看用例和历史通过率")

    cases = load_jsonl(CASES_FILE)
    evals = load_jsonl(EVAL_LOG)

    # 概览
    col1, col2, col3, col4 = st.columns(4)
    sections = Counter(c.get("section", "未分类") for c in cases)
    with col1:
        st.metric("总用例数", len(cases))
    with col2:
        st.metric("防复发", sections.get("防复发", 0))
    with col3:
        st.metric("能力矩阵", sections.get("能力矩阵", 0))
    with col4:
        last_pass = f"{evals[-1].get('pass_rate', 0):.0%}" if evals else "—"
        st.metric("最近通过率", last_pass)

    # 历史通过率趋势
    if evals and len(evals) > 1:
        st.subheader("📈 通过率趋势")
        eval_dates = [e.get("ts", "")[:10] for e in evals[-14:]]
        eval_rates = [round(e.get("pass_rate", 0) * 100, 1) for e in evals[-14:]]
        st.line_chart({"日期": eval_dates, "通过率%": eval_rates}, x="日期", y="通过率%")

    st.markdown("---")

    # 运行 Eval
    if st.button("▶️ 运行 Eval", type="primary", use_container_width=True):
        with st.spinner("正在运行 Eval（预计 1~2 分钟）..."):
            eval_script = os.path.join(EVAL_DIR, "eval.py")
            result = subprocess.run(
                [sys.executable, eval_script],
                capture_output=True, text=True, cwd=BASE_DIR, timeout=180,
            )

        if result.returncode == 0:
            output = result.stdout
            st.success("✅ Eval 完成！")
            st.code(output, language=None)

            # 提取通过率并记录
            pass_rate = 0.0
            total_cases = 0
            for line in output.split("\n"):
                if "通过率" in line:
                    m = re.search(r"(\d+(?:\.\d+)?)%", line)
                    if m:
                        pass_rate = float(m.group(1)) / 100
                if "总用例" in line:
                    m2 = re.search(r"(\d+)", line)
                    if m2:
                        total_cases = int(m2.group(1))

            eval_entry = {
                "ts": datetime.now().isoformat(),
                "total": total_cases or len(cases),
                "pass_rate": pass_rate,
                "output": output[-500:],
            }
            evals = load_jsonl(EVAL_LOG)
            evals.append(eval_entry)
            save_jsonl(EVAL_LOG, evals)
        else:
            st.error("❌ Eval 运行失败")
            st.code(result.stderr, language=None)

    # 用例列表
    st.markdown("---")
    st.subheader(f"📋 用例列表（{len(cases)} 条）")
    for i, case in enumerate(cases):
        section = case.get("section", "")
        q = case.get("q", "")
        with st.expander(f"#{i+1} [{section}] {q[:50]}"):
            st.json(case)

    # 历史记录
    if evals:
        st.markdown("---")
        st.subheader(f"📋 历史记录（{len(evals)} 次）")
        for e in reversed(evals[-10:]):
            ts = e.get("ts", "")[:19].replace("T", " ")
            pr = e.get("pass_rate", 0)
            total = e.get("total", 0)
            color = "🟢" if pr >= 0.9 else "🟡" if pr >= 0.7 else "🔴"
            st.markdown(f"{color} `{ts}` — 通过率 **{pr:.0%}** ({total} 条用例)")


# ════════════════════════════════════════
# 模块 ④ 测评管理
# ════════════════════════════════════════
def page_eval_mgmt():
    st.title("🧪 测评管理")
    st.caption("Eval 中心 · 坏例闭环 · BUG 反馈")

    tab_eval, tab_badcase, tab_bug = st.tabs(["📝 Eval 中心", "🚩 坏例闭环", "🐛 BUG 反馈"])

    # ── Tab1: Eval 中心 ──
    with tab_eval:
        cases = load_jsonl(CASES_FILE)
        evals = load_jsonl(EVAL_LOG)

        # 概览
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总用例数", len(cases))
        with col2:
            last_pass = f"{evals[-1].get('pass_rate', 0):.0%}" if evals else "—"
            st.metric("最近通过率", last_pass)
        with col3:
            sections = Counter(c.get("section", "未分类") for c in cases)
            st.metric("防复发用例", sections.get("防复发", 0))

        st.markdown("---")

        # 新增用例
        with st.expander("➕ 新增用例"):
            new_q = st.text_input("问题", key="new_case_q")
            new_intent = st.text_input("期望意图", key="new_case_intent", placeholder="query×seller×ship_time")
            new_section = st.selectbox("分类", ["防复发", "能力矩阵", "坏例审核"], key="new_case_section")
            new_banned = st.text_input("禁词（逗号分隔）", key="new_case_banned")
            new_required = st.text_input("必含词（逗号分隔）", key="new_case_required")

            if st.button("✅ 添加用例", key="add_case"):
                if new_q and new_intent:
                    new_case = {
                        "section": new_section,
                        "q": new_q,
                        "expect_intent": new_intent,
                        "banned_answer_contains_intent": True,
                    }
                    if new_banned:
                        new_case["banned_words"] = [w.strip() for w in new_banned.split(",") if w.strip()]
                    if new_required:
                        new_case["required_words"] = [w.strip() for w in new_required.split(",") if w.strip()]
                    with open(CASES_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(new_case, ensure_ascii=False) + "\n")
                    st.success("✅ 用例已添加")
                    st.rerun()

        # 用例列表（可编辑/删除）
        st.subheader(f"📋 用例列表（{len(cases)} 条）")
        for i, case in enumerate(cases):
            section = case.get("section", "")
            q = case.get("q", "")
            intent = case.get("expect_intent", "")
            col_a, col_b, col_c = st.columns([4, 1, 1])
            with col_a:
                with st.expander(f"#{i+1} [{section}] {q[:50]}"):
                    st.json(case)
            with col_b:
                if st.button("✏️", key=f"edit_case_{i}"):
                    st.session_state[f"editing_case_{i}"] = True
            with col_c:
                if st.button("🗑️", key=f"del_case_{i}"):
                    cases.pop(i)
                    save_jsonl(CASES_FILE, cases)
                    st.rerun()

            # 编辑模式
            if st.session_state.get(f"editing_case_{i}", False):
                with st.form(f"edit_form_{i}"):
                    edit_q = st.text_input("问题", value=case.get("q", ""))
                    edit_intent = st.text_input("期望意图", value=case.get("expect_intent", ""))
                    edit_section = st.text_input("分类", value=case.get("section", ""))
                    edit_banned = st.text_input("禁词", value=",".join(case.get("banned_words", [])))
                    edit_required = st.text_input("必含词", value=",".join(case.get("required_words", [])))
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.form_submit_button("💾 保存"):
                            cases[i]["q"] = edit_q
                            cases[i]["expect_intent"] = edit_intent
                            cases[i]["section"] = edit_section
                            if edit_banned:
                                cases[i]["banned_words"] = [w.strip() for w in edit_banned.split(",")]
                            if edit_required:
                                cases[i]["required_words"] = [w.strip() for w in edit_required.split(",")]
                            save_jsonl(CASES_FILE, cases)
                            st.session_state[f"editing_case_{i}"] = False
                            st.rerun()
                    with col_cancel:
                        if st.form_submit_button("取消"):
                            st.session_state[f"editing_case_{i}"] = False
                            st.rerun()

        # 运行 Eval
        st.markdown("---")
        if st.button("▶️ 运行 Eval", type="primary", use_container_width=True):
            with st.spinner("正在运行 Eval（预计 1~2 分钟）..."):
                eval_script = os.path.join(EVAL_DIR, "eval.py")
                result = subprocess.run(
                    [sys.executable, eval_script],
                    capture_output=True, text=True, cwd=BASE_DIR, timeout=180,
                )
            if result.returncode == 0:
                output = result.stdout
                st.success("✅ Eval 完成！")
                st.code(output, language=None)
                # 提取通过率
                pass_rate = 0.0
                for line in output.split("\n"):
                    if "通过率" in line:
                        m = re.search(r"(\d+(?:\.\d+)?)%", line)
                        if m:
                            pass_rate = float(m.group(1)) / 100
                eval_entry = {"ts": datetime.now().isoformat(), "total": len(cases), "pass_rate": pass_rate, "output": output[-500:]}
                evals.append(eval_entry)
                save_jsonl(EVAL_LOG, evals)
            else:
                st.error("❌ Eval 运行失败")
                st.code(result.stderr, language=None)

    # ── Tab2: 坏例闭环 ──
    with tab_badcase:
        convs = load_jsonl(CONV_LOG)
        evaluations = load_jsonl(EVALUATIONS_LOG)

        if not convs:
            st.info("暂无对话记录。使用助手产生对话后，这里会显示最近的对话。")
        else:
            st.caption(f"共 {len(convs)} 条对话记录。标记坏例会同时生成 Eval 用例 + BUG 反馈单。")

            display = convs[-50:]
            start_idx = len(convs) - len(display)

            for i, conv in enumerate(display):
                idx = start_idx + i
                ts = conv.get("ts", "")[:19].replace("T", " ")
                q = conv.get("question", "")
                intent = conv.get("intent", "")
                answer = conv.get("answer", "")[:200]

                with st.expander(f"[{ts}] {q[:60]}"):
                    st.markdown(f"**意图：** `{intent}`")
                    st.markdown(f"**回答：** {answer}")

                    # 关联评审记录
                    matched_eval = None
                    for ev in evaluations:
                        if ev.get("question", "") == q:
                            matched_eval = ev
                            break
                    if matched_eval:
                        fw = matched_eval.get("framework", {})
                        mj = matched_eval.get("model_judge", {})
                        st.markdown(f"**评审：** 框架 {'✅' if fw.get('pass') else '❌'} | 裁判评分 **{mj.get('overall', '?')}**/10 — {mj.get('comment', '')}")

                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        if st.button("🚩 标记为坏例", key=f"bad_{idx}"):
                            # 1. 生成 Eval 用例
                            bad_case = {
                                "section": "坏例审核",
                                "q": q,
                                "expect_intent": intent,
                                "banned_answer_contains_intent": True,
                                "note": f"从对话日志标记 @ {ts}",
                            }
                            with open(CASES_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(bad_case, ensure_ascii=False) + "\n")

                            # 2. 生成 BUG 反馈单
                            bug_entry = {
                                "ts": datetime.now().isoformat(),
                                "question": q,
                                "intent": intent,
                                "answer": conv.get("answer", "")[:500],
                                "status": "待修复",
                                "note": f"从对话日志标记 @ {ts}",
                            }
                            with open(BUG_FEEDBACK_LOG, "a", encoding="utf-8") as f:
                                f.write(json.dumps(bug_entry, ensure_ascii=False) + "\n")

                            st.success(f"✅ 已追加到 cases.jsonl + bug_feedback.jsonl")

                    with col_b:
                        if st.button("🗑️ 删除该条", key=f"del_{idx}"):
                            convs.pop(idx)
                            save_jsonl(CONV_LOG, convs)
                            st.rerun()

    # ── Tab3: BUG 反馈 ──
    with tab_bug:
        bugs = load_jsonl(BUG_FEEDBACK_LOG)

        if not bugs:
            st.info("暂无 BUG 反馈。在「坏例闭环」中标记坏例后，会自动生成 BUG 反馈单。")
        else:
            # 统计
            pending = sum(1 for b in bugs if b.get("status") == "待修复")
            fixed = sum(1 for b in bugs if b.get("status") == "已修复")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总反馈", len(bugs))
            with col2:
                st.metric("待修复", pending)
            with col3:
                st.metric("已修复", fixed)

            st.markdown("---")

            # 筛选
            filter_status = st.selectbox("筛选状态", ["全部", "待修复", "已修复"], key="bug_filter")
            filtered = bugs if filter_status == "全部" else [b for b in bugs if b.get("status") == filter_status]

            for i, bug in enumerate(filtered):
                ts = bug.get("ts", "")[:19].replace("T", " ")
                q = bug.get("question", "")
                status = bug.get("status", "待修复")
                icon = "🔴" if status == "待修复" else "🟢"

                with st.expander(f"{icon} [{ts}] {q[:60]}"):
                    st.markdown(f"**问题：** {q}")
                    st.markdown(f"**意图：** `{bug.get('intent', '')}`")
                    st.markdown(f"**回答：** {bug.get('answer', '')[:200]}")
                    st.markdown(f"**状态：** {status}")

                    if status == "待修复":
                        # 找到在原始列表中的索引
                        real_idx = bugs.index(bug)
                        if st.button("✅ 标记已修复", key=f"fix_{real_idx}"):
                            bugs[real_idx]["status"] = "已修复"
                            bugs[real_idx]["fixed_at"] = datetime.now().isoformat()
                            save_jsonl(BUG_FEEDBACK_LOG, bugs)
                            st.rerun()


# ════════════════════════════════════════
# 模块 ⑤ Agent 管理
# ════════════════════════════════════════
def page_agent():
    st.title("⚙️ Agent 管理")
    st.caption("干预开关 · 提示词调优 · 版本回滚 · AB 测试")

    tab_filters, tab_prompts = st.tabs(["🛡️ 干预开关", "📝 提示词调优"])

    # ── 干预开关 ──
    with tab_filters:
        filters = load_yaml(FILTERS_FILE)
        if not filters:
            filters = {"disabled_combinations": [], "disabled_states": [], "disabled_categories": []}

        # 确保三个 key 都存在
        disabled_combinations = filters.get("disabled_combinations", [])
        disabled_states = filters.get("disabled_states", [])
        disabled_categories = filters.get("disabled_categories", [])

        col1, col2, col3 = st.columns(3)

        # ── 禁用组合 ──
        with col1:
            st.subheader("🚫 禁用组合")
            st.caption("格式：operation×dimension×metric")
            new_combo = st.text_input("新增禁用组合", placeholder="compare×seller×price", key="new_combo")
            if st.button("➕ 添加组合", key="add_combo"):
                if new_combo and new_combo not in disabled_combinations:
                    disabled_combinations.append(new_combo)
                    filters["disabled_combinations"] = disabled_combinations
                    save_yaml(FILTERS_FILE, filters)
                    reload_agent_config()
                    st.rerun()

            for i, combo in enumerate(disabled_combinations):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.code(combo)
                with col_b:
                    if st.button("❌", key=f"del_combo_{i}"):
                        disabled_combinations.pop(i)
                        filters["disabled_combinations"] = disabled_combinations
                        save_yaml(FILTERS_FILE, filters)
                        reload_agent_config()
                        st.rerun()

        # ── 禁用州 ──
        with col2:
            st.subheader("🚫 禁用州")
            st.caption("巴西 2 字母大写州代码")
            new_state = st.text_input("新增禁用州", placeholder="AC", key="new_state", max_chars=2)
            if st.button("➕ 添加州", key="add_state"):
                state_upper = new_state.upper().strip()
                if state_upper and state_upper not in disabled_states:
                    disabled_states.append(state_upper)
                    filters["disabled_states"] = disabled_states
                    save_yaml(FILTERS_FILE, filters)
                    reload_agent_config()
                    st.rerun()

            for i, state in enumerate(disabled_states):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.code(state)
                with col_b:
                    if st.button("❌", key=f"del_state_{i}"):
                        disabled_states.pop(i)
                        filters["disabled_states"] = disabled_states
                        save_yaml(FILTERS_FILE, filters)
                        reload_agent_config()
                        st.rerun()

        # ── 禁用品类 ──
        with col3:
            st.subheader("🚫 禁用品类")
            st.caption("品类名称（中文或英文）")
            new_cat = st.text_input("新增禁用品类", placeholder="二手商品", key="new_cat")
            if st.button("➕ 添加品类", key="add_cat"):
                if new_cat and new_cat not in disabled_categories:
                    disabled_categories.append(new_cat)
                    filters["disabled_categories"] = disabled_categories
                    save_yaml(FILTERS_FILE, filters)
                    reload_agent_config()
                    st.rerun()

            for i, cat in enumerate(disabled_categories):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.code(cat)
                with col_b:
                    if st.button("❌", key=f"del_cat_{i}"):
                        disabled_categories.pop(i)
                        filters["disabled_categories"] = disabled_categories
                        save_yaml(FILTERS_FILE, filters)
                        reload_agent_config()
                        st.rerun()

        st.markdown("---")
        st.subheader("🔍 拦截预览")
        preview_combo = st.text_input("输入三元组预览拦截效果", placeholder="compare×seller×price", key="preview_combo")
        if preview_combo:
            if preview_combo in disabled_combinations:
                st.error(f"🚫 「{preview_combo}」已被禁用，前端会返回拦截引导")
            else:
                st.success(f"✅ 「{preview_combo}」未被禁用，正常查询")

    # ── 提示词调优 ──
    with tab_prompts:
        prompts = load_yaml(PROMPTS_FILE)
        versions = load_jsonl(PROMPT_VERSIONS_FILE)

        # 提示词编辑
        st.subheader("✏️ 编辑提示词")
        prompt_keys = [k for k in prompts.keys() if k != "safety"]
        selected_key = st.selectbox("选择提示词", prompt_keys, key="select_prompt_key")

        if selected_key:
            current_content = prompts.get(selected_key, "")
            edited = st.text_area(
                f"编辑 {selected_key}",
                value=current_content,
                height=300,
                key=f"edit_{selected_key}",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💾 保存提示词", type="primary", use_container_width=True):
                    # 保存到 prompts.yaml
                    prompts[selected_key] = edited
                    save_yaml(PROMPTS_FILE, prompts)

                    # 记录版本快照
                    version_entry = {
                        "ts": datetime.now().isoformat(),
                        "key": selected_key,
                        "content": edited,
                        "version": len([v for v in versions if v.get("key") == selected_key]) + 1,
                    }
                    versions.append(version_entry)
                    save_jsonl(PROMPT_VERSIONS_FILE, versions)

                    reload_agent_config()
                    st.success(f"✅ 已保存 {selected_key}（版本 {version_entry['version']}）")
                    st.rerun()

            with col_b:
                if st.button("🔄 回滚到历史版本", use_container_width=True):
                    st.session_state["show_rollback"] = True

        # 版本历史与回滚
        if st.session_state.get("show_rollback", False):
            st.markdown("---")
            st.subheader("📜 版本历史")
            key_versions = [v for v in versions if v.get("key") == selected_key]
            if key_versions:
                for i, v in enumerate(reversed(key_versions[-10:])):
                    ts = v.get("ts", "")[:19].replace("T", " ")
                    ver = v.get("version", "?")
                    with st.expander(f"版本 {ver} — {ts}"):
                        st.code(v.get("content", "")[:500])
                        if st.button(f"↩️ 回滚到此版本", key=f"rollback_{i}"):
                            prompts[selected_key] = v.get("content", "")
                            save_yaml(PROMPTS_FILE, prompts)
                            reload_agent_config()
                            st.success(f"✅ 已回滚到版本 {ver}")
                            st.rerun()
            else:
                st.info("暂无历史版本")

        # ── AB 测试 ──
        st.markdown("---")
        st.subheader("🧪 AB 测试")
        st.caption("选某提示词 → 填 B 版内容 → 跑两版 Eval → 对比通过率 → 选胜者")

        ab_key = st.selectbox("选择提示词进行 AB 测试", prompt_keys, key="ab_key")
        if ab_key:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**A 版（当前）**")
                a_content = prompts.get(ab_key, "")
                st.code(a_content[:300])
                if st.button("▶️ 用 A 版跑 Eval", key="run_eval_a"):
                    with st.spinner("运行 A 版 Eval..."):
                        # A 版已是当前，直接跑
                        eval_script = os.path.join(EVAL_DIR, "eval.py")
                        result = subprocess.run(
                            [sys.executable, eval_script],
                            capture_output=True, text=True, cwd=BASE_DIR, timeout=180,
                        )
                        if result.returncode == 0:
                            output = result.stdout
                            pass_rate_a = 0.0
                            for line in output.split("\n"):
                                if "通过率" in line:
                                    m = re.search(r"(\d+(?:\.\d+)?)%", line)
                                    if m:
                                        pass_rate_a = float(m.group(1)) / 100
                            st.session_state["ab_a_rate"] = pass_rate_a
                            st.success(f"A 版通过率: {pass_rate_a:.0%}")
                        else:
                            st.error("A 版 Eval 运行失败")

            with col_b:
                st.markdown("**B 版（候选）**")
                b_content = st.text_area(
                    f"输入 B 版 {ab_key} 内容",
                    value=st.session_state.get(f"ab_b_{ab_key}", ""),
                    height=150,
                    key=f"ab_b_{ab_key}",
                )
                if st.button("▶️ 用 B 版跑 Eval", key="run_eval_b"):
                    if not b_content.strip():
                        st.warning("请先输入 B 版内容")
                    else:
                        with st.spinner("运行 B 版 Eval..."):
                            # 临时替换为 B 版
                            prompts_b = dict(prompts)
                            prompts_b[ab_key] = b_content
                            save_yaml(PROMPTS_FILE, prompts_b)
                            reload_agent_config()

                            eval_script = os.path.join(EVAL_DIR, "eval.py")
                            result = subprocess.run(
                                [sys.executable, eval_script],
                                capture_output=True, text=True, cwd=BASE_DIR, timeout=180,
                            )

                            # 恢复 A 版
                            save_yaml(PROMPTS_FILE, prompts)
                            reload_agent_config()

                            if result.returncode == 0:
                                output = result.stdout
                                pass_rate_b = 0.0
                                for line in output.split("\n"):
                                    if "通过率" in line:
                                        m = re.search(r"(\d+(?:\.\d+)?)%", line)
                                        if m:
                                            pass_rate_b = float(m.group(1)) / 100
                                st.session_state["ab_b_rate"] = pass_rate_b
                                st.success(f"B 版通过率: {pass_rate_b:.0%}")
                            else:
                                st.error("B 版 Eval 运行失败")

        # AB 结果对比
        a_rate = st.session_state.get("ab_a_rate")
        b_rate = st.session_state.get("ab_b_rate")
        if a_rate is not None and b_rate is not None:
            st.markdown("---")
            st.subheader("📊 AB 对比结果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("A 版通过率", f"{a_rate:.0%}")
            with col2:
                st.metric("B 版通过率", f"{b_rate:.0%}")
            with col3:
                if b_rate > a_rate:
                    st.success(f"🏆 B 版胜出 (+{(b_rate-a_rate)*100:.1f}%)")
                    if st.button("✅ 采用 B 版", type="primary"):
                        prompts[ab_key] = st.session_state.get(f"ab_b_{ab_key}", "")
                        save_yaml(PROMPTS_FILE, prompts)
                        reload_agent_config()
                        st.success(f"✅ 已将 B 版设为 {ab_key} 的当前版本")
                        st.rerun()
                elif a_rate > b_rate:
                    st.info(f"🏆 A 版胜出 (+{(a_rate-b_rate)*100:.1f}%)")
                else:
                    st.warning("平局")


# ── 路由 ──
if page == "① 指标看板":
    page_dashboard()
elif page == "② 模型管理":
    page_model()
elif page == "③ Eval 评测中心":
    page_eval()
elif page == "④ 测评管理":
    page_eval_mgmt()
elif page == "⑤ Agent 管理":
    page_agent()
