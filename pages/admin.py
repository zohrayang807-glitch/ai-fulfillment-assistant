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
import pandas as _pd
import altair as _alt
from datetime import datetime, timedelta
from collections import Counter

st.set_page_config(page_title="运营后台 · V2.0", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # pages/ 的上一级 = 项目根

# ── 数据库访问层 ──
sys.path.insert(0, BASE_DIR)
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))
import db
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
/* 隐藏默认页面导航 */
[data-testid="stSidebarNav"] { display: none !important; }

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
/* 图表 x 轴标签横排/斜排，避免竖排 */
[data-testid="stChart"] g[aria-label*="axis"] text,
[data-testid="stVegaLiteChart"] text[aria-label*="axis"] {
    writing-mode: horizontal-tb !important;
}
</style>
""", unsafe_allow_html=True)


# ── 工具函数 ──
def _eval_rate(e: dict) -> float:
    """从评审记录中取 Eval 通过率（兼容 overall / scores.pass_rate / pass_rate）"""
    v = e.get("overall")
    if v is None:
        v = (e.get("scores") or {}).get("pass_rate")
    if v is None:
        v = e.get("pass_rate", 0)
    return v or 0


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

    # 操作者昵称（多人协作用）
    if "admin_nickname" not in st.session_state:
        st.session_state.admin_nickname = "admin"
    admin_nick = st.text_input(
        "👤 操作者昵称",
        value=st.session_state.admin_nickname,
        max_chars=20,
        key="admin_nick_input",
    )
    if admin_nick != st.session_state.admin_nickname:
        st.session_state.admin_nickname = admin_nick

    st.markdown("---")

PAGES = {
    "指标看板": "dashboard",
    "模型管理": "model",
    "测评管理": "eval_mgmt",
    "Agent 管理": "agent",
}

page = st.sidebar.radio("导航", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
# 多页模式下 switch_page 无法切回主文件，用链接跳转（兼容本地 + Streamlit Cloud）
st.sidebar.markdown(
    '<a href="/" target="_self" style="text-decoration:none;font-weight:600">← 返回助手</a>',
    unsafe_allow_html=True,
)


# ════════════════════════════════════════
# 模块：指标看板
# ════════════════════════════════════════
def page_dashboard():
    st.title("📈 指标看板")
    st.caption("Agent 层 + 模型层合一的运营视图")

    convs = db.get_conversations(limit=1000)
    tokens = db.get_token_usage()
    evals = db.get_evaluations(limit=200)

    # ── KPI 卡片 ──
    total_convs = len(convs)
    total_calls = len(tokens)
    total_prompt = sum(t.get("prompt_tokens", 0) for t in tokens)
    total_completion = sum(t.get("completion_tokens", 0) for t in tokens)
    total_tokens = sum(t.get("total_tokens", 0) for t in tokens)
    total_cost = sum(calc_cost(t.get("model", ""), t.get("prompt_tokens", 0), t.get("completion_tokens", 0)) for t in tokens)

    # 找最新的 Eval run 记录（对话评审不算）
    _eval_runs = [e for e in evals if "Eval run" in str(e.get("question", ""))]
    last_eval_pass = "—"
    if _eval_runs:
        _last = _eval_runs[0]
        _rate = _last.get("overall") or (_last.get("scores") or {}).get("pass_rate") or 0
        last_eval_pass = f"{_rate:.0%}"

    # 统计坏例数
    cases = db.get_cases()
    bad_count = sum(1 for c in cases if c.get("section") == "坏例审核")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{total_convs}</div><div class="kpi-label">总对话数</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{total_calls}</div><div class="kpi-label">模型调用数</div></div>""", unsafe_allow_html=True)
    with col3:
        _tk_display = f"{total_tokens/1000:.1f}" if total_tokens >= 1000 else str(total_tokens)
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{_tk_display}</div><div class="kpi-label">总 Token（K）</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">${total_cost:.2f}</div><div class="kpi-label">预估成本</div></div>""", unsafe_allow_html=True)
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
            today_dt = datetime.now()
            # 生成近7天 "MM-DD" 格式的日期键
            date_keys = [(today_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
            for t in tokens:
                d = t.get("ts", "")[:10]  # YYYY-MM-DD
                if d:
                    date_counts[d] += 1
            dates = [(today_dt - timedelta(days=i)).strftime("%m-%d") for i in range(6, -1, -1)]
            counts = [date_counts.get(dk, 0) for dk in date_keys]
            _df = {"日期": dates, "调用数": counts}
            st.altair_chart(
                _alt.Chart(_pd.DataFrame(_df)).mark_bar().encode(
                    x=_alt.X("日期:N", axis=_alt.Axis(labelAngle=-45, labelLimit=60)),
                    y="调用数:Q",
                ).properties(height=300),
                use_container_width=True,
            )
        else:
            st.info("暂无调用记录")

    with col_b:
        st.subheader("🎯 Eval 通过率走势")
        # 只取 Eval run 记录（对话评审不算），避免 0% 和 100% 交替跳变
        _evals_only = [e for e in evals if "Eval run" in str(e.get("question", ""))]
        if _evals_only:
            eval_dates = [str(e.get("ts", ""))[5:10] for e in _evals_only[:14]]
            eval_rates = [round(_eval_rate(e) * 100, 1) for e in _evals_only[:14]]
            _df = {"日期": eval_dates, "通过率%": eval_rates}
            st.altair_chart(
                _alt.Chart(_pd.DataFrame(_df)).mark_line(point=True).encode(
                    x=_alt.X("日期:N", axis=_alt.Axis(labelAngle=-45, labelLimit=60)),
                    y="通过率%:Q",
                ).properties(height=300),
                use_container_width=True,
            )
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
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.markdown("**操作 (operation)**")
            if op_counter:
                _d = dict(op_counter.most_common())
                _dd = _pd.DataFrame({"val": list(_d.values())}, index=list(_d.keys())).reset_index()
                _dd.columns = ["label", "val"]
                st.altair_chart(
                    _alt.Chart(_dd).mark_bar().encode(
                        x=_alt.X("label:N", axis=_alt.Axis(labelAngle=-45, labelLimit=60)),
                        y="val:Q",
                    ).properties(height=260),
                    use_container_width=True,
                )
        with col_y:
            st.markdown("**维度 (dimension)**")
            if dim_counter:
                _d = dict(dim_counter.most_common())
                _dd = _pd.DataFrame({"val": list(_d.values())}, index=list(_d.keys())).reset_index()
                _dd.columns = ["label", "val"]
                st.altair_chart(
                    _alt.Chart(_dd).mark_bar().encode(
                        x=_alt.X("label:N", axis=_alt.Axis(labelAngle=-45, labelLimit=60)),
                        y="val:Q",
                    ).properties(height=260),
                    use_container_width=True,
                )
        with col_z:
            st.markdown("**指标 (metric)**")
            if metric_counter:
                _d = dict(metric_counter.most_common())
                _dd = _pd.DataFrame({"val": list(_d.values())}, index=list(_d.keys())).reset_index()
                _dd.columns = ["label", "val"]
                st.altair_chart(
                    _alt.Chart(_dd).mark_bar().encode(
                        x=_alt.X("label:N", axis=_alt.Axis(labelAngle=-45, labelLimit=60)),
                        y="val:Q",
                    ).properties(height=260),
                    use_container_width=True,
                )

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
# 模块：模型管理
# ════════════════════════════════════════
def page_model():
    st.title("🤖 模型管理")
    st.caption("管理主模型、裁判模型、参数配置，保存后 Agent 下次对话自动生效")

    cfg = load_yaml(MODEL_FILE)
    tokens = db.get_token_usage()

    # ── 当前配置预览 ──
    st.subheader("📋 当前配置")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**主模型**  \n<code style='font-size:0.9rem'>{cfg.get('model', 'deepseek-v4-flash')}</code>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**裁判模型**  \n<code style='font-size:0.9rem'>{cfg.get('judge_model', 'deepseek-v4-pro')}</code>", unsafe_allow_html=True)
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
# 模块：测评管理（含 Eval 中心 + 坏例闭环 + BUG 反馈）
# ════════════════════════════════════════
def page_eval_mgmt():
    st.title("🧪 测评管理")
    st.caption("Eval 中心 · 坏例闭环 · BUG 反馈")

    tab_eval, tab_badcase, tab_bug = st.tabs(["📝 Eval 中心", "🚩 坏例闭环", "🐛 BUG 反馈"])

    # ── Tab1: Eval 中心 ──
    with tab_eval:
        cases = db.get_cases()
        evals = db.get_evaluations()

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
            last_pass = f"{_eval_rate(evals[-1]):.0%}" if evals else "—"
            st.metric("最近通过率", last_pass)

        # 通过率趋势（只取 Eval run，和指标看板统一）
        _trend_evals = [e for e in evals if "Eval run" in str(e.get("question", ""))]
        if _trend_evals and len(_trend_evals) > 1:
            st.subheader("📈 通过率趋势")
            eval_dates = [str(e.get("ts", ""))[5:10] for e in _trend_evals[:14]]
            eval_rates = [round(_eval_rate(e) * 100, 1) for e in _trend_evals[:14]]
            _df = {"日期": eval_dates, "通过率%": eval_rates}
            st.altair_chart(
                _alt.Chart(_pd.DataFrame(_df)).mark_line(point=True).encode(
                    x=_alt.X("日期:N", axis=_alt.Axis(labelAngle=-45, labelLimit=60)),
                    y="通过率%:Q",
                ).properties(height=300),
                use_container_width=True,
            )

        st.markdown("---")

        # 运行 Eval
        # Eval 异步后台跑：点按钮启动后台进程，不阻塞 Streamlit 请求（避免云端超时）
        _EVAL_STATUS_FILE = os.path.join(LOG_DIR, "eval_status.json")

        # 检查是否有正在运行的 Eval
        _eval_running = False
        _eval_status = {}
        if os.path.exists(_EVAL_STATUS_FILE):
            try:
                with open(_EVAL_STATUS_FILE, encoding="utf-8") as f:
                    _eval_status = json.load(f)
                if _eval_status.get("running"):
                    _eval_running = True
            except Exception:
                pass

        if st.button("▶️ 运行 Eval", type="primary", use_container_width=True):
            if _eval_running:
                st.warning("⏳ 已有 Eval 正在运行，请等待完成。")
            else:
                # 写"运行中"状态，后台启动
                import threading
                with open(_EVAL_STATUS_FILE, "w", encoding="utf-8") as f:
                    json.dump({"running": True, "ts": datetime.now().isoformat(), "output": ""}, f, ensure_ascii=False)
                eval_script = os.path.join(EVAL_DIR, "eval.py")

                def _run_eval_bg():
                    try:
                        _r = subprocess.run(
                            [sys.executable, eval_script],
                            capture_output=True, text=True, cwd=BASE_DIR, timeout=900,
                        )
                        with open(_EVAL_STATUS_FILE, "w", encoding="utf-8") as f:
                            json.dump({"running": False, "ts": datetime.now().isoformat(),
                                       "returncode": _r.returncode, "output": _r.stdout, "error": _r.stderr}, f, ensure_ascii=False)
                    except Exception as _e:
                        with open(_EVAL_STATUS_FILE, "w", encoding="utf-8") as f:
                            json.dump({"running": False, "ts": datetime.now().isoformat(), "output": "", "error": str(_e)}, f, ensure_ascii=False)

                threading.Thread(target=_run_eval_bg, daemon=True).start()
                st.success("✅ Eval 已后台启动，跑完自动更新结果（约 1-2 分钟）。可刷新页面查看。")
                st.rerun()

        # 显示 Eval 状态/结果
        if _eval_status:
            if _eval_status.get("running"):
                st.info("⏳ Eval 正在后台运行，请稍后刷新查看结果...")
            else:
                _output = _eval_status.get("output", "")
                _error = _eval_status.get("error", "")
                if _output:
                    st.success("✅ Eval 完成！")
                    st.code(_output, language=None)
                elif _error:
                    st.error(f"❌ Eval 运行失败：{_error[:200]}")
                else:
                    st.warning("Eval 无输出。")

            # 有结果时解析并记录
            if not _eval_status.get("running") and _eval_status.get("output"):
                output = _eval_status.get("output", "")

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
                db.insert_evaluation(
                    ts=eval_entry["ts"],
                    question=f"Eval run @ {eval_entry['ts'][:19]}",
                    answer=output[-500:],
                    scores={"total": eval_entry["total"], "pass_rate": eval_entry["pass_rate"]},
                    overall=eval_entry["pass_rate"],
                    comment=f"Eval 通过率 {eval_entry['pass_rate']:.0%}",
                    user=st.session_state.get("admin_nickname", "admin"),
                )
                evals = db.get_evaluations()
            else:
                st.error("❌ Eval 运行失败")
                st.code(result.stderr, language=None)

        st.markdown("---")

        # 新增用例
        with st.expander("➕ 新增用例"):
            new_q = st.text_input("问题", key="new_case_q")
            new_intent = st.text_input("期望意图", key="new_case_intent", placeholder="query×seller×ship_time")
            new_section = st.selectbox("分类", ["防复发", "能力矩阵", "坏例审核"], key="new_case_section")
            new_user = st.text_input("创建者", value="我", max_chars=20, key="new_case_user")
            new_banned = st.text_input("禁词（逗号分隔）", key="new_case_banned")
            new_required = st.text_input("必含词（逗号分隔）", key="new_case_required")

            if st.button("✅ 添加用例", key="add_case"):
                if new_q and new_intent:
                    new_case = {
                        "section": new_section,
                        "q": new_q,
                        "expected_intents": [new_intent],
                        "banned_answer_contains_intent": True,
                        "user": new_user,
                    }
                    if new_banned:
                        new_case["banned"] = [w.strip() for w in new_banned.split(",") if w.strip()]
                    if new_required:
                        new_case["required"] = [w.strip() for w in new_required.split(",") if w.strip()]
                    db.add_case(new_case)
                    st.success("✅ 用例已添加")
                    st.rerun()

        # 用例列表（可编辑/删除）
        st.subheader(f"📋 用例列表（{len(cases)} 条）")
        # 用户筛选
        all_users = sorted(set(c.get("user", "我") for c in cases))
        selected_user = st.selectbox("按用户筛选", ["全部"] + all_users, key="cases_user_filter")
        if selected_user != "全部":
            cases = [c for c in cases if c.get("user", "我") == selected_user]

        for i, case in enumerate(cases):
            section = case.get("section", "")
            q = case.get("q", "")
            intent = case.get("expect_intent", "")
            user = case.get("user", "我")
            is_multi_turn = bool(case.get("turns"))
            label = "多轮对话用例" if is_multi_turn else q[:50]
            col_a, col_spacer, col_b, col_c = st.columns([6, 2, 1, 1])
            with col_a:
                with st.expander(f"#{i+1} [{section}] 👤{user} | {label}"):
                    st.json(case)
            with col_b:
                if st.button("✏️", key=f"edit_case_{i}"):
                    st.session_state[f"editing_case_{i}"] = True
            with col_c:
                if st.button("🗑️", key=f"del_case_{i}"):
                    st.session_state[f"confirm_del_case_{i}"] = True
            # 确认删除弹窗
            if st.session_state.get(f"confirm_del_case_{i}", False):
                st.warning(f"确认删除用例「{q[:40]}」？")
                c_yes, c_no = st.columns(2)
                with c_yes:
                    if st.button("✅ 确认删除", key=f"confirm_yes_case_{i}"):
                        case_id = case.get("id")
                        if case_id:
                            db.delete_case(case_id)
                        st.session_state.pop(f"confirm_del_case_{i}", None)
                        st.rerun()
                with c_no:
                    if st.button("❌ 取消", key=f"confirm_no_case_{i}"):
                        st.session_state.pop(f"confirm_del_case_{i}", None)
                        st.rerun()

            # 编辑模式
            if st.session_state.get(f"editing_case_{i}", False):
                with st.form(f"edit_form_{i}"):
                    edit_q = st.text_input("问题", value=case.get("q", ""))
                    edit_intent = st.text_input("期望意图", value=case.get("expected_intents", [""])[0] if case.get("expected_intents") else "")
                    edit_section = st.text_input("分类", value=case.get("section", ""))
                    _ban = case.get("banned") or []
                    _req = case.get("required") or []
                    _ban = _ban if isinstance(_ban, list) else []
                    _req = _req if isinstance(_req, list) else []
                    edit_banned = st.text_input("禁词", value=",".join(_ban))
                    edit_required = st.text_input("必含词", value=",".join(_req))
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.form_submit_button("💾 保存"):
                            updates = {
                                "q": edit_q,
                                "expected_intents": [edit_intent],
                                "section": edit_section,
                            }
                            if edit_banned:
                                updates["banned"] = [w.strip() for w in edit_banned.split(",")]
                            if edit_required:
                                updates["required"] = [w.strip() for w in edit_required.split(",")]
                            case_id = case.get("id")
                            if case_id:
                                db.update_case(case_id, updates)
                            st.session_state[f"editing_case_{i}"] = False
                            st.rerun()
                    with col_cancel:
                        if st.form_submit_button("取消"):
                            st.session_state[f"editing_case_{i}"] = False
                            st.rerun()

        # 历史记录（只显示 Eval run，过滤对话评审）
        eval_runs = [e for e in evals if "Eval run" in str(e.get("question", ""))]
        if eval_runs:
            st.markdown("---")
            st.subheader(f"📋 Eval 历史记录（{len(eval_runs)} 次）")
            for e in eval_runs[:10]:
                ts = e.get("ts", "")[:19].replace("T", " ")
                pr = _eval_rate(e)
                total = e.get("total") or (e.get("scores") or {}).get("total") or 0
                color = "🟢" if pr >= 0.9 else "🟡" if pr >= 0.7 else "🔴"
                st.markdown(f"{color} `{ts}` — 通过率 **{pr:.0%}** ({total} 条用例)")

    # ── Tab2: 坏例闭环 ──
    with tab_badcase:
        convs = db.get_conversations(limit=200)
        convs.sort(key=lambda x: x.get("ts", ""), reverse=True)
        evaluations = db.get_evaluations()

        if not convs:
            st.info("暂无对话记录。使用助手产生对话后，这里会显示最近的对话。")
        else:
            # 用户筛选
            all_users = sorted(set(c.get("user", "我") for c in convs))
            selected_user = st.selectbox("按用户筛选", ["全部"] + all_users, key="badcase_user_filter")
            if selected_user != "全部":
                convs = [c for c in convs if c.get("user", "我") == selected_user]

            st.caption(f"共 {len(convs)} 条对话记录。标记坏例会同时生成 Eval 用例 + BUG 反馈单。")

            display = convs[:50]
            start_idx = 0

            for i, conv in enumerate(display):
                idx = start_idx + i
                ts = conv.get("ts", "")[:19].replace("T", " ")
                q = conv.get("question", "")
                intent = conv.get("intent", "")
                answer = conv.get("answer", "")[:200]
                user = conv.get("user", "我")

                with st.expander(f"[{ts}] 👤{user} | {q[:60]}"):
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

                    # 去重：检查是否已在 cases 中
                    existing_cases = db.get_cases()
                    already_marked = any(c.get("q") == q and c.get("section") == "坏例审核" for c in existing_cases)
                    if already_marked:
                        st.caption("⚠️ 该坏例已存在")
                    else:
                        reason = st.text_input("为什么是坏例（必填）", key=f"reason_{idx}", placeholder="描述问题...")
                        # 期望意图：默认填入当前识别意图，可修正（Eval 会用这个意图测试）
                        exp_intent = st.text_input("期望意图（Eval 测试用）", key=f"exp_intent_{idx}", value=intent or "", placeholder="如 query×seller×ship_time")
                        col_mark, col_del = st.columns([1, 1])
                        with col_mark:
                            if st.button("🚩 标记为坏例", key=f"bad_{idx}", disabled=not (reason.strip() and exp_intent.strip()), use_container_width=True):
                                conv_user = conv.get("user", "我")
                                # 1. 生成 Eval 用例（用填写的期望意图）
                                bad_case = {
                                    "section": "坏例审核",
                                    "q": q,
                                    "expected_intents": [exp_intent.strip()],
                                    "banned_answer_contains_intent": True,
                                    "user": conv_user,
                                    "reason": reason.strip(),
                                    "note": f"从对话日志标记 @ {ts}，原因：{reason.strip()}",
                                }
                                db.add_case(bad_case)

                                # 2. 生成 BUG 反馈单（intent 用用户填的期望意图，并带原因）
                                db.insert_bug(
                                    user=conv_user,
                                    question=q,
                                    intent=exp_intent.strip(),
                                    answer=conv.get("answer", "")[:500],
                                    status="待修复",
                                    reason=reason.strip(),
                                    note=f"从对话日志标记 @ {ts}，原因：{reason.strip()}",
                                )

                                st.success("✅ 已补充到 Eval 测试中")
                        with col_del:
                            if st.button("🗑️ 删除该条", key=f"del_{idx}", use_container_width=True):
                                st.session_state[f"confirm_del_conv_{idx}"] = True

                    # 确认删除弹窗
                    if st.session_state.get(f"confirm_del_conv_{idx}", False):
                        st.warning(f"确认删除对话「{q[:40]}」？")
                        c_yes, c_no = st.columns(2)
                        with c_yes:
                            if st.button("✅ 确认删除", key=f"confirm_yes_conv_{idx}"):
                                conv_id = conv.get("id")
                                if conv_id:
                                    db.delete_conversation(conv_id)
                                st.session_state.pop(f"confirm_del_conv_{idx}", None)
                                st.rerun()
                        with c_no:
                            if st.button("❌ 取消", key=f"confirm_no_conv_{idx}"):
                                st.session_state.pop(f"confirm_del_conv_{idx}", None)
                                st.rerun()

    # ── Tab3: BUG 反馈 ──
    with tab_bug:
        bugs = db.get_bugs()
        bugs.sort(key=lambda x: x.get("ts", ""), reverse=True)

        if not bugs:
            st.info("暂无 BUG 反馈。在「坏例闭环」中标记坏例后，会自动生成 BUG 反馈单。")
        else:
            # 用户筛选
            all_users = sorted(set(b.get("user", "我") for b in bugs))
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                selected_user = st.selectbox("按用户筛选", ["全部"] + all_users, key="bug_user_filter")
            with col_filter2:
                filter_status = st.selectbox("筛选状态", ["全部", "待修复", "已修复"], key="bug_filter")

            filtered = bugs
            if selected_user != "全部":
                filtered = [b for b in filtered if b.get("user", "我") == selected_user]
            if filter_status != "全部":
                filtered = [b for b in filtered if b.get("status") == filter_status]

            # 统计（基于筛选后）
            pending = sum(1 for b in filtered if b.get("status") == "待修复")
            fixed = sum(1 for b in filtered if b.get("status") == "已修复")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总反馈", len(filtered))
            with col2:
                st.metric("待修复", pending)
            with col3:
                st.metric("已修复", fixed)

            st.markdown("---")

            for i, bug in enumerate(filtered):
                ts = bug.get("ts", "")[:19].replace("T", " ")
                q = bug.get("question", "")
                status = bug.get("status", "待修复")
                user = bug.get("user", "我")
                icon = "🔴" if status == "待修复" else "🟢"

                with st.expander(f"{icon} [{ts}] 👤{user} | {q[:60]}"):
                    st.markdown(f"**用户：** {user}")
                    st.markdown(f"**问题：** {q}")
                    st.markdown(f"**反馈原因：** {bug.get('reason', '') or '—'}")
                    st.markdown(f"**意图：** `{bug.get('intent', '')}`")
                    st.markdown(f"**回答：** {bug.get('answer', '')[:200]}")
                    st.markdown(f"**状态：** {status}")

                    if status == "待修复":
                        bug_id = bug.get("id")
                        if st.button("✅ 标记已修复", key=f"fix_{bug.get('id', i)}"):
                            if bug_id:
                                db.update_bug_status(bug_id, "已修复")
                            st.rerun()


# ════════════════════════════════════════
# 模块：Agent 管理
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

        # 预置可选组合（中文标签 → 三元组）
        COMBO_OPTIONS = [
            "对比价格（compare×seller×price）",
            "对比时效（compare×seller×transit_time）",
            "对比发货（compare×seller×ship_time）",
            "对比风险（compare×seller×neg_rate）",
            "推荐卖家（recommend×category×neg_rate）",
            "查品类运费（query×category×freight）",
            "查品类价格（query×category×price）",
            "查路线运费（query×route×freight）",
            "查路线时效（query×route×transit_time）",
            "聚合品类运费（aggregate×category×freight）",
            "聚合路线运费（aggregate×route×freight）",
        ]
        # 解析：中文标签 → (op×dim×metric) 三元组
        COMBO_MAP = {}
        for opt in COMBO_OPTIONS:
            _lbl = opt.split("（")[0]
            _val = opt.split("（")[1].rstrip("）")
            COMBO_MAP[_lbl] = _val

        # ── 禁用组合 ──
        with col1:
            st.subheader("🚫 禁用功能")
            st.caption("选择要停用的功能（可多选），停用后用户询问会被引导到其他能力")
            # 当前已禁用的中文标签
            current_labels = [k for k, v in COMBO_MAP.items() if v in disabled_combinations]
            selected_labels = st.multiselect("停用功能", list(COMBO_MAP.keys()), default=current_labels, key="combo_multi")
            if st.button("💾 保存停用功能", key="save_combo"):
                new_disabled = [COMBO_MAP[lbl] for lbl in selected_labels]
                filters["disabled_combinations"] = new_disabled
                save_yaml(FILTERS_FILE, filters)
                reload_agent_config()
                st.rerun()

            # 展示当前停用的（中文）
            if disabled_combinations:
                st.markdown("**当前已停用：**")
                for combo in disabled_combinations:
                    _lbl = [k for k, v in COMBO_MAP.items() if v == combo]
                    display = f"{_lbl[0]}（{combo}）" if _lbl else combo
                    st.markdown(f"- 🚫 {display}")

        # ── 禁用州 ──
        STATES_CN = {
            "SP": "圣保罗", "MG": "米纳斯吉拉斯", "RJ": "里约热内卢", "RN": "北里奥格兰德",
            "BA": "巴伊亚", "PR": "巴拉那", "RS": "南里奥格兰德", "SC": "圣卡塔琳娜",
            "PE": "伯南布哥", "CE": "塞阿拉", "MS": "南马托格罗索", "GO": "戈亚斯",
            "DF": "巴西利亚", "AM": "亚马孙", "PA": "帕拉", "MT": "马托格罗索",
        }
        with col2:
            st.subheader("🚫 禁用州")
            st.caption("停用后，收货地在该州的查询会被拦截")
            current_state_labels = [k for k in STATES_CN if k in disabled_states]
            sel_states = st.multiselect("停用州", list(STATES_CN.keys()), default=current_state_labels, key="state_multi",
                                        format_func=lambda s: f"{s}（{STATES_CN[s]}）")
            if st.button("💾 保存停用州", key="save_state"):
                filters["disabled_states"] = sel_states
                save_yaml(FILTERS_FILE, filters)
                reload_agent_config()
                st.rerun()
            if disabled_states:
                st.markdown("**当前已停用：**")
                for s in disabled_states:
                    st.markdown(f"- 🚫 {s}（{STATES_CN.get(s, '')}）")

        # ── 禁用品类 ──
        CATS_CN = {
            "office_furniture": "办公家具", "food_drink": "食品饮料", "books_general_interest": "图书",
            "watches_gifts": "手表礼品", "fashion_shoes": "鞋", "electronics": "电子产品",
            "audio": "音响", "sports_leisure": "运动", "computers": "电脑", "telephony": "手机",
            "baby": "母婴", "perfumery": "香水",
        }
        with col3:
            st.subheader("🚫 禁用品类")
            st.caption("停用后，该类目的查询会被拦截")
            current_cat_labels = [k for k in CATS_CN if k in disabled_categories]
            sel_cats = st.multiselect("停用品类", list(CATS_CN.keys()), default=current_cat_labels, key="cat_multi",
                                      format_func=lambda c: f"{CATS_CN[c]}（{c}）")
            if st.button("💾 保存停用品类", key="save_cat"):
                filters["disabled_categories"] = sel_cats
                save_yaml(FILTERS_FILE, filters)
                reload_agent_config()
                st.rerun()
            if disabled_categories:
                st.markdown("**当前已停用：**")
                for c in disabled_categories:
                    st.markdown(f"- 🚫 {CATS_CN.get(c, c)}（{c}）")

        st.markdown("---")
        st.subheader("🔍 当前拦截效果")
        # 用中文说明当前停用规则会拦截什么，不用用户输入三元组
        if disabled_combinations or disabled_states or disabled_categories:
            st.markdown("**根据当前停用规则，以下查询会被拦截：**")
            for combo in disabled_combinations:
                _lbl = [k for k, v in COMBO_MAP.items() if v == combo]
                st.markdown(f"- 🚫 停用**{_lbl[0] if _lbl else combo}** → 用户问这类问题会被引导到其他能力")
            for s in disabled_states:
                st.markdown(f"- 🚫 停用**{s}（{STATES_CN.get(s, '')}）** → 收货地在 {s} 的查询会被拦截")
            for c in disabled_categories:
                st.markdown(f"- 🚫 停用**{CATS_CN.get(c, c)}（{c}）** → 该类目查询会被拦截")
        else:
            st.info("当前没有停用任何功能，所有查询正常。")

        # ── 双层评审开关 ──
        st.markdown("---")
        st.subheader("🔬 双层评审开关")
        st.caption("开启后，每次对话会调用裁判模型评审（慢、费 token）。关闭时只在 Eval 环节评审。")
        toggles = load_toggles()
        dual_review_enabled = toggles.get("dual_review_enabled", False)
        new_val = st.toggle("对话时启用双层评审", value=dual_review_enabled, key="toggle_dual_review")
        if new_val != dual_review_enabled:
            toggles["dual_review_enabled"] = new_val
            save_toggles(toggles)
            st.rerun()

    # ── 提示词调优 ──
    with tab_prompts:
        prompts = load_yaml(PROMPTS_FILE)
        versions = db.get_prompt_versions()

        # ── 英文术语词典 ──
        PROMPT_GLOSSARY = {
            "提示词 Key": {
                "safety": "安全约束", "intent": "意图识别", "answer": "回答生成",
                "compare_answer": "对比回答", "aggregate_answer": "聚合回答",
                "recommend_answer": "推荐回答", "capability": "能力介绍",
                "methodology": "方法论", "unsupported": "不支持的功能", "other": "无关闲聊",
            },
            "三元组轴 + 指标": {
                "operation": "操作", "dimension": "维度", "metric": "指标",
                "seller": "商家", "category": "品类", "route": "路线",
                "ship_time": "发货时长", "transit_time": "运输时长", "total_time": "总时长",
                "freight": "运费", "price": "价格", "neg_rate": "差评率",
                "ontime_rate": "准时率", "promise_gap": "承诺偏差",
                "query": "查询", "compare": "对比", "aggregate": "聚合", "recommend": "推荐",
                "asc": "升序（从小到大）", "desc": "降序（从大到小）", "sort_direction": "排序方向",
            },
            "参数与字段": {
                "entities": "实体参数", "seller_ids": "卖家ID列表", "buyer_state": "收货州",
                "seller_state": "发货州", "intents": "意图数组", "chat_intent": "对话意图",
                "history_block": "对话历史", "cross_category": "跨品类",
                "review_reasons": "差评原因", "freight_estimate": "运费估算",
                "median_days": "中位天数", "avg_freight": "平均运费",
                "avg_promise": "平均承诺", "avg_actual": "平均实际", "n_reviews": "评论数",
                "data": "数据", "json": "JSON 数据格式", "null": "空值", "llm": "大语言模型",
            },
            "品类名": {
                "office_furniture": "办公家具", "food_drink": "食品饮料",
                "fashion_shoes": "时尚鞋类", "watches_gifts": "手表礼品",
                "books_general_interest": "大众图书", "bed_bath_table": "床品卫浴",
            },
            "其他": {
                "token": "令牌", "prompt": "提示词", "temperature": "温度",
                "max_tokens": "最大令牌数", "safety_rules": "安全规则",
                "olist": "巴西电商数据集",
            },
        }

        with st.expander("📖 英文术语词典（点击展开/收起）", expanded=False):
            glossary_search = st.text_input("🔍 搜索英文术语或中文释义", key="glossary_search", placeholder="输入关键词...")
            for group, terms in PROMPT_GLOSSARY.items():
                if glossary_search.strip():
                    q = glossary_search.strip().lower()
                    filtered = {k: v for k, v in terms.items() if q in k.lower() or q in v.lower()}
                else:
                    filtered = terms
                if filtered:
                    st.markdown(f"**{group}**")
                    cols = st.columns(3)
                    for j, (en, zh) in enumerate(filtered.items()):
                        cols[j % 3].markdown(f"- `{en}` — {zh}")

        st.markdown("---")

        # 提示词编辑
        st.subheader("✏️ 编辑提示词")
        prompt_keys = [k for k in prompts.keys() if k != "safety"]
        selected_key = st.selectbox("选择提示词", prompt_keys, key="select_prompt_key")

        if selected_key:
            current_content = prompts.get(selected_key, "")

            # MD 文件上传
            uploaded_md = st.file_uploader("📄 上传 MD 文件覆盖提示词", type=["md"], key="upload_prompt_md")
            if uploaded_md is not None:
                md_content = uploaded_md.getvalue().decode("utf-8")
                st.session_state[f"edit_{selected_key}"] = md_content
                st.info(f"✅ 已读取「{uploaded_md.name}」（{len(md_content)} 字符），内容已填入编辑框，点击「保存提示词」生效。")

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
                    new_ver = len([v for v in versions if v.get("key") == selected_key]) + 1
                    db.save_prompt_version(
                        key=selected_key,
                        content=edited,
                        version=new_ver,
                        user=st.session_state.get("admin_nickname", "admin"),
                    )

                    reload_agent_config()
                    st.success(f"✅ 已保存 {selected_key}（版本 {new_ver}）")
                    st.rerun()

            with col_b:
                if st.button("🔄 刷新版本历史", use_container_width=True):
                    st.rerun()

        # 版本历史（常驻展示）
        st.markdown("---")
        st.subheader("📜 版本历史")
        if selected_key:
            key_versions = [v for v in versions if v.get("key") == selected_key]
            # 当前生效版本号 = 最新一条
            current_ver = key_versions[-1].get("version", "?") if key_versions else None
            if key_versions:
                for i, v in enumerate(reversed(key_versions[-10:])):
                    ts = v.get("ts", "")[:19].replace("T", " ")
                    ver = v.get("version", "?")
                    v_user = v.get("user", "admin")
                    preview = (v.get("content", "")[:100] + "...") if len(v.get("content", "")) > 100 else v.get("content", "")
                    is_current = (ver == current_ver)
                    tag = " 🟢当前" if is_current else ""
                    with st.expander(f"v{ver}{tag} — {ts} ｜ 👤{v_user} ｜ {preview}"):
                        st.code(v.get("content", ""))
                        st.caption(f"操作者：**{v_user}**")
                        col_dl, col_rb, col_del = st.columns(3)
                        with col_dl:
                            st.download_button(
                                f"⬇️ 下载 v{ver}",
                                data=v.get("content", ""),
                                file_name=f"{selected_key}_v{ver}.md",
                                mime="text/markdown",
                                key=f"dl_{selected_key}_{ver}",
                                use_container_width=True,
                            )
                        with col_rb:
                            if st.button(f"↩️ 回滚", key=f"rollback_{i}", use_container_width=True):
                                prompts[selected_key] = v.get("content", "")
                                save_yaml(PROMPTS_FILE, prompts)
                                reload_agent_config()
                                st.success(f"✅ 已回滚到版本 {ver}")
                                st.rerun()
                        with col_del:
                            if is_current:
                                st.button("🔒 当前版本", key=f"del_{i}", disabled=True, use_container_width=True)
                            else:
                                if st.button(f"🗑️ 删除", key=f"del_{i}", use_container_width=True):
                                    st.session_state[f"confirm_del_ver_{i}"] = True

                        # 确认删除弹窗
                        if st.session_state.get(f"confirm_del_ver_{i}", False):
                            st.warning(f"确认删除 {selected_key} 版本 v{ver}（{ts}）？")
                            c_yes, c_no = st.columns(2)
                            with c_yes:
                                if st.button("✅ 确认删除", key=f"confirm_yes_ver_{i}"):
                                    ver_id = v.get("id")
                                    if ver_id:
                                        db.delete_prompt_version(ver_id)
                                    st.session_state.pop(f"confirm_del_ver_{i}", None)
                                    st.rerun()
                            with c_no:
                                if st.button("❌ 取消", key=f"confirm_no_ver_{i}"):
                                    st.session_state.pop(f"confirm_del_ver_{i}", None)
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
if page == "指标看板":
    page_dashboard()
elif page == "模型管理":
    page_model()
elif page == "测评管理":
    page_eval_mgmt()
elif page == "Agent 管理":
    page_agent()
