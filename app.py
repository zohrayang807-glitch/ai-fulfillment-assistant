#!/usr/bin/env python3
"""
懂履约的 AI 购物助手 — Streamlit 双栏 Demo
每轮对话一行：左列对话气泡 + 右列推理链路，1:1 对齐。
"""

import streamlit as st
import time

from agent import chat

st.set_page_config(page_title="懂履约的 AI 购物助手", layout="wide")

# ── 全局样式 ──
st.markdown("""
<style>
/* 全局字体 */
html, body, [class*="css"] { font-family: "Inter", "Noto Sans SC", sans-serif; }

/* 标题 */
h1 { font-weight: 700 !important; letter-spacing: -0.5px; }

/* 对话气泡 */
.bubble-user {
    background: linear-gradient(135deg, #d9fdd3, #c8f7c1);
    padding: 14px 18px; border-radius: 16px 16px 4px 16px;
    margin: 6px 0; font-size: 0.95rem; line-height: 1.6;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.bubble-ai {
    background: #f7f7f8;
    padding: 14px 18px; border-radius: 16px 16px 16px 4px;
    margin: 6px 0; font-size: 0.95rem; line-height: 1.6;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* 轮次行容器 */
.round-row {
    border: 1px solid #e8e8e8; border-radius: 12px;
    padding: 20px; margin: 12px 0;
    background: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

/* trace 步骤卡片 */
.trace-card {
    padding: 8px 12px; margin: 4px 0; border-radius: 8px;
    font-size: 0.82rem; cursor: pointer; transition: all 0.15s;
}
.trace-card:hover { transform: translateX(2px); }
.t1 { background: #e3f2fd; border-left: 3px solid #1976d2; }
.t2 { background: #e8f5e9; border-left: 3px solid #388e3c; }
.t3 { background: #fff8e1; border-left: 3px solid #f9a825; }
.t4 { background: #f3e5f5; border-left: 3px solid #7b1fa2; }
.t5 { background: #e0f7fa; border-left: 3px solid #00838f; }

/* 输入区固定底部 */
.input-area {
    position: sticky; bottom: 0; background: #fff;
    padding: 16px 0 8px 0; border-top: 1px solid #eee;
    z-index: 10;
}

/* 分隔线美化 */
hr { border: none; border-top: 1px solid #eee; margin: 8px 0; }

/* 列标题 */
.col-header {
    font-size: 0.85rem; font-weight: 600; color: #666;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 12px; padding-bottom: 6px;
    border-bottom: 2px solid #eee;
}
</style>
""", unsafe_allow_html=True)

st.title("🛒 懂履约的 AI 购物助手")
st.caption("输入购物问题，看 AI 如何用数据帮你决策 · 左边是你的对话，右边是推理过程")

# ── 初始化 ──
if "rounds" not in st.session_state:
    st.session_state.rounds = []

STEP_META = {
    "①意图识别": ("t1", "🎯"),
    "②参数提取": ("t2", "🔧"),
    "③品类推断": ("t3", "🔗"),
    "④数据查询": ("t4", "📊"),
    "⑤回答生成": ("t5", "💬"),
}


def render_trace(trace):
    """渲染推理链路步骤"""
    for t in trace:
        step = t["step"]
        css, icon = STEP_META.get(step, ("", "•"))
        with st.expander(f"{icon}  {step}", expanded=False):
            st.code(t["content"], language=None)


# ── 渲染历史轮次 ──
for idx, rnd in enumerate(st.session_state.rounds):
    st.markdown(f'<div class="round-row">', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f'<div class="col-header">💬 对话</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-user">🧑‍💻 {rnd["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-ai">🤖 {rnd["answer"]}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown(f'<div class="col-header">🔍 推理链路</div>', unsafe_allow_html=True)
        render_trace(rnd["trace"])

    st.markdown('</div>', unsafe_allow_html=True)

# ── 输入区 ──
st.markdown("---")
with st.form("ask_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "你的问题",
            placeholder="例如：买书架送到 SP 要多久？",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("发送 🚀", use_container_width=True)

if submitted and user_input.strip():
    # 构建历史（最近 5 轮 = 10 条）
    history = []
    for rnd in st.session_state.rounds[-5:]:
        history.append(f"用户：{rnd['question']}")
        history.append(f"AI：{rnd['answer']}")

    with st.spinner("🤔 思考中..."):
        time.sleep(0.5)
        intent_result, params, data, answer, trace = chat(user_input.strip(), history or None)

    st.session_state.rounds.append({
        "question": user_input.strip(),
        "answer": answer,
        "trace": trace,
    })
    st.rerun()
