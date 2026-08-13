#!/usr/bin/env python3
"""
懂履约的 AI 购物助手 — Streamlit 双栏 Demo
每轮对话 = 一行，左列对话气泡 + 右列推理链路，自然对齐。
"""

import streamlit as st
import time

from agent import chat

st.set_page_config(page_title="懂履约的 AI 购物助手", layout="wide")

# ── 样式 ──
st.markdown("""
<style>
/* 气泡 */
.chat-bubble-user {
    background: #dcf8c6; padding: 10px 14px; border-radius: 12px 12px 2px 12px;
    margin: 4px 0; font-size: 0.95rem;
}
.chat-bubble-assistant {
    background: #f1f0f0; padding: 10px 14px; border-radius: 12px 12px 12px 2px;
    margin: 4px 0; font-size: 0.95rem;
}
/* trace 步骤 */
.trace-step { padding: 6px 10px; margin: 3px 0; border-radius: 6px; font-size: 0.82rem; }
.trace-1 { background: #e3f2fd; }
.trace-2 { background: #e8f5e9; }
.trace-3 { background: #fff3e0; }
.trace-4 { background: #f3e5f5; }
.trace-5 { background: #e0f7fa; }
/* 两列之间不留多余间距 */
[data-testid="column"] { padding: 0 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🛒 懂履约的 AI 购物助手")

# ── 初始化 ──
if "rounds" not in st.session_state:
    st.session_state.rounds = []  # [{question, answer, trace}]

STEP_META = {
    "①意图识别": ("trace-1", "🎯"),
    "②参数提取": ("trace-2", "🔧"),
    "③品类推断": ("trace-3", "🔗"),
    "④数据查询": ("trace-4", "📊"),
    "⑤回答生成": ("trace-5", "💬"),
}

# ── 渲染历史轮次（每轮一行，左右对齐）──
for idx, rnd in enumerate(st.session_state.rounds):
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f'<div class="chat-bubble-user">🧑 {rnd["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble-assistant">🤖 {rnd["answer"]}</div>', unsafe_allow_html=True)

    with col_right:
        for t in rnd["trace"]:
            step = t["step"]
            css, icon = STEP_META.get(step, ("", "•"))
            with st.expander(f"{icon} {step}", expanded=False):
                st.code(t["content"], language=None)

    st.divider()

# ── 输入框 ──
if user_input := st.chat_input("输入你的问题，例如：买书架送到 SP 要多久？"):
    # 调用 agent
    with st.status("🤔 思考中...", expanded=False):
        time.sleep(0.6)
        intent_result, params, data, answer, trace = chat(user_input)

    # 存储本轮
    st.session_state.rounds.append({
        "question": user_input,
        "answer": answer,
        "trace": trace,
    })

    # 立即渲染本轮（左右对齐）
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f'<div class="chat-bubble-user">🧑 {user_input}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble-assistant">🤖 {answer}</div>', unsafe_allow_html=True)

    with col_right:
        for t in trace:
            step = t["step"]
            css, icon = STEP_META.get(step, ("", "•"))
            with st.expander(f"{icon} {step}", expanded=False):
                st.code(t["content"], language=None)

    st.rerun()
