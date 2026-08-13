#!/usr/bin/env python3
"""
懂履约的 AI 购物助手 — Streamlit 双栏 Demo
左栏：用户视角（对话 + 模糊化回答）
右栏：面试官视角（trace 步骤流程图）
"""

import streamlit as st
import time, json

from agent import chat

st.set_page_config(page_title="懂履约的 AI 购物助手", layout="wide")

# ── 样式 ──
st.markdown("""
<style>
/* 左右栏高度对齐 */
[data-testid="column"] { min-height: 80vh; }
/* trace 步骤样式 */
.trace-step { padding: 8px 12px; margin: 4px 0; border-radius: 6px; font-size: 0.85rem; }
.trace-step-1 { background: #e3f2fd; }
.trace-step-2 { background: #e8f5e9; }
.trace-step-3 { background: #fff3e0; }
.trace-step-4 { background: #f3e5f5; }
.trace-step-5 { background: #e0f7fa; }
</style>
""", unsafe_allow_html=True)

st.title("🛒 懂履约的 AI 购物助手")

# ── 双栏 ──
col_chat, col_trace = st.columns([3, 1])

# ── 初始化会话状态 ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = []

# ────────────────────────────────────────
#  左栏：用户对话
# ────────────────────────────────────────
with col_chat:
    st.subheader("💬 对话")

    # 显示历史消息
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    if user_input := st.chat_input("输入你的问题，例如：买书架送到 SP 要多久？"):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 模拟思考
        with st.chat_message("assistant"):
            thinking = st.status("🤔 思考中...", expanded=False)
            time.sleep(0.8)  # 模拟延迟

            # 调用 agent
            intent_result, params, data, answer, trace = chat(user_input)

            thinking.update(label="✅ 思考完成", state="complete", expanded=False)
            st.markdown(answer)

        # 存储
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.traces.append(trace)

# ────────────────────────────────────────
#  右栏：面试官视角
# ────────────────────────────────────────
with col_trace:
    st.subheader("🔍 面试官视角")

    if not st.session_state.traces:
        st.info("对话后这里会显示内部推理过程。")
    else:
        # 显示最新的 trace
        latest = st.session_state.traces[-1]
        st.caption(f"最近一轮的推理链路（共 {len(latest)} 步）")

        step_styles = {
            "①意图识别": ("trace-step-1", "🎯"),
            "②参数提取": ("trace-step-2", "🔧"),
            "③品类推断": ("trace-step-3", "🔗"),
            "④数据查询": ("trace-step-4", "📊"),
            "⑤回答生成": ("trace-step-5", "💬"),
        }

        for t in latest:
            step = t["step"]
            css_class, icon = step_styles.get(step, ("", "•"))
            with st.expander(f"{icon} {step}", expanded=False):
                st.code(t["content"], language=None)

        # 历史轮次
        if len(st.session_state.traces) > 1:
            st.divider()
            st.caption(f"历史共 {len(st.session_state.traces)} 轮")
            for i, hist in enumerate(reversed(st.session_state.traces[:-1]), 1):
                with st.expander(f"第 {len(st.session_state.traces) - i} 轮", expanded=False):
                    for t in hist:
                        st.text(f"{t['step']}: {t['content'][:60]}...")
