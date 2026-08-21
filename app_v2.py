#!/usr/bin/env python3
"""
懂履约的 AI 购物助手 V2.0 — Streamlit 双栏 Demo
左栏对话气泡 + 右栏推理链路，和 V1 (app.py) 并存对比。
"""

import streamlit as st
import json
import time
import os
import sys
from datetime import datetime

st.set_page_config(page_title="懂履约的 AI 购物助手 V2.0", layout="wide", initial_sidebar_state="collapsed")

# 确保 logs 目录存在
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 数据库访问层
sys.path.insert(0, os.path.dirname(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    import db
    from agent_v2 import chat
except Exception as _import_err:
    import traceback
    st.error(f"⚠️ 模块加载失败：\n\n```\n{traceback.format_exc()}\n```")
    st.stop()

# ── 全局样式（和 V1 保持一致）──
st.markdown("""
<style>
/* 隐藏默认页面导航 */
[data-testid="stSidebarNav"] { display: none !important; }

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

/* 意图标签 */
.intent-tag {
    display: inline-block; padding: 2px 8px; margin: 2px;
    border-radius: 4px; font-size: 0.78rem; font-weight: 500;
    background: #e8eaf6; color: #3949ab;
}

/* ── 侧边栏按钮：细边框 + 内部文字，与后台统一 ── */
[data-testid="stSidebar"] .stButton button {
    border: 1px solid rgba(49,51,63,0.2);
    background: #fff;
    color: rgb(49,51,63);
    border-radius: 0.5rem;
    font-weight: 500;
    transition: background .2s, border-color .2s, color .2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #f0f2f6;
    border-color: rgba(49,51,63,0.4);
    color: rgb(49,51,63);
}
</style>
""", unsafe_allow_html=True)

st.title("🛒 懂履约的 AI 购物助手 V2.0")
st.caption("三元组意图识别 · query / compare / aggregate / recommend · 左边对话，右边推理链路")

# ── 测试引导（方便用户快速体验各能力，不与 Eval 用例重复）──
st.markdown("""
> **💡 你可以试试这些问题：**
> - 手表送到 RJ 大概几天能到？
> - 音响从 SP 发货到我这要多久？
> - 6560211a 这家店发货快吗？
> - b1a812 和 6560211a 哪家更便宜？
> - 咖啡和书架哪个运费更贵？
> - 电脑类目靠谱的卖家有推荐吗？
> - 各州发货速度最快的是哪些？
> - 这家店差评主要因为什么？
> - 帮我查一下 6560211a 家的包到手多少钱？
> - 你能帮我规划一下双十一购物清单吗？
""")

# ── 昵称输入（多人协作用）──
if "user_nickname" not in st.session_state:
    st.session_state.user_nickname = "我"

with st.sidebar:
    st.markdown("### 👤 用户昵称")
    nickname = st.text_input(
        "昵称",
        value=st.session_state.user_nickname,
        max_chars=20,
        help="标记你是谁，便于后台区分不同用户的记录",
        key="nickname_input",
    )
    if nickname != st.session_state.user_nickname:
        st.session_state.user_nickname = nickname

    st.markdown("---")
    if st.button("📊 运营工作台", use_container_width=True):
        st.switch_page("pages/admin.py")

# ── 初始化 ──
if "rounds" not in st.session_state:
    st.session_state.rounds = []

# V2 trace 步骤映射
STEP_META = {
    "①意图识别": ("t1", "🎯"),
    "②参数提取": ("t2", "🔧"),
    "③品类推断": ("t3", "🔗"),
    "③州名校验": ("t3", "🔗"),
    "③数据查询": ("t4", "📊"),
    "④回答生成": ("t5", "💬"),
}


def _format_intent(intent_result: dict) -> str:
    """将 V2 三元组意图格式化为可读 HTML"""
    if "chat_intent" in intent_result:
        return f'<span class="intent-tag">💬 {intent_result["chat_intent"]}</span>'

    parts = []
    for it in intent_result.get("intents", []):
        op = it.get("operation", "?")
        dim = it.get("dimension", "?")
        metric = it.get("metric", "?")
        sd = it.get("sort_direction", "")
        label = f"{op} × {dim} × {metric}"
        if sd:
            label += f" ({sd})"
        parts.append(f'<span class="intent-tag">{label}</span>')
    return " ".join(parts) if parts else '<span class="intent-tag">unknown</span>'


def render_trace(trace):
    """渲染推理链路步骤"""
    for t in trace:
        step = t["step"]
        # 动态匹配（③查询·xxx 等变体）
        css, icon = "t4", "📊"
        for key, (c, i) in STEP_META.items():
            if step.startswith(key):
                css, icon = c, i
                break
        with st.expander(f"{icon}  {step}", expanded=False):
            st.code(t["content"], language=None)


# ── 渲染历史轮次 ──
for idx, rnd in enumerate(st.session_state.rounds):
    st.markdown('<div class="round-row">', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="col-header">💬 对话</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-user">🧑‍💻 {rnd["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-ai">🤖 {rnd["answer"]}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="col-header">🔍 推理链路</div>', unsafe_allow_html=True)
        # 显示意图标签
        if "intent_html" in rnd:
            st.markdown(rnd["intent_html"], unsafe_allow_html=True)
        render_trace(rnd["trace"])

    st.markdown('</div>', unsafe_allow_html=True)

# ── 输入区 ──
st.markdown("---")
with st.form("ask_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "你的问题",
            placeholder="例如：哪些品类运费最贵？b1a812 和 5058e8 谁发货快？",
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
        intent_result, entities, all_data, answer, trace = chat(user_input.strip(), history or None, user=st.session_state.user_nickname)

    # ── 写对话日志 ──
    # 提取意图字符串
    if "chat_intent" in intent_result:
        intent_str = f"chat:{intent_result['chat_intent']}"
    else:
        parts = []
        for it in intent_result.get("intents", []):
            op, dim, metric = it.get("operation", "?"), it.get("dimension", "?"), it.get("metric", "?")
            parts.append(f"{op}×{dim}×{metric}")
        intent_str = "|".join(parts) if parts else "unknown"

    log_entry = {
        "ts": datetime.now().isoformat(),
        "user": st.session_state.user_nickname,
        "question": user_input.strip(),
        "intent": intent_str,
        "answer": answer,
    }
    db.insert_conversation(
        ts=log_entry["ts"], user=log_entry["user"],
        question=log_entry["question"], intent=log_entry["intent"],
        answer=log_entry["answer"]
    )

    st.session_state.rounds.append({
        "question": user_input.strip(),
        "answer": answer,
        "trace": trace,
        "intent_html": _format_intent(intent_result),
    })
    st.rerun()

st.caption("V2.0 · 三元组意图 · 对比/聚合/推荐")
