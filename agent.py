#!/usr/bin/env python3
"""
懂履约的 AI 购物助手 — 命令行原型
三步流程：意图识别 → 工具调用 → 回答生成
"""

import sys, os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ── 加载环境变量 ──
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── 初始化 DeepSeek 客户端 ──
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ── 导入知识库查询 ──
sys.path.insert(0, str(Path(__file__).resolve().parent / "knowledge_base"))
from query import query_timing, query_seller_risk, query_cost


# ═══════════════════════════════════════════════════════
#  Step 1 · 意图识别
# ═══════════════════════════════════════════════════════
INTENT_PROMPT = """你是一个意图分类器。用户在网购，请判断他关心哪个决策维度：
- time：能不能按时到、多久到、配送时效
- risk：这家店靠不靠谱、退货方不方便、售后保障
- cost：哪个更值、到手价、运费、价格对比

输入用户问题，只输出 JSON：
{"intent": "time|risk|cost", "reason": "简短理由"}"""


def classify_intent(user_question: str) -> dict:
    """调 DeepSeek 识别意图，返回 {"intent": ..., "reason": ...}"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": user_question},
        ],
        temperature=0,
        max_tokens=200,
    )
    import json
    text = resp.choices[0].message.content.strip()
    # 兼容 markdown code block 包裹
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


# ═══════════════════════════════════════════════════════
#  Step 2 · 工具调用（实体抽取 + 知识库查询）
# ═══════════════════════════════════════════════════════
EXTRACT_PROMPT = """从用户问题中提取查询参数，只输出 JSON：

如果意图是 time：
{"seller_state": "卖家州（2字母大写，如 SP/RJ/MG）", "buyer_state": "买家州（2字母大写，如 SP/RN/PE）"}

如果意图是 risk：
{"seller_id": "卖家ID前缀（10位hex）", "category": "商品类目英文名"}

如果意图是 cost：
{"seller_id": "卖家ID前缀（10位hex）", "category": "商品类目英文名", "buyer_state": "买家州（2字母大写）"}

常见类目名映射（用户说中文时用英文查）：
- 办公椅/办公家具 → office_furniture
- 手表/礼品 → watches_gifts
- 音响/音频 → audio
- 床上用品 → bed_bath_table
- 运动户外 → sports_leisure
- 电子产品 → electronics
- 家具 → furniture_decor

提取不到的字段填 null。"""


def extract_params(user_question: str, intent: str) -> dict:
    """调 DeepSeek 提取查询参数"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f"意图：{intent}\n用户问题：{user_question}"},
        ],
        temperature=0,
        max_tokens=300,
    )
    import json
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def call_tool(intent: str, params: dict) -> Optional[dict]:
    """根据意图调用对应的知识库查询函数"""
    if intent == "time":
        return query_timing(params.get("seller_state"), params.get("buyer_state"))
    elif intent == "risk":
        return query_seller_risk(params.get("seller_id"), params.get("category"))
    elif intent == "cost":
        return query_cost(params.get("seller_id"), params.get("category"), params.get("buyer_state"))
    return None


# ═══════════════════════════════════════════════════════
#  Step 3 · 回答生成
# ═══════════════════════════════════════════════════════
ANSWER_PROMPT = """你是懂履约的购物助手。以下是查询到的结构化数据：

{data}

请按规则组织回答：
1. 先给结论
2. 给数据依据（中位数/P90、占比、样本量）
3. 标注不确定性（样本小、代理信号、非实时）
4. 决定权交回用户（"如果你…可以…"）

铁律：只基于提供的数据回答，不得编造或推算。"""


def generate_answer(user_question: str, data: Optional[dict]) -> str:
    """把数据 + 用户问题喂给 DeepSeek，生成最终回答"""
    if data is None:
        return "抱歉，这个数据暂时查不到，建议你直接联系卖家确认。"

    import json
    data_str = json.dumps(data, ensure_ascii=False, indent=2)
    prompt = ANSWER_PROMPT.format(data=data_str)

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════
#  完整流程
# ═══════════════════════════════════════════════════════
def chat(user_question: str) -> str:
    """完整三步流程"""
    # Step 1: 意图识别
    intent_result = classify_intent(user_question)
    intent = intent_result["intent"]

    # Step 2: 实体抽取 + 工具调用
    params = extract_params(user_question, intent)
    data = call_tool(intent, params)

    # Step 3: 回答生成
    answer = generate_answer(user_question, data)
    return intent_result, params, data, answer


# ═══════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════
def _run_one(label: str, question: str, intent_result, params, data, answer):
    """打印单个测试用例的结果"""
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)

    print(f"\n📌 Step 1 · 意图识别")
    print(f"   意图: {intent_result['intent']}")
    print(f"   理由: {intent_result['reason']}")

    print(f"\n📌 Step 2 · 工具调用")
    print(f"   参数: {params}")
    print(f"   数据: {data}")

    print(f"\n📌 Step 3 · 回答生成")
    print(f"   {answer}")
    print()


def self_test():
    import json

    # ── 故事 1 · time 意图 ──
    q1 = "这个音响能10天内到吗？我在RN"
    ir1, p1, d1, a1 = chat(q1)
    _run_one("故事1（time）：音响能10天内到吗？我在RN", q1, ir1, p1, d1, a1)

    # ── 故事 2 · risk 意图 ──
    q2 = "卖家 a7f13822ce 的办公椅，退货靠不靠谱？"
    ir2, p2, d2, a2 = chat(q2)
    _run_one("故事2（risk）：a7f13822ce 办公椅退货靠不靠谱？", q2, ir2, p2, d2, a2)

    # ── 故事 3 · cost 意图（两个卖家对比）──
    q3 = "买家在SP，这两块表 b33e7c5544 和 d650b663c3 哪个更值得买？"
    print("=" * 60)
    print("  故事3（cost）：SP买家，b33e7c5544 vs d650b663c3 手表")
    print("=" * 60)

    # 意图识别（复用一次 LLM 调用）
    ir3 = classify_intent(q3)
    print(f"\n📌 Step 1 · 意图识别")
    print(f"   意图: {ir3['intent']}")
    print(f"   理由: {ir3['reason']}")

    # 分别查两个卖家
    d3a = query_cost("b33e7c5544", "watches_gifts", "SP")
    d3b = query_cost("d650b663c3", "watches_gifts", "SP")
    print(f"\n📌 Step 2 · 工具调用（×2）")
    print(f"   b33e7c5544: {d3a}")
    print(f"   d650b663c3: {d3b}")

    # 把两家数据合并喂给 LLM
    combined = {"b33e7c5544": d3a, "d650b663c3": d3b}
    combined_str = json.dumps(combined, ensure_ascii=False, indent=2)
    a3 = generate_answer(q3, combined)

    print(f"\n📌 Step 3 · 回答生成")
    print(f"   {a3}")
    print()


if __name__ == "__main__":
    self_test()
