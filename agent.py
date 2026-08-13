#!/usr/bin/env python3
"""
懂履约的 AI 购物助手 — 命令行原型（通用版）
三步流程：意图识别 → 工具调用 → 回答生成
支持任意品类 × 任意收货地 × 任意卖家
"""

import sys, os, json
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

# ── 加载品类→主要发货州映射 ──
import pandas as pd
_KB = Path(__file__).resolve().parent / "knowledge_base"
_cat_main_state = pd.read_csv(_KB / "category_main_state.csv")
CAT_MAIN_STATE = dict(zip(_cat_main_state["category_en"], _cat_main_state["main_seller_state"]))


def get_main_seller_state(category: str) -> Optional[str]:
    """查品类的主要发货州，查不到返回 None"""
    if category and category in CAT_MAIN_STATE:
        return CAT_MAIN_STATE[category]
    return None


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
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": user_question},
        ],
        temperature=0,
        max_tokens=200,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


# ═══════════════════════════════════════════════════════
#  Step 2 · 参数提取 + 工具调用
# ═══════════════════════════════════════════════════════
EXTRACT_PROMPT = """从用户问题中提取查询参数，只输出 JSON，字段缺失填 null：

{
  "category": "商品品类英文名（尽量映射到 Olist 品类，如 书→books_general_interest, 音响→audio, 办公椅/办公家具→office_furniture, 手表→watches_gifts, 咖啡→food_drink, 鞋→fashion_shoes, 床上用品→bed_bath_table, 电子产品→electronics, 运动→sports_leisure）",
  "buyer_state": "收货州（巴西2字母大写，如 SP/RN/MG/RJ/PE）",
  "seller_ids": ["卖家ID前缀列表（用户提到的所有卖家都填进去，没有则填空数组 []）"],
  "seller_state": "卖家发货州（用户明确提到时才填，否则 null）"
}

seller_ids 是数组，用户提到几个就填几个。例如提到两个卖家就填 ["aaa","bbb"]。"""


def extract_params(user_question: str, intent: str) -> dict:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f"意图：{intent}\n用户问题：{user_question}"},
        ],
        temperature=0,
        max_tokens=300,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def call_tool(intent: str, params: dict) -> Optional[dict]:
    """根据意图调用知识库查询，支持品类→主要发货州的自动推断"""
    if intent == "time":
        seller_state = params.get("seller_state")
        buyer_state = params.get("buyer_state")
        category = params.get("category")

        # 有卖家州 → 直接查
        if seller_state and buyer_state:
            return query_timing(seller_state, buyer_state)

        # 没有卖家州，有品类+买家州 → 先查品类主要发货州
        if category and buyer_state:
            main_state = get_main_seller_state(category)
            if main_state:
                result = query_timing(main_state, buyer_state)
                if result:
                    result["source"] = f"品类 {category} 主要发货州 {main_state}→{buyer_state}"
                    result["inferred_seller_state"] = main_state
                    return result

        # 兜底：全卖家→买家州
        if buyer_state:
            return query_timing(None, buyer_state)

        return None

    elif intent == "risk":
        seller_ids = params.get("seller_ids") or []
        sid = seller_ids[0] if seller_ids else None
        return query_seller_risk(sid, params.get("category"))

    elif intent == "cost":
        seller_ids = params.get("seller_ids") or []
        category = params.get("category")
        buyer_state = params.get("buyer_state")

        # 多个卖家 → 分别查询，返回对比结果
        if len(seller_ids) >= 2:
            results = []
            for sid in seller_ids:
                r = query_cost(sid, category, buyer_state)
                if r:
                    results.append(r)
            if results:
                return {"compare": True, "sellers": results}

        # 单个卖家
        if len(seller_ids) == 1:
            return query_cost(seller_ids[0], category, buyer_state)

        # 没提卖家
        return query_cost(None, category, buyer_state)

    return None


# ═══════════════════════════════════════════════════════
#  Step 3 · 回答生成
# ═══════════════════════════════════════════════════════
ANSWER_PROMPT = """你是懂履约的购物助手。以下是查询到的结构化数据：

{data}

请按规则组织回答：
1. 先给结论
2. 给数据依据（中位数/P90、占比、样本量）
3. 标注不确定性（样本小、代理信号、非实时、推断的发货州）
4. 决定权交回用户（"如果你…可以…"）

铁律：只基于提供的数据回答，不得编造或推算。"""


def generate_answer(user_question: str, data: Optional[dict]) -> str:
    if data is None:
        return "抱歉，这个数据暂时查不到，建议你直接联系卖家确认。"

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
def chat(user_question: str):
    intent_result = classify_intent(user_question)
    intent = intent_result["intent"]
    params = extract_params(user_question, intent)
    data = call_tool(intent, params)
    answer = generate_answer(user_question, data)
    return intent_result, params, data, answer


# ═══════════════════════════════════════════════════════
#  自测（5 句通用问题，不硬编码）
# ═══════════════════════════════════════════════════════
def self_test():
    questions = [
        "我想买个书架，送到 SP，要多久？",
        "买咖啡豆送到 MG 要几天？",
        "这个卖家 a7f13822ce 的办公家具退货靠谱吗？",
        "两个卖家 b33e7c5544 和 d650b663c3 的手表，在 SP 哪个划算？",
        "买鞋送到 RJ 要多久？",
    ]

    for i, q in enumerate(questions, 1):
        print("=" * 60)
        print(f"  测试 {i}: {q}")
        print("=" * 60)

        intent_result, params, data, answer = chat(q)

        print(f"\n📌 意图: {intent_result['intent']}（{intent_result['reason']}）")
        print(f"📌 参数: {json.dumps(params, ensure_ascii=False)}")
        print(f"📌 数据: {json.dumps(str(data), ensure_ascii=False)}")
        print(f"\n💬 回答:\n{answer}\n")


if __name__ == "__main__":
    self_test()
