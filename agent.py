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


# ── 巴西 27 个州 ──
VALID_STATES = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}


def validate_states(params: dict) -> Optional[str]:
    """校验 buyer_state / seller_state 是否合法，返回无效州缩写或 None"""
    for key in ("buyer_state", "seller_state"):
        val = params.get(key)
        if val and val.upper() not in VALID_STATES:
            return val
    return None


# ═══════════════════════════════════════════════════════
#  Step 1 · 意图识别
# ═══════════════════════════════════════════════════════
INTENT_PROMPT = """你是一个意图分类器。判断用户问题属于哪类：
- time：配送时效相关（能不能按时到、多久到）
- risk：卖家靠谱度相关（退货方不方便、售后保障、差评）
- cost：价格对比相关（哪个更值、到手价、运费）
- capability：用户问你是谁、能做什么、有哪些功能（"你能帮我做什么""你是谁""有什么功能"）
- unsupported：购物/履约相关但未实现的功能（砍价、查物流轨迹、催发货、退货售后流程、改地址）
- out_of_scope：与购物完全无关（天气、股票、闲聊、问时间）

输入用户问题，只输出 JSON：
{{"intent": "time|risk|cost|capability|unsupported|out_of_scope", "reason": "简短理由"}}

{history_block}"""


def classify_intent(user_question: str, history=None) -> dict:
    history_block = ""
    if history:
        history_block = "以下是本次会话最近几轮的对话历史，用于理解用户的追问和省略：\n" + "\n".join(history)

    prompt = INTENT_PROMPT.format(history_block=history_block)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
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

{{
  "category": "商品品类英文名（尽量映射到 Olist 品类，如 书→books_general_interest, 音响→audio, 办公椅/办公家具→office_furniture, 手表→watches_gifts, 咖啡→food_drink, 鞋→fashion_shoes, 床上用品→bed_bath_table, 电子产品→electronics, 运动→sports_leisure）",
  "buyer_state": "收货州（巴西2字母大写，如 SP/RN/MG/RJ/PE）",
  "seller_ids": ["卖家ID前缀列表（用户提到的所有卖家都填进去，没有则填空数组 []）"],
  "seller_state": "卖家发货州（用户明确提到时才填，否则 null）"
}}

seller_ids 是数组，用户提到几个就填几个。例如提到两个卖家就填 ["aaa","bbb"]。
如果用户用了"那""换个""换回"等追问词，请根据对话历史补全省略的参数。

{history_block}"""


def extract_params(user_question: str, intent: str, history=None) -> dict:
    history_block = ""
    if history:
        history_block = "以下是本次会话最近几轮的对话历史，用于理解用户的追问和省略：\n" + "\n".join(history)

    prompt = EXTRACT_PROMPT.format(history_block=history_block)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
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
    if intent in ("capability", "unsupported", "out_of_scope"):
        return {"special_intent": intent}

    if intent == "time":
        seller_state = params.get("seller_state")
        buyer_state = params.get("buyer_state")
        category = params.get("category")

        # 缺收货地 → 反问
        if not buyer_state:
            return {"need_info": "buyer_state"}

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
        return query_timing(None, buyer_state)

    elif intent == "risk":
        seller_ids = params.get("seller_ids") or []
        if not seller_ids:
            return {"need_info": "seller_ids"}
        return query_seller_risk(seller_ids[0], params.get("category"))

    elif intent == "cost":
        seller_ids = params.get("seller_ids") or []
        category = params.get("category")
        buyer_state = params.get("buyer_state")

        # 缺收货地 → 反问
        if not buyer_state:
            return {"need_info": "buyer_state"}

        # 缺卖家 → 反问
        if not seller_ids:
            return {"need_info": "seller_ids"}

        # 多个卖家 → 分别查询，返回对比结果
        if len(seller_ids) >= 2:
            results = []
            for sid in seller_ids:
                r = query_cost(sid, category, buyer_state)
                if r:
                    results.append(r)
            if results:
                return {"compare": True, "sellers": results}
            return None

        # 单个卖家
        return query_cost(seller_ids[0], category, buyer_state)

    return None


# ═══════════════════════════════════════════════════════
#  Step 3 · 回答生成
# ═══════════════════════════════════════════════════════
ANSWER_PROMPT = """你是懂履约的购物助手。以下是查询到的结构化数据：

{data}

回答规则：
1. 先给结论，用口语化表达
2. 数据依据必须模糊化：用"约/大概/左右/一成/大多数/少一半/近八成"等表述，禁止输出 n=、P50、P90、精确到手价等原始指标
   ✅ "大多数订单大概 18 天能到"
   ✅ "差不多 1 成订单能在 10 天内收到"
   ✅ "到手大概 300 雷亚尔左右"
   ❌ "n=332, P50=18天, P90=35天"
   ❌ "均价 289.89+运费 13.57=303.47"
3. 标注不确定性：样本少、非实时、推断的发货州
4. 决定权交回用户（"如果你…可以…"）

铁律：只基于提供的数据回答，不得编造或推算。"""


_SPECIAL_ANSWERS = {
    "capability": (
        "我可以帮你做三件事，都是网购下单前的决策：\n\n"
        "1️⃣ **判断时效**——某件商品送到你那里大概要多久、能不能赶上你的时间；\n"
        "2️⃣ **识别卖家风险**——这家店退货靠不靠谱、差评率高不高；\n"
        "3️⃣ **对比价格**——两家店哪个更划算、到手价差多少。\n\n"
        "你可以直接问我，比如『买书架送到 SP 要多久』『卖家 xxx 靠谱吗』『这两家哪个值』。"
    ),
    "unsupported": "这个功能我暂时还没做（我目前主要帮你判断配送时效、退货风险、价格）。不过我可以帮你看看这类商品的时效或价格，需要吗？",
    "out_of_scope": "我主要帮你做网购决策，这个问题我帮不上，建议你用专门的天气/资讯工具。",
}

_NEED_INFO_QUESTIONS = {
    "buyer_state": "我需要知道你的收货地（比如你在哪个州），才能帮你查。",
    "seller_ids": "你想查哪家卖家？请提供卖家 ID 或名称。",
}


def _invalid_state_msg(code: str) -> str:
    return (
        f"「{code}」不是有效的巴西州缩写。巴西的州缩写是 2 位大写字母，"
        f"比如 SP（圣保罗）、MG（米纳斯吉拉斯）、RJ（里约）等。"
        f"你是不是想写 MG 或 MS？告诉我正确的州名，我帮你查。"
    )


def generate_answer(user_question: str, data: Optional[dict]) -> str:
    # 特殊意图直接返回固定话术
    if data and "special_intent" in data:
        return _SPECIAL_ANSWERS.get(data["special_intent"], "暂不支持。")

    # 缺参数 → 反问
    if data and "need_info" in data:
        return _NEED_INFO_QUESTIONS.get(data["need_info"], "请补充信息。")

    # 无效州
    if data and "invalid_state" in data:
        return _invalid_state_msg(data["invalid_state"])

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
def chat(user_question: str, history=None):
    trace = []

    # Step 1: 意图识别
    intent_result = classify_intent(user_question, history)
    intent = intent_result["intent"]
    trace.append({"step": "①意图识别", "content": f"{intent}（{intent_result['reason']}）"})

    # Step 2: 参数提取
    params = extract_params(user_question, intent, history)
    trace.append({"step": "②参数提取", "content": json.dumps(params, ensure_ascii=False)})

    # Step 2.5: 州名校验
    bad_state = validate_states(params) if intent in ("time", "cost") else None
    if bad_state:
        trace.append({"step": "②.5州名校验", "content": f"❌ {bad_state} 不是合法巴西州"})
        answer = _invalid_state_msg(bad_state)
        trace.append({"step": "⑤回答生成", "content": answer[:100]})
        return intent_result, params, {"invalid_state": bad_state}, answer, trace

    # Step 3: 品类推断（如有）
    if intent == "time" and params.get("category") and params.get("buyer_state") and not params.get("seller_state"):
        inferred_state = get_main_seller_state(params["category"])
        if inferred_state:
            trace.append({"step": "③品类推断", "content": f"{params['category']} → 主要发货州 {inferred_state}"})

    # Step 4: 数据查询
    data = call_tool(intent, params)
    trace.append({"step": "④数据查询", "content": json.dumps(str(data), ensure_ascii=False) if data else "无结果"})

    # Step 5: 回答生成
    answer = generate_answer(user_question, data)
    trace.append({"step": "⑤回答生成", "content": answer[:100] + "..." if len(answer) > 100 else answer})

    return intent_result, params, data, answer, trace


# ═══════════════════════════════════════════════════════
#  自测（5 句通用问题，不硬编码）
# ═══════════════════════════════════════════════════════
def self_test():
    questions = [
        "买书架送到 SP 要多久？",
        "那换咖啡呢？",
        "咖啡退货靠谱吗？",
        "那换回书架，送到 RJ 呢？",
    ]

    history = []
    for i, q in enumerate(questions, 1):
        print("=" * 60)
        print(f"  测试 {i}: {q}")
        print("=" * 60)

        intent_result, params, data, answer, trace = chat(q, history or None)

        print(f"\n📌 意图: {intent_result['intent']}（{intent_result['reason']}）")
        print(f"📌 参数: {json.dumps(params, ensure_ascii=False)}")
        print(f"📌 数据: {json.dumps(str(data), ensure_ascii=False)}")
        print(f"\n💬 回答:\n{answer}")
        print(f"\n🔍 Trace:")
        for t in trace:
            print(f"  {t['step']}: {t['content'][:80]}")
        print()

        # 更新历史（滑动窗口 5 轮）
        history.append(f"用户：{q}")
        history.append(f"AI：{answer}")
        history = history[-10:]  # 5轮 × 2条 = 10条


if __name__ == "__main__":
    self_test()
