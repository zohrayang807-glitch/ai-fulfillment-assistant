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
INTENT_PROMPT = """你是一个意图分类器。将用户问题分到以下标签，可多选。

【A·业务意图 —— 需查数据给结论】
- time：配送时效。判据：问"多久到/几天/能不能按时到/会不会迟到"。典型："送到 RN 要多久？""来得及吗？"
- risk：卖家靠谱度。判据：问"靠不靠谱/退货方便吗/差评多不多/售后怎么样"。典型："这家店靠谱吗？""退货方便吗？"
- cost：价格对比。判据：问"多少钱/贵不贵/哪个更值/到手价/运费"。典型："两家哪个划算？""运费贵吗？"

【B·类业务意图 —— 讲解或温和拒绝，不查数据】
- capability：自我介绍。判据：问"你是谁/能做什么/有哪些功能"。典型："你能帮我做什么？""你是什么助手？"
- methodology：方法论。判据：问"你怎么判断/凭什么/怎么算的/数据哪来的"。典型："你是怎么判断配送时效的？""这个结论怎么来的？"
- unsupported：购物相关但未实现。判据：想让助手执行动作但该功能没做（砍价、查物流轨迹、催发货、退货售后流程、改地址）。典型："帮我砍价""我的包裹到哪了？"

【C·其他】
- other：与网购完全无关。判据：天气、股票、闲聊、问时间、讲笑话等。典型："今天天气怎么样？""讲个笑话"

【易混淆边界（必须遵守）】
1. 问"好不好/靠谱吗"是评价→risk；求"帮我做某事"是动作→unsupported
2. 问"你是谁/能做什么"→capability；问"你怎么判断/凭什么"→methodology
3. 问"多久/几天/来得及吗"→time；问"多少钱/贵不贵/运费"→cost
4. 跟网购相关但没实现→unsupported；跟网购完全无关→other

【输出格式】
一句话可能涉及多个意图（如"多久到+运费贵吗"= time+cost）。只输出 JSON：
{{"intents": ["time"], "reason": "简短理由"}}
intents 是数组，单意图时也用数组。合法值：time / risk / cost / capability / methodology / unsupported / other

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
    result = json.loads(text)

    # 兼容：新格式有 intents 数组，旧格式有 intent 字符串
    if "intents" in result:
        result["intent"] = result["intents"][0]
    elif "intent" in result:
        result["intents"] = [result["intent"]]

    return result


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
    if intent in ("capability", "methodology", "unsupported", "other"):
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

# ── 全局安全约束（所有回答共用）──
SAFETY_RULES = """【安全约束——必须遵守】
- 禁止替用户做最终购买决定，决定权交回用户
- 禁止使用煽动性带货话术
- 美妆/护肤/保健品：严禁承诺美白/抗衰/祛斑/治病功效，仅解读公开成分，提示"效果因人而异"
- 母婴用品：禁用"绝对安全、零风险"等绝对词汇，提示关注 3C 认证、国标、官方说明书
- 医疗器械：不能替代医生医嘱，仅做消费品参数对比，提示"遵从医嘱"
- 食品/生鲜：不承诺口味，提醒生产日期、配料表、过敏风险
- 二手商品：重点强调高风险和个体差异，不担保卖家靠谱，只提供验机避坑流程"""

# ── 业务意图回答 prompt（查数据给结论）──
ANSWER_PROMPT = f"""你是懂履约的购物助手。以下是查询到的结构化数据：

{{data}}

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

铁律：只基于提供的数据回答，不得编造或推算。

{SAFETY_RULES}"""

# ── 类业务意图 + 其他意图的 system prompt ──
_CAPABILITY_PROMPT = f"""你是"懂履约的购物助手"，一个帮用户做网购下单前决策的 AI。

用户问你是什么/能做什么。请口语化介绍你能帮用户做的三件事：
1. 判断时效——某件商品送到用户那里大概要多久
2. 识别卖家风险——这家店退货靠不靠谱、差评率高不高
3. 对比价格——两家店哪个更划算、到手价差多少

可以带一两个提问示例。语气自然亲切，像朋友推荐工具。不要编造不存在的能力。

{SAFETY_RULES}"""

_METHODOLOGY_PROMPT = f"""你是"懂履约的购物助手"。用户在问你的判断方法/数据来源。

请用口语解释你怎么帮用户做判断，要求：
- 模糊表述，禁止出现 P50、P90、n=、样本数、中位数、精确百分比等内部指标
- 用"参考历史订单里大多数人的实际收货时间""跟同类目其他卖家的平均线比"这类自然语言
- 说明"数字是从真实订单数据里查出来的，不是我编的"
- 说明数据是离线快照不是实时的，会标注不确定性
- 语气自然，不要像在念说明书

{SAFETY_RULES}"""

_UNSUPPORTED_PROMPT = f"""你是"懂履约的购物助手"。用户想要你做一个你目前做不到的事。

要求：
- 温和说明"目前还不具备这个功能"，不要冷冰冰地拒绝
- 然后自然地引导回你现有的能力——"但我可以帮你看看这类商品的时效/价格/卖家风险，需要吗？"
- 语气软化，像朋友说"这个我还不会，但我可以帮你看看别的"
- 不要编造不存在的功能

{SAFETY_RULES}"""

_OTHER_PROMPT = f"""你是"懂履约的购物助手"，但用户在问和网购无关的事。

要求：
- 直接陪用户聊，自然回答，不要拒绝、不要跳回购物话题
- 保持温和亲切的语气
- 你是一个懂电商履约的助手，但也可以正常闲聊
- 如果话题敏感（医疗/法律），提醒用户咨询专业人士

{SAFETY_RULES}"""

_SPECIAL_PROMPTS = {
    "capability": _CAPABILITY_PROMPT,
    "methodology": _METHODOLOGY_PROMPT,
    "unsupported": _UNSUPPORTED_PROMPT,
    "other": _OTHER_PROMPT,
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


def _llm_generate(system_prompt: str, user_question: str) -> str:
    """通用 LLM 生成，用于类业务意图和其他意图"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def generate_answer(user_question: str, data: Optional[dict]) -> str:
    # 类业务意图 / 其他意图 → LLM 动态生成
    if data and "special_intent" in data:
        intent = data["special_intent"]
        prompt = _SPECIAL_PROMPTS.get(intent)
        if prompt:
            return _llm_generate(prompt, user_question)
        return "暂不支持。"

    # 缺参数 → 反问
    if data and "need_info" in data:
        return _NEED_INFO_QUESTIONS.get(data["need_info"], "请补充信息。")

    # 无效州
    if data and "invalid_state" in data:
        return _invalid_state_msg(data["invalid_state"])

    if data is None:
        return "抱歉，这个数据暂时查不到，建议你直接联系卖家确认。"

    # 业务意图 → 数据驱动 + 安全约束
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
    # ── 多轮追问链 ──
    print("\n" + "▓" * 60)
    print("  Part 1: 多轮追问链")
    print("▓" * 60)
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

        print(f"\n📌 意图: {intent_result.get('intents', [intent_result.get('intent')])}（{intent_result['reason']}）")
        print(f"📌 参数: {json.dumps(params, ensure_ascii=False)}")
        print(f"📌 数据: {json.dumps(str(data), ensure_ascii=False)}")
        print(f"\n💬 回答:\n{answer}")
        print(f"\n🔍 Trace:")
        for t in trace:
            print(f"  {t['step']}: {t['content'][:80]}")
        print()

        history.append(f"用户：{q}")
        history.append(f"AI：{answer}")
        history = history[-10:]

    # ── 新意图验收（意图分类）──
    print("\n" + "▓" * 60)
    print("  Part 2: 意图分类验收")
    print("▓" * 60)
    verify = [
        ("你是怎么判断配送时效的？", "methodology"),
        ("这家到 RN 要多久，运费贵吗？", "time + cost"),
    ]
    for q, expected in verify:
        print(f"\n{'=' * 60}")
        print(f"  验收: {q}")
        print(f"  期望: {expected}")
        print("=" * 60)
        result = classify_intent(q)
        intents = result.get("intents", [result.get("intent")])
        print(f"  实际: {' + '.join(intents)}（{result['reason']}）")
        print()

    # ── 动态生成验收 ──
    print("\n" + "▓" * 60)
    print("  Part 3: 动态生成验收（类业务+其他意图）")
    print("▓" * 60)
    dynamic_tests = [
        ("今天天气怎么样？", "other", ["我主要帮你做网购决策"]),
        ("帮我砍个价", "unsupported", ["暂时还不具备", "但我可以帮你"]),
        ("你是怎么判断配送时效的？", "methodology", ["P50", "P90", "n=", "中位数"]),
        ("这款美白霜能祛斑吗？", "other", ["效果因人而异"]),
    ]
    for q, expected_intent, banned_keywords in dynamic_tests:
        print(f"\n{'=' * 60}")
        print(f"  测试: {q}")
        print(f"  意图: {expected_intent}")
        print(f"  禁止出现: {banned_keywords}")
        print("=" * 60)

        intent_result = classify_intent(q)
        intents = intent_result.get("intents", [intent_result.get("intent")])
        params = extract_params(q, intents[0])
        data = call_tool(intents[0], params)
        answer = generate_answer(q, data)

        print(f"\n💬 回答:\n{answer}")

        # 检查禁止词
        violations = [kw for kw in banned_keywords if kw in answer]
        if violations:
            print(f"\n⚠️  违规: 出现了禁止词 {violations}")
        else:
            print(f"\n✅ 未出现禁止词")
        print()


if __name__ == "__main__":
    self_test()
