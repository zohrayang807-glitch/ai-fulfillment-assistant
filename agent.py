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
from query import query_timing, query_promise, query_seller_risk, query_cost, query_recommend, query_seller_state, query_seller_categories, query_review_reason, query_value_score, query_freight_estimate, query_cost_baseline

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
- recommend：卖家推荐。判据：问"买XX哪家靠谱/推荐一家/有没有好的卖家/选哪家好"。典型："买书架哪家靠谱？""推荐个卖咖啡的"

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
5. 问"买XX哪家靠谱/推荐一家"→recommend（泛推荐）；问"这家卖家靠谱吗"→risk（指定卖家）
6. "买XX哪家靠谱"同时涉及品类→recommend；"XX卖家靠不靠谱"同时提到卖家ID→risk

【输出格式】
一句话可能涉及多个意图（如"多久到+运费贵吗"= time+cost）。只输出 JSON：
{{"intents": ["time"], "reason": "简短理由"}}
intents 是数组，单意图时也用数组。合法值：time / risk / cost / recommend / capability / methodology / unsupported / other

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
recommend 意图时 seller_ids 不适用（不查特定卖家），但 category 是核心，必须提取。

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
        seller_ids = params.get("seller_ids") or []

        # 缺收货地 → 反问
        if not buyer_state:
            return {"need_info": "buyer_state"}

        # ── 确定 effective_seller_state（4 级优先级）──
        effective_ss = None

        # 优先级 1：用户显式指定发货州
        if seller_state:
            effective_ss = seller_state
        # 优先级 2：有卖家 ID → 反查发货州
        elif seller_ids:
            ss, err = query_seller_state(seller_ids[0])
            if ss:
                effective_ss = ss
        # 优先级 3：品类→主要发货州推断
        elif category:
            main_state = get_main_seller_state(category)
            if main_state:
                effective_ss = main_state

        # ── 查实际时效分布 + 承诺偏差 ──
        timing = query_timing(effective_ss, buyer_state) if effective_ss else query_timing(None, buyer_state)
        promise = query_promise(effective_ss, buyer_state) if effective_ss else None

        if timing is None:
            return None

        # 给 timing 加上推断来源
        if effective_ss:
            if seller_ids:
                timing["source"] = f"卖家 {seller_ids[0][:10]}..（{effective_ss}）→{buyer_state}"
            elif category:
                timing["source"] = f"品类 {category} 主要发货州 {effective_ss}→{buyer_state}"
            else:
                timing["source"] = f"route {effective_ss}→{buyer_state}"
            timing["inferred_seller_state"] = effective_ss

        # 合并承诺数据（如有）
        if promise:
            timing["avg_promise"] = promise["avg_promise"]
            timing["avg_actual"] = promise["avg_actual"]
            timing["avg_gap"] = promise["avg_gap"]
            timing["ontime_rate"] = promise["ontime_rate"]

        return timing

    elif intent == "risk":
        seller_ids = params.get("seller_ids") or []
        category = params.get("category")

        # 有卖家 → 聚合风险 + 跨品类 + 差评原因
        if seller_ids:
            sid = seller_ids[0]
            risk = query_seller_risk(sid, category)
            if risk is None:
                return None

            # 跨品类表现
            cats = query_seller_categories(sid)
            if cats:
                risk["cross_category"] = cats

            # 卖家差评原因
            reasons = query_review_reason(seller_id=sid)
            if reasons:
                risk["review_reasons"] = reasons

            return risk

        # 只有品类（无卖家）→ 品类级差评原因
        if category:
            reasons = query_review_reason(category=category)
            if reasons:
                return {"category_reasons": reasons}
            return None

        return {"need_info": "seller_ids"}

    elif intent == "cost":
        seller_ids = params.get("seller_ids") or []
        category = params.get("category")
        buyer_state = params.get("buyer_state")

        # 缺收货地 → 反问
        if not buyer_state:
            return {"need_info": "buyer_state"}

        # 有卖家但缺品类 → 反问品类（不同品类价格差异大，必须指定）
        if seller_ids and not category:
            return {"need_info": "category"}

        # 多个卖家 → 对比 + 性价比评分
        if len(seller_ids) >= 2:
            results = []
            for sid in seller_ids:
                r = query_cost(sid, category, buyer_state)
                if r:
                    results.append(r)
            if not results:
                return None

            data = {"compare": True, "sellers": results}

            # 追加性价比评分
            if category and len(seller_ids) >= 2:
                vs = query_value_score(seller_ids, category, buyer_state)
                if isinstance(vs, list):
                    data["value_scores"] = vs

            return data

        # 只有品类（无卖家）→ 运费参考 + 价格基线
        if not seller_ids and category:
            freight = query_freight_estimate(category, buyer_state)
            price_baseline = query_cost_baseline(category, buyer_state)
            data = {}
            if freight:
                data["freight_estimate"] = freight
            if price_baseline:
                data["price_baseline"] = price_baseline
            return data if data else None

        # 单个卖家
        if seller_ids:
            return query_cost(seller_ids[0], category, buyer_state)

        return {"need_info": "seller_ids"}

    elif intent == "recommend":
        category = params.get("category")
        if not category:
            return {"need_info": "category"}
        return query_recommend(category, params.get("buyer_state"))

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

【防幻觉铁律——违反即失败】
5. 每个维度（时效/风险/价格/推荐）只基于提供的数据回答。该维度数据为 null 或未提供 → 明确说"这个暂时没有数据"或不主动提，严禁自行估算天数、金额、概率。
   ❌ 数据里没有 time 字段，却回答"大概 15 天" → 幻觉
   ✅ 数据里没有 time 字段 → "时效数据暂时查不到"或不提
6. 回答按用户提问顺序分点组织（先问的先答），不要混成一团。

【时效意图专属规则】
7. 如果数据中有 avg_promise / avg_actual / ontime_rate（承诺偏差数据），补一句：
   "平台承诺约 X 天，实际平均 Y 天，约 Z 成订单能按时到"
   模糊化：X/Y 用"约/左右"，Z 用"九成/绝大多数"等。没有承诺数据时严禁编造承诺天数。
8. 可预测性：用 p90 体现波动——"大多数约 X 天，九成在 Y 天内"（Y-X 大说明波动大），让用户感知时效稳不稳。
9. 承诺信息是加分项不是必答项，别把回答堆成数据报表。用户只问"多久到"时，以实际时效为主，承诺偏差点到为止。

【风险意图专属规则】
10. 数据中含 cross_category（跨品类表现）时，概括为：
    - "这家在 X 类卖得比较多，差评率比平均线低/高/差不多"
    - 如果 best 和 worst 品类差距大，点一句"Y 类比较稳，Z 类差评偏高"
    - 品类列表不用逐个罗列，挑最好和最差的说
11. 数据中含 review_reasons（差评原因）时，用自然语言概括：
    - "被骂得最多的是 W（约占 X 成），其次是 Y"
    - "其他"占比高时如实说"还有一部分评论没法归类"
    - note 字段标注"样本少"时要提及，不要忽略
12. 数据中含 category_reasons（品类级差评原因，无卖家）时：
    - "这类商品最常见的差评是 W，占了差不多 X 成"
    - 强调是品类共性，不是某个卖家的问题
13. 差评原因描述必须来自数据，严禁编造原因关键词或比例。

【价格意图专属规则】
14. 数据中含 value_scores（性价比评分）时，讲综合排序：
    - "综合看 X 更值：价格更低 + 时效更快 + 风险更低"
    - 如果某卖家某维度拖后腿，点一句"但时效偏慢"或"差评率稍高"
    - 评分是数据层算好的，LLM 只解释排序理由，不得自创评分依据
15. 数据中含 freight_estimate（品类运费参考）时：
    - "这类商品运费一般 X 左右"
    - 必须提及 note 中的不确定性（"受重量、距离、物流商影响"）
    - 如有 price_baseline，补一句"到手价大概 Y 左右"
16. 多卖家对比时，按 value_score 从高到低说，不要只报价格。
17. 价格数字必须模糊化（"大概 X 左右""约 Y"），禁止精确到小数。

{SAFETY_RULES}"""

# ── 类业务意图 + 其他意图的 system prompt ──
_CAPABILITY_PROMPT = f"""你是"懂履约的购物助手"，一个帮用户做网购下单前决策的 AI。

用户问你是什么/能做什么。请口语化介绍你能帮用户做的四件事：
1. 判断时效——某件商品送到用户那里大概要多久
2. 识别卖家风险——这家店退货靠不靠谱、差评率高不高
3. 对比价格——两家店哪个更划算、到手价差多少
4. 推荐卖家——帮你找出某品类里口碑好的卖家

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

你有且只有以下三件能力：
1. 配送时效——某件商品送到用户那里大概要多久
2. 卖家风险——这家店差评率高不高、退货靠不靠谱
3. 到手价格对比——两家店哪个更划算、含运费到手价差多少

【硬约束——违反即失败】
- 你没有任何其他功能。引导用户时只能提到上述三件，提到任何额外功能（如"历史价格走势""比价记录""物流轨迹""优惠券""砍价"等）都是错误。
- 温和说明"目前还不具备这个功能"
- 引导话术只能是："但我可以帮你看看配送时效/卖家靠不靠谱/到手价格，需要吗？"
- 语气软化，像朋友说"这个我还不会，但我可以帮你看看别的"

{SAFETY_RULES}"""

_OTHER_PROMPT = f"""你是"懂履约的购物助手"，但用户在问和网购无关的事。

要求：
- 直接陪用户聊，自然回答，不要拒绝、不要跳回购物话题
- 保持温和亲切的语气
- 你是一个懂电商履约的助手，但也可以正常闲聊
- 如果话题敏感（医疗/法律），提醒用户咨询专业人士

{SAFETY_RULES}"""

_RECOMMEND_PROMPT = f"""你是"懂履约的购物助手"。用户在问某品类推荐哪家卖家。

以下是查询到的推荐数据（该品类中好评率高于平均水平、且有足够订单量的卖家）：

{{data}}

回答规则：
1. 先给结论：推荐了哪几家，一句话概括口碑好在哪里
2. 数据依据必须模糊化：禁止输出 neg_rate、n_reviews 等原始指标
   ✅ "这几家的差评率都比品类平均低不少"
   ✅ "订单量也够，口碑比较稳定"
   ❌ "neg_rate=0.03，n_reviews=156"
3. 如有时效数据，可以补一句"送到你那里大概X天"
4. 强调仅供参考，决定权交回用户
5. 语气自然，像朋友帮你挑店

铁律：只基于提供的数据回答，不得编造或推算。

{SAFETY_RULES}"""

_SPECIAL_PROMPTS = {
    "capability": _CAPABILITY_PROMPT,
    "methodology": _METHODOLOGY_PROMPT,
    "unsupported": _UNSUPPORTED_PROMPT,
    "other": _OTHER_PROMPT,
    "recommend": _RECOMMEND_PROMPT,
}

_NEED_INFO_QUESTIONS = {
    "buyer_state": "我需要知道你的收货地（比如你在哪个州），才能帮你查。",
    "seller_ids": "你想查哪家卖家？请提供卖家 ID 或名称。",
    "category": "你想买哪类商品？比如书架、咖啡、美妆、手表等。",
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

    # recommend 返回 list → 用推荐专用 prompt
    if isinstance(data, list):
        data_str = json.dumps(data, ensure_ascii=False, indent=2)
        prompt = _RECOMMEND_PROMPT.format(data=data_str)
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
#  完整流程（多意图端到端）
# ═══════════════════════════════════════════════════════
_BUSINESS_INTENTS = {"time", "risk", "cost", "recommend"}


def chat(user_question: str, history=None):
    trace = []

    # Step 1: 意图识别
    intent_result = classify_intent(user_question, history)
    all_intents = intent_result.get("intents", [intent_result.get("intent")])
    trace.append({"step": "①意图识别", "content": f"{' + '.join(all_intents)}（{intent_result['reason']}）"})

    # 分离业务意图 vs 特殊意图
    biz_intents = [i for i in all_intents if i in _BUSINESS_INTENTS]
    special_intents = [i for i in all_intents if i not in _BUSINESS_INTENTS]

    # ── 无业务意图 → 走原有单意图逻辑 ──
    if not biz_intents:
        intent = all_intents[0]
        params = extract_params(user_question, intent, history)
        trace.append({"step": "②参数提取", "content": json.dumps(params, ensure_ascii=False)})
        data = call_tool(intent, params)
        trace.append({"step": "④数据查询", "content": json.dumps(str(data), ensure_ascii=False) if data else "无结果"})
        answer = generate_answer(user_question, data)
        trace.append({"step": "⑤回答生成", "content": answer[:100] + "..." if len(answer) > 100 else answer})
        return intent_result, params, data, answer, trace

    # ── 有业务意图 → 统一参数提取 ──
    # 用所有业务意图的并集来提取参数（一次提取，各意图取用）
    primary_intent = biz_intents[0]
    params = extract_params(user_question, primary_intent, history)
    trace.append({"step": "②参数提取", "content": json.dumps(params, ensure_ascii=False)})

    # 州名校验（time/cost 需要）
    if any(i in ("time", "cost") for i in biz_intents):
        bad_state = validate_states(params)
        if bad_state:
            trace.append({"step": "②.5州名校验", "content": f"❌ {bad_state} 不是合法巴西州"})
            answer = _invalid_state_msg(bad_state)
            trace.append({"step": "⑤回答生成", "content": answer[:100]})
            return intent_result, params, {"invalid_state": bad_state}, answer, trace

    # ── 遍历每个业务意图，逐个查询 ──
    results = []  # [{"intent": "time", "data": {...}}, ...]
    need_info_map = {}  # {"buyer_state": ["time"], "seller_ids": ["risk"]}

    for biz_i in biz_intents:
        d = call_tool(biz_i, params)
        # 收集 need_info
        if d and "need_info" in d:
            key = d["need_info"]
            need_info_map.setdefault(key, []).append(biz_i)
            trace.append({"step": f"④数据查询·{biz_i}", "content": f"⚠ 缺参数: {key}"})
        else:
            results.append({"intent": biz_i, "data": d})
            trace.append({"step": f"④数据查询·{biz_i}", "content": json.dumps(str(d), ensure_ascii=False) if d else "无结果"})

    # ── 缺参数处理：所有业务意图都缺 → 合并反问；部分缺 → 标记 ──
    if len(need_info_map) == len(biz_intents):
        # 全部缺参数 → 合并一次反问
        missing = []
        for key, intents in need_info_map.items():
            missing.append(_NEED_INFO_QUESTIONS.get(key, "请补充信息。"))
        answer = "还需要你补充一些信息：\n" + "\n".join(f"- {m}" for m in missing)
        trace.append({"step": "⑤回答生成", "content": answer[:100]})
        return intent_result, params, {"need_info_all": need_info_map}, answer, trace

    # ── 有结果 → 聚合回答 ──
    # 构建结构化数据块（只含真实查询结果）
    data_block = {}
    for r in results:
        if r["data"] is not None:
            data_block[r["intent"]] = r["data"]

    # 缺参数的意图单独说明
    missing_notes = []
    for key, intents in need_info_map.items():
        missing_notes.append(f"{'+'.join(intents)}: {_NEED_INFO_QUESTIONS.get(key, '请补充信息。')}")

    # 单意图 → 走原有 generate_answer
    if len(results) == 1 and not missing_notes:
        intent = results[0]["intent"]
        data = results[0]["data"]
        # recommend 返回 list 时 generate_answer 已能处理
        answer = generate_answer(user_question, data)
        trace.append({"step": "⑤回答生成", "content": answer[:100] + "..." if len(answer) > 100 else answer})
        return intent_result, params, data, answer, trace

    # 多意图 → 聚合 LLM 回答
    data_str = json.dumps(data_block, ensure_ascii=False, indent=2)
    extra = ""
    if missing_notes:
        missing_list = "\n".join(f"- {m}" for m in missing_notes)
        extra = f"""

【缺参数意图——强指令——违反即失败】
以下意图因缺少关键信息无法查询，必须在回答中明确告知用户：
{missing_list}

回答结构必须遵守：
1. 先回答已查到数据的维度（正常回答）
2. 再用单独一段说明缺失维度："关于 XX，需要你补充 YY 才能查询"（例："价格对比需要你告诉我是哪类商品，才能帮你比到手价"）
3. 引导用户补充信息

铁律：
- 缺失维度一律不给结论——没查到价格就不能说"综合更推荐 X"，没查到风险就不能说"这家更靠谱"
- 只能基于已查到的维度谈，且要明说"这只是时效维度的看法"或"这只是风险维度的参考"
- 违反以上规则即为失败回答"""

    prompt = ANSWER_PROMPT.format(data=data_str) + extra

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=0.7,
        max_tokens=600,
    )
    answer = resp.choices[0].message.content.strip()
    trace.append({"step": "⑤回答生成", "content": answer[:100] + "..." if len(answer) > 100 else answer})

    return intent_result, params, data_block, answer, trace


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
        ("买书架哪家靠谱？", "recommend"),
        ("282f23 这家卖家靠谱吗？", "risk"),
        ("咖啡到 RN 多久？", "time"),
        ("帮我砍个价", "unsupported"),
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
        ("帮我砍个价", "unsupported", ["历史价格走势", "比价记录", "物流轨迹", "优惠券"]),
        ("你是怎么判断配送时效的？", "methodology", ["P50", "P90", "n=", "中位数"]),
        ("这款美白霜能祛斑吗？", "other", ["效果因人而异"]),
        ("买书架哪家靠谱？", "recommend", ["neg_rate", "n_reviews", "n="]),
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

    # ── 多意图端到端验收 ──
    print("\n" + "▓" * 60)
    print("  Part 4: 多意图端到端验收")
    print("▓" * 60)

    multi_tests = [
        {
            "q": "a3dd39 退货靠谱吗？发到 MG 大概几天？",
            "expect_intents": ["risk", "time"],
            "must_contain_trace": ["④数据查询·risk", "④数据查询·time"],
            "banned_in_answer": ["15 天"],  # 防幻觉：不应出现编造数字
        },
        {
            "q": "这家到 RN 要多久？运费贵吗？",
            "expect_intents": ["time", "cost"],
            "must_contain_trace": ["④数据查询·time", "④数据查询·cost"],
            "banned_in_answer": [],
        },
        {
            "q": "a3dd39 靠谱吗？咖啡到 RN 多久？",
            "expect_intents": ["risk", "time"],
            "must_contain_trace": ["④数据查询·risk", "④数据查询·time"],
            "banned_in_answer": [],
        },
    ]

    for tc in multi_tests:
        q = tc["q"]
        print(f"\n{'=' * 60}")
        print(f"  测试: {q}")
        print(f"  期望意图: {' + '.join(tc['expect_intents'])}")
        print("=" * 60)

        intent_result, params, data, answer, trace = chat(q)
        intents = intent_result.get("intents", [intent_result.get("intent")])

        print(f"\n📌 意图: {' + '.join(intents)}")
        print(f"\n💬 回答:\n{answer}")
        print(f"\n🔍 Trace:")
        for t in trace:
            print(f"  {t['step']}: {t['content'][:80]}")

        # 检查 trace 中是否有多意图数据查询
        trace_steps = [t["step"] for t in trace]
        for must in tc["must_contain_trace"]:
            if must in trace_steps:
                print(f"\n✅ Trace 包含 {must}")
            else:
                print(f"\n⚠️  Trace 缺少 {must}")

        # 检查禁止词
        for banned in tc["banned_in_answer"]:
            if banned in answer:
                print(f"\n⚠️  违规: 回答中出现「{banned}」（疑似幻觉）")
            else:
                print(f"\n✅ 未出现「{banned}」")
        print()


if __name__ == "__main__":
    self_test()
