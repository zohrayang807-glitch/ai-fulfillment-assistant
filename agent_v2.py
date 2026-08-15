#!/usr/bin/env python3
"""
履约 AI 助手 V2.0 — 第一阶段骨架
核心升级：意图识别从「单标签」改为「三元组」（操作 × 维度 × 指标）

⚠️ 硬约束：不改动 agent.py（V1）和 query.py（数据层），只 import 不重写。
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

# ── 导入知识库查询（只调用，不重写）──
sys.path.insert(0, str(Path(__file__).resolve().parent / "knowledge_base"))
from query import (
    query_timing,
    query_promise,
    query_seller_risk,
    query_cost,
    query_recommend,
    query_seller_state,
    query_seller_categories,
    query_review_reason,
    query_value_score,
    query_freight_estimate,
    query_cost_baseline,
    query_ship_time,
)

# ── V2 聚合查询（不改 query.py）──
from query_v2 import (
    query_category_freight,
    query_route_freight,
    query_route_freight_single,
)

# ── 加载品类→主要发货州映射 ──
import pandas as pd

_KB = Path(__file__).resolve().parent / "knowledge_base"
_cat_main_state = pd.read_csv(_KB / "category_main_state.csv")
CAT_MAIN_STATE = dict(
    zip(_cat_main_state["category_en"], _cat_main_state["main_seller_state"])
)

# ── 加载 seller_risk → seller_id 到 city/state 的映射 ──
_seller_risk_df = pd.read_csv(_KB / "seller_risk.csv")
_SELLER_LOCATION = {}
for _, row in _seller_risk_df[["seller_id", "seller_city", "seller_state"]].drop_duplicates("seller_id").iterrows():
    _SELLER_LOCATION[row["seller_id"]] = {
        "seller_city": row["seller_city"],
        "seller_state": row["seller_state"],
    }


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


# ═══════════════════════════════════════════════════════
#  Step 1 · 三元组意图识别
# ═══════════════════════════════════════════════════════

INTENT_V2_PROMPT = """你是一个意图分类器。将用户问题解析为结构化三元组。

【三元组结构：操作 × 维度 × 指标】

操作（operation）——用户要"做什么"
- query：查单个对象某个指标的参考值（"这家发货多久""咖啡运费一般多少""大概多少钱"）
- compare：多个对象比较（"两家谁发货快"）
- aggregate：某个维度下的分布/排名（"哪些品类运费最贵""各品类发货速度排名""运费排行"）
- recommend：按品类推荐卖家（"买XX哪家靠谱""推荐个卖XX的"），dimension 固定为 category

⚠️ query vs aggregate 区分：
- "一般多少/平均多少/大概多少"→ query（查参考值），不是 aggregate
- "排名/排行/最贵/最便宜/分布"→ aggregate（需要排序或分布）
- 例："咖啡运费一般多少"→ query×category×freight；"哪些品类运费最贵"→ aggregate×category×freight

⚠️ aggregate 排序方向（sort_direction）——按指标语义区分默认值：
- ship_time / transit_time / total_time（时效类）："排名/排行"→ 默认 asc（最快在前）；"最慢"→ desc
- freight / price（价格运费类）："排名/排行"→ 默认 desc（最贵在前）；"最便宜"→ asc
- neg_rate（风险类）："排名/排行"→ 默认 desc（最差在前）
- 用户明确说"最快/最慢/最贵/最便宜"时，以用户说法为准，覆盖上述默认值

维度（dimension）——用户关心的"主体"
- seller：商家
- category：品类
- route：路线（发货州→收货州）

指标（metric）——用户要"什么信息"
- ship_time：发货时长（下单→交给快递）
- transit_time：运输时长（交给快递→送达）
- total_time：总时长（发货+运输）
- freight：运费
- price：价格
- neg_rate：差评率
- ontime_rate：准时率
- promise_gap：承诺偏差（平台承诺 vs 实际）

⚠️ total_time vs transit_time 区分：
- total_time 只在 seller 维度时使用（有发货段可拆，能同时查发货+运输）
- route / category 维度只有运输数据（查不到发货段），所以"多久到/几天到"在无卖家时一律识别为 transit_time
- 例："b1a812 多久能到"→ query×seller×total_time（有卖家，可拆段）
- 例："买书架送到 SP 要多久"→ query×route×transit_time（无卖家，只有运输段）
- 例："咖啡送到 RN 要几天"→ query×category×transit_time（无卖家，只有运输段）

⚠️ ship_time（发货时长）只支持 seller 维度：
- 发货是商家行为，品类不会发货，所以"category×ship_time"是伪命题
- 用户问"哪类商品发货快/各品类发货速度排名/哪些品类发货慢"→ 识别为 unsupported（不支持的操作），不要识别为 aggregate×category×ship_time
- route 维度同理：发货时长是卖家行为，不是路线属性

【对话类意图（不查数据，单独处理）】
- capability：自我介绍（"你是谁/能做什么"）
- methodology：方法论（"你怎么判断/数据哪来的"）
- unsupported：购物相关但未实现的功能（砍价、查物流轨迹、催发货）
- other：与网购完全无关（天气、闲聊）

【易混淆边界】
1. 问"发货多久/几天才发货"→ metric=ship_time（注意：是发货段，不是运输段）
2. 问"多久到/几天到/来得及吗"→ 无卖家时 metric=transit_time；有卖家时 metric=total_time
2a. ⚠️ "发到XX多久/运到XX几天"是问运输段（transit_time），不是问总时长。只有明确说"总共多久/一共几天"才是 total_time
3. 问"运费/物流费"→ metric=freight；问"价格/多少钱/到手价"→ metric=price
4. 问"靠不靠谱/差评多不多"→ metric=neg_rate
5. 问"准时吗/会不会迟到/承诺几天"→ metric=ontime_rate 或 promise_gap
6. "买XX哪家靠谱"→ recommend；"这家卖家靠谱吗"→ query×seller×neg_rate
7. 问"你是谁/能做什么"→ capability；问"你怎么判断"→ methodology
8. 跟网购相关但没实现→ unsupported；完全无关→ other
9. 问"谁快/谁便宜/谁靠谱/对比/比较/哪个更好"→ compare（即使只提到一个卖家也识别为 compare，系统会校验数量）
10. "A 和 B 谁发货快"→ compare×seller×ship_time；"A 对比价格"→ compare×seller×price

【多轮对话继承规则】
当用户的话只是补充信息（如"我在MS州""SP""就书架吧""送到RN"），没有提出新的问题时，继承上一轮的三元组（operation×dimension×metric），只更新 entities 参数重新查询。
- 例：上一轮 recommend → 用户说"我在MS州" → 仍为 recommend×category（带 buyer_state=MS），不要把补充信息识别成新的 query
- 例：上一轮 query×seller×ship_time → 用户说"送到RN" → 仍为 query×seller×ship_time（补充 buyer_state）
- 判断标准：用户没有问新的指标或操作，只是补充了收货地/品类/卖家等参数

【输出格式】
只输出 JSON。一句话可能涉及多个意图就输出多个三元组。对话类意图不进入三元组。

业务意图格式：
{{"intents": [{{"operation": "query", "dimension": "seller", "metric": "ship_time"}}],
  "entities": {{"seller_ids": ["b1a812"], "category": null, "buyer_state": null, "seller_state": null}}}}

对话类意图格式：
{{"chat_intent": "capability"}}

说明：
- seller_ids：卖家ID前缀列表，用户提到几个填几个，没有则填空数组 []
- category：商品品类英文名（映射到 Olist 品类，如 书→books_general_interest, 咖啡→food_drink, 书架→office_furniture, 手表→watches_gifts, 鞋→fashion_shoes, 床上用品→bed_bath_table）
- buyer_state：收货州（巴西2字母大写，如 SP/RN/MG/RJ/PE）
- seller_state：卖家发货州（用户明确提到时才填，否则 null）
- sort_direction：aggregate 时的排序方向，"desc"（最贵/最慢，默认）或 "asc"（最便宜/最快）
- dimension 为 route 时，buyer_state 必填（收货地）；seller_state 可由商家反查
- dimension 为 seller 时，seller_ids 必须至少有一个
- dimension 为 category 时，category 必须有值（aggregate 时可为空，返回全品类排名）

{history_block}"""


def classify_intent_v2(user_question: str, history=None) -> dict:
    """
    V2 三元组意图识别。
    返回格式：
      业务意图: {"intents": [...], "entities": {...}}
      对话意图: {"chat_intent": "capability|methodology|unsupported|other"}
    """
    history_block = ""
    if history:
        history_block = (
            "以下是本次会话最近几轮的对话历史，用于理解用户的追问和省略：\n"
            + "\n".join(history)
        )

    prompt = INTENT_V2_PROMPT.format(history_block=history_block)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=0,
        max_tokens=400,
    )
    text = resp.choices[0].message.content.strip()
    # 去掉可能的 markdown 代码块包裹
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


# ═══════════════════════════════════════════════════════
#  Step 2 · 数据映射表 & 参数必填规则
# ═══════════════════════════════════════════════════════

# 参数必填规则：(operation, dimension, metric) → 必填字段列表
REQUIRED_PARAMS = {
    # ship_time × seller：必须有商家
    ("query", "seller", "ship_time"): ["seller_ids"],
    # freight × seller：商家 + 品类缺一不可
    ("query", "seller", "freight"): ["seller_ids", "category"],
    # price × seller：商家 + 品类缺一不可
    ("query", "seller", "price"): ["seller_ids", "category"],
    # transit_time × route：收货地必填（始发地可经商家反查或品类推断）
    ("query", "route", "transit_time"): ["buyer_state"],
    # transit_time × seller：商家必填（收货地也必填，但发货州可反查）
    ("query", "seller", "transit_time"): ["seller_ids", "buyer_state"],
    # transit_time × category：品类 + 收货地必填
    ("query", "category", "transit_time"): ["category", "buyer_state"],
    # freight × route：收货地 + 始发地
    ("query", "route", "freight"): ["buyer_state", "seller_state"],
    # freight × category：品类必填
    ("query", "category", "freight"): ["category"],
    # price × category：品类必填
    ("query", "category", "price"): ["category"],
    # neg_rate × seller：商家必填
    ("query", "seller", "neg_rate"): ["seller_ids"],
    # ontime_rate / promise_gap × route：收货地必填
    ("query", "route", "ontime_rate"): ["buyer_state"],
    ("query", "route", "promise_gap"): ["buyer_state"],
    # ontime_rate / promise_gap × seller：商家 + 收货地
    ("query", "seller", "ontime_rate"): ["seller_ids", "buyer_state"],
    ("query", "seller", "promise_gap"): ["seller_ids", "buyer_state"],
    # total_time：同 transit_time
    ("query", "route", "total_time"): ["buyer_state"],
    ("query", "seller", "total_time"): ["seller_ids", "buyer_state"],
    ("query", "category", "total_time"): ["category", "buyer_state"],
    # recommend：品类必填
    ("recommend", "category", None): ["category"],
    # ── aggregate（无需额外参数，返回维度级排名）──
    ("aggregate", "category", "freight"): [],
    ("aggregate", "route", "freight"): [],
    # ── compare × seller（复用 query 的必填规则）──
    ("compare", "seller", "ship_time"): ["seller_ids"],
    ("compare", "seller", "transit_time"): ["seller_ids", "buyer_state"],
    ("compare", "seller", "total_time"): ["seller_ids", "buyer_state"],
    ("compare", "seller", "freight"): ["seller_ids", "category"],
    ("compare", "seller", "price"): ["seller_ids", "category"],
    ("compare", "seller", "neg_rate"): ["seller_ids"],
}

# 缺参数引导话术（seller_ids 按 metric 区分）
_MISSING_HINTS_DEFAULT = {
    "category": "你想查哪类商品呀？比如书架、咖啡、美妆这些～",
    "buyer_state": "你想送到哪个州？告诉我收货地址，我帮你查～",
    "seller_state": "是哪家店的发货地？告诉我卖家 ID，我可以帮你查～",
}

_SELLER_HINTS_BY_METRIC = {
    "ship_time": "想了解哪家店的发货速度？告诉我卖家 ID 或名称～",
    "transit_time": "想了解哪家店的运输时效？告诉我卖家 ID 或名称～",
    "total_time": "想了解哪家店的整体时效？告诉我卖家 ID 或名称～",
    "neg_rate": "想了解哪家店的靠谱程度？告诉我卖家 ID 或名称～",
    "ontime_rate": "想了解哪家店的准时率？告诉我卖家 ID 或名称～",
    "promise_gap": "想了解哪家店的承诺偏差？告诉我卖家 ID 或名称～",
    "freight": "想查哪家店的运费？告诉我卖家 ID 或名称～",
    "price": "想查哪家店的价格？告诉我卖家 ID 或名称～",
}


def _get_hint(field: str, metric: str = None) -> str:
    """获取缺参引导话术，seller_ids 按 metric 区分"""
    if field == "seller_ids":
        return _SELLER_HINTS_BY_METRIC.get(
            metric, "你想了解哪家店？告诉我卖家 ID 或名称～"
        )
    return _MISSING_HINTS_DEFAULT.get(field, f"请补充{field}")


def check_required_params(intent: dict, entities: dict) -> Optional[str]:
    """
    校验三元组所需实体参数是否齐全。
    返回缺参引导话术（合并多个缺失），或 None 表示参数齐全。
    """
    op = intent.get("operation", "query")
    dim = intent.get("dimension")
    metric = intent.get("metric")

    # 尝试精确匹配 (op, dim, metric) → 回退到 (op, dim, None)
    key = (op, dim, metric)
    if key not in REQUIRED_PARAMS:
        key = (op, dim, None)

    required = REQUIRED_PARAMS.get(key, [])
    if not required:
        return None  # 无必填要求

    missing = []
    for field in required:
        val = entities.get(field)
        if not val or (isinstance(val, list) and len(val) == 0):
            missing.append(field)

    if not missing:
        return None

    hints = [_get_hint(m, metric) for m in missing]
    return "还需要你补充一些信息哦：\n" + "\n".join(f"- {h}" for h in hints)


# ═══════════════════════════════════════════════════════
#  Step 3 · 数据查询（按映射表路由）
# ═══════════════════════════════════════════════════════


def _query_ship_time_seller(entities: dict) -> Optional[dict]:
    """ship_time × seller"""
    sid = entities["seller_ids"][0]
    result = query_ship_time(sid)
    if result and "error" in result:
        return None
    return result


def _query_transit_time_route(entities: dict) -> Optional[dict]:
    """transit_time × route（收货地 + 始发地）"""
    buyer = entities.get("buyer_state")
    seller = entities.get("seller_state")

    # 始发地缺失 → 尝试从商家反查
    if not seller:
        sids = entities.get("seller_ids") or []
        if sids:
            ss, err = query_seller_state(sids[0])
            if ss:
                seller = ss

    timing = query_timing(seller, buyer) if seller else query_timing(None, buyer)
    if timing is None:
        return None

    # 附加承诺数据
    if seller:
        promise = query_promise(seller, buyer)
        if promise:
            timing["avg_promise"] = promise["avg_promise"]
            timing["avg_actual"] = promise["avg_actual"]
            timing["avg_gap"] = promise["avg_gap"]
            timing["ontime_rate"] = promise["ontime_rate"]

    return timing


def _query_transit_time_seller(entities: dict) -> Optional[dict]:
    """transit_time × seller：先反查发货州，再查时效"""
    sid = entities["seller_ids"][0]
    buyer = entities.get("buyer_state")

    ss, err = query_seller_state(sid)
    if not ss:
        return None

    timing = query_timing(ss, buyer)
    if timing is None:
        return None

    timing["source"] = f"卖家 {sid[:10]}..（{ss}）→{buyer}"

    # 附加承诺数据
    promise = query_promise(ss, buyer)
    if promise:
        timing["avg_promise"] = promise["avg_promise"]
        timing["avg_actual"] = promise["avg_actual"]
        timing["avg_gap"] = promise["avg_gap"]
        timing["ontime_rate"] = promise["ontime_rate"]

    return timing


def _query_transit_time_category(entities: dict) -> Optional[dict]:
    """transit_time × category：按品类推断主要发货州，再查时效"""
    category = entities.get("category")
    buyer = entities.get("buyer_state")

    main_state = get_main_seller_state(category)
    if not main_state:
        return None

    timing = query_timing(main_state, buyer)
    if timing is None:
        return None

    timing["source"] = f"品类 {category} 主要发货州 {main_state}→{buyer}"
    timing["inferred_seller_state"] = main_state

    # 附加承诺数据
    promise = query_promise(main_state, buyer)
    if promise:
        timing["avg_promise"] = promise["avg_promise"]
        timing["avg_actual"] = promise["avg_actual"]
        timing["avg_gap"] = promise["avg_gap"]
        timing["ontime_rate"] = promise["ontime_rate"]

    return timing


def _query_total_time_seller(entities: dict) -> Optional[dict]:
    """total_time × seller：发货时长 + 运输时长"""
    sid = entities["seller_ids"][0]
    buyer = entities.get("buyer_state")

    # 发货段
    ship = query_ship_time(sid)

    # 运输段（反查发货州）
    ss, _ = query_seller_state(sid)
    timing = None
    if ss:
        timing = query_timing(ss, buyer)

    if not ship and not timing:
        return None

    result = {}
    if ship and "error" not in ship:
        result["ship_time"] = ship
    if timing:
        result["transit_time"] = timing
    return result


def _query_total_time_route(entities: dict) -> Optional[dict]:
    """total_time × route：无发货段，只返回运输时长"""
    return _query_transit_time_route(entities)


def _query_total_time_category(entities: dict) -> Optional[dict]:
    """total_time × category：品类推断发货州，返回运输时长"""
    return _query_transit_time_category(entities)


def _query_freight_seller(entities: dict) -> Optional[dict]:
    """freight × seller"""
    sid = entities["seller_ids"][0]
    category = entities.get("category")
    buyer = entities.get("buyer_state")
    return query_cost(sid, category, buyer)


def _query_freight_category(entities: dict) -> Optional[dict]:
    """freight × category"""
    category = entities.get("category")
    buyer = entities.get("buyer_state")
    return query_freight_estimate(category, buyer)


def _query_freight_route(entities: dict) -> Optional[dict]:
    """freight × route：读 route_freight.csv 单条查询"""
    seller = entities.get("seller_state")
    buyer = entities.get("buyer_state")
    # 尝试从 seller_ids 反查发货州
    if not seller:
        sids = entities.get("seller_ids") or []
        if sids:
            ss, _ = query_seller_state(sids[0])
            if ss:
                seller = ss
    if not seller or not buyer:
        return None
    return query_route_freight_single(seller, buyer)


def _query_price_seller(entities: dict) -> Optional[dict]:
    """price × seller"""
    sid = entities["seller_ids"][0]
    category = entities.get("category")
    buyer = entities.get("buyer_state")
    return query_cost(sid, category, buyer)


def _query_price_category(entities: dict) -> Optional[dict]:
    """price × category"""
    category = entities.get("category")
    buyer = entities.get("buyer_state")
    return query_cost_baseline(category, buyer)


def _query_neg_rate_seller(entities: dict) -> Optional[dict]:
    """neg_rate × seller"""
    sid = entities["seller_ids"][0]
    category = entities.get("category")
    return query_seller_risk(sid, category)


def _query_ontime_rate_route(entities: dict) -> Optional[dict]:
    """ontime_rate × route"""
    buyer = entities.get("buyer_state")
    seller = entities.get("seller_state")
    if not seller:
        sids = entities.get("seller_ids") or []
        if sids:
            ss, _ = query_seller_state(sids[0])
            if ss:
                seller = ss
    return query_promise(seller, buyer) if seller else query_promise(None, buyer)


def _query_promise_gap_route(entities: dict) -> Optional[dict]:
    """promise_gap × route（同 ontime_rate）"""
    return _query_ontime_rate_route(entities)


def _query_recommend(entities: dict) -> Optional[dict]:
    """recommend 操作：query_recommend + 补充 city/state + 截断 seller_id"""
    category = entities.get("category")
    buyer = entities.get("buyer_state")
    raw = query_recommend(category, buyer)
    if not raw or not isinstance(raw, list):
        return raw
    enriched = []
    for row in raw:
        sid = row.get("seller_id", "")
        loc = _SELLER_LOCATION.get(sid, {})
        enriched.append({
            "seller_id": sid[:8],  # 只保留前 8 位
            "seller_city": loc.get("seller_city"),
            "seller_state": loc.get("seller_state"),
            "neg_rate": row.get("neg_rate"),
            "n_reviews": row.get("n_reviews"),
            "median_days": row.get("median_days"),
            "p90_days": row.get("p90_days"),
            "timing_source": row.get("timing_source"),
        })
    return enriched


# ── aggregate 查询函数 ──

def _agg_category_freight(entities: dict) -> list:
    """aggregate × category × freight"""
    sd = entities.get("sort_direction", "desc")
    return query_category_freight(top_n=5, ascending=(sd == "asc"))


def _agg_route_freight(entities: dict) -> list:
    """aggregate × route × freight"""
    sd = entities.get("sort_direction", "desc")
    return query_route_freight(top_n=5, ascending=(sd == "asc"))


# ── 映射表：(operation, dimension, metric) → 查询函数 ──
QUERY_DISPATCH = {
    # ship_time
    ("query", "seller", "ship_time"): _query_ship_time_seller,
    # transit_time
    ("query", "route", "transit_time"): _query_transit_time_route,
    ("query", "seller", "transit_time"): _query_transit_time_seller,
    ("query", "category", "transit_time"): _query_transit_time_category,
    # total_time
    ("query", "seller", "total_time"): _query_total_time_seller,
    ("query", "route", "total_time"): _query_total_time_route,
    ("query", "category", "total_time"): _query_total_time_category,
    # freight
    ("query", "seller", "freight"): _query_freight_seller,
    ("query", "category", "freight"): _query_freight_category,
    ("query", "route", "freight"): _query_freight_route,
    # price
    ("query", "seller", "price"): _query_price_seller,
    ("query", "category", "price"): _query_price_category,
    # neg_rate
    ("query", "seller", "neg_rate"): _query_neg_rate_seller,
    # ontime_rate / promise_gap
    ("query", "route", "ontime_rate"): _query_ontime_rate_route,
    ("query", "route", "promise_gap"): _query_promise_gap_route,
    ("query", "seller", "ontime_rate"): _query_ontime_rate_route,
    ("query", "seller", "promise_gap"): _query_promise_gap_route,
    # recommend
    ("recommend", "category", None): _query_recommend,
    # aggregate
    ("aggregate", "category", "freight"): _agg_category_freight,
    ("aggregate", "route", "freight"): _agg_route_freight,
}


def dispatch_query(intent: dict, entities: dict):
    """按映射表路由到对应查询函数。

    多对象通用逻辑：若 dimension 对应的标识是多值（如 seller_ids 长度 > 1），
    则循环调用单对象 query 函数，返回对比列表（每条带对象标识 + 数据）。
    当前仅 seller 维度有多值场景，route/category 预留结构。
    """
    op = intent.get("operation", "query")
    dim = intent.get("dimension")
    metric = intent.get("metric")

    # 精确匹配 → 回退 (op, dim, None)
    key = (op, dim, metric)
    fn = QUERY_DISPATCH.get(key)
    if fn is None:
        key = (op, dim, None)
        fn = QUERY_DISPATCH.get(key)

    if fn is None:
        return None

    # ── 多对象检测：seller_ids 长度 > 1 → 循环对比 ──
    sids = entities.get("seller_ids") or []
    if len(sids) > 1:
        # 对每个卖家调用单对象 query 函数
        query_fn = QUERY_DISPATCH.get(("query", dim, metric))
        if query_fn is None:
            query_fn = QUERY_DISPATCH.get(("query", dim, None))
        if query_fn is None:
            return None

        compare_results = []
        for sid in sids:
            single_entities = {**entities, "seller_ids": [sid]}
            try:
                result = query_fn(single_entities)
            except Exception:
                result = None
            entry = {"seller_id": sid}
            if result and "error" not in result:
                entry["data"] = result
            else:
                entry["data"] = None
                entry["note"] = "无数据"
            compare_results.append(entry)

        # 按指标排序（越小越好：时效/运费/价格）
        compare_results.sort(
            key=lambda x: _extract_sort_value(x, metric),
            reverse=False,
        )
        return compare_results

    # ── 单对象 → 正常查询 ──
    return fn(entities)


# ═══════════════════════════════════════════════════════
#  Step 4 · 回答生成
# ═══════════════════════════════════════════════════════

SAFETY_RULES = """【安全约束——必须遵守】
- 禁止替用户做最终购买决定，决定权交回用户
- 禁止使用煽动性带货话术
- 美妆/护肤/保健品：严禁承诺功效，仅解读公开成分，提示"效果因人而异"
- 母婴用品：禁用"绝对安全、零风险"，提示关注 3C 认证、国标
- 医疗器械：不能替代医嘱，仅做消费品参数对比，提示"遵从医嘱"
- 食品/生鲜：不承诺口味，提醒生产日期、配料表、过敏风险
- 二手商品：强调高风险和个体差异，不担保卖家"""


ANSWER_V2_PROMPT = f"""你是懂履约的购物助手，语气柔和、亲切、带一点俏皮，像贴心的购物顾问。

以下是查询到的结构化数据：

{{data}}

回答规则：
1. 先给结论，用口语化表达，语气柔和亲切
2. 数据依据必须模糊化：用"约/大概/左右/一成/大多数/不到一成"等
   ✅ "这家店发货挺利索的，一般当天就交快递啦～"
   ✅ "快递运到你那大概 18 天左右"
   ✅ "运费大概 15 雷亚尔左右"
   ❌ "n=332, P50=18天, P90=35天"
   ❌ "均价 289.89+运费 13.57=303.47"
3. 标注不确定性：样本少、非实时、推断的发货州
4. 决定权交回用户（"如果你…可以…"）
5. 适度使用语气词（"啦""哦""试试看"），但不油腻不卖萌

【防幻觉铁律——违反即失败】
6. 每个维度只基于提供的数据回答。数据为 null → 明确说"暂时查不到"，严禁编造
7. 数字由数据层计算，LLM 不参与任何计算
8. 只回答用户明确问的指标，不主动补充用户没问的维度。例如用户只问"多久发货"→ 只答发货时长，不得脑补"运输 X 天、总共 Y 天"；用户只问"运费"→ 只答运费，不加价格对比
9. 禁止脑补数据之外的信息（如"清关""转运""分拣""派送"等数据里没有的环节），只基于提供的数据说话，不添加任何推测性描述

【发货+运输拆段规则——total_time 必须拆三段】
10. 用户问 total_time（"发到XX多久""总共多久""多久能到"）时：
    ⚠️ 前置条件：数据中同时有 ship_time 和 transit_time 时才拆三段：
    - 发货段："发货大概 X 天"（用 ship_time.median_days 模糊化）
    - 运输段："快递运到你那大概 Y 天"（用 transit_time.median_days 模糊化）
    - 总计："总共约 Z 天"
    如果数据只有 transit_time（没有 ship_time 字段），只答运输段，不拆段
11. 用户只问发货时长（ship_time）→ 只答发货段，绝不脑补运输时长和总时长
12. 发货天数模糊化（当天/一两天/三天左右/一周左右），禁止输出精确值
13. 无 ship_time 数据时只讲运输段，严禁编造发货天数。如果数据里只有 transit_time（没有 ship_time 字段），回答只说"快递运到你那大概 X 天"，禁止出现"发货"相关内容

【承诺偏差】
14. 有 avg_promise / avg_actual / ontime_rate 时补一句：
    "平台承诺约 X 天，实际平均 Y 天，约 Z 成订单能按时到"
15. 承诺信息是加分项不是必答项

【风险指标】
16. 有 neg_rate 时用口语表达："差评率约 3%，比同类平均还低一点，比较稳"
17. 有 cross_category 时概括最好和最差品类
18. 有 review_reasons 时用自然语言概括差评原因

【价格/运费】
19. 价格/运费数字模糊化（"大概 X 左右""约 Y"），禁止精确到小数
20. 有 freight_estimate 时提及不确定性（"受重量、距离影响"）

{SAFETY_RULES}"""


# ── 对话类意图 prompt ──
_CAPABILITY_PROMPT = f"""你是"懂履约的购物助手"，一个帮用户做网购下单前决策的 AI。

用户问你是什么/能做什么。请口语化介绍你能帮用户做的几件事：
1. 判断时效——某件商品送到用户那里大概要多久
2. 识别卖家风险——这家店退货靠不靠谱、差评率高不高
3. 对比价格——两家店哪个更划算、到手价差多少
4. 推荐卖家——帮你找出某品类里口碑好的卖家

语气自然亲切，像朋友推荐工具。可以带一两个提问示例。

【硬约束——违反即失败】
- 你只能处理用户用文字描述的信息。不能读也不能处理任何非文字内容，禁止提及"链接""截图""图片""聊天记录""详情页"等词（即使是说"不能"也不行，因为会误导用户以为有这些入口）
- 你有且只有上述 4 项能力，禁止承诺任何其他能力（如"帮你下单""查物流轨迹""砍价""催发货"等）
- 如果用户问"能不能读XX"，只说"你用文字告诉我就行"，不要列举你不能读的东西

{SAFETY_RULES}"""

_METHODOLOGY_PROMPT = f"""你是"懂履约的购物助手"。用户在问你的判断方法/数据来源。

请用口语解释：
- 模糊表述，禁止出现 P50、P90、n=、样本数、中位数、精确百分比等内部指标
- 用"参考历史订单里大多数人的实际收货时间""跟同类目其他卖家的平均线比"这类自然语言
- 说明"数字是从真实订单数据里查出来的，不是我编的"
- 说明数据是离线快照不是实时的，会标注不确定性
- 语气自然亲切

{SAFETY_RULES}"""

_UNSUPPORTED_PROMPT = f"""你是"懂履约的购物助手"。用户想要你做一个你目前做不到的事。

你有且只有以下几件能力：
1. 配送时效——某件商品送到用户那里大概要多久
2. 卖家风险——这家店差评率高不高、退货靠不靠谱
3. 到手价格对比——两家店哪个更划算、含运费到手价差多少
4. 卖家推荐——帮你找出某品类里口碑好的卖家

温和说明"目前还不具备这个功能"，引导回现有能力。引导时只能提上述几件。
语气软化，像朋友说"这个我还不会，但我可以帮你看看别的"。

{SAFETY_RULES}"""

_OTHER_PROMPT = f"""你是"懂履约的购物助手"，但用户在问和网购无关的事。

要求：
- 直接陪用户聊，自然回答，不要拒绝、不要跳回购物话题
- 保持温和亲切的语气
- 如果话题敏感（医疗/法律），提醒用户咨询专业人士

{SAFETY_RULES}"""

COMPARE_ANSWER_PROMPT = f"""你是懂履约的购物助手，语气柔和、亲切、带一点俏皮。

以下是多个卖家的对比数据：

{{data}}

回答规则：
1. 先给结论：谁更好/更快/更便宜，一句话概括
2. 对比说明，不是逐个罗列。例如"b1a812 发货快（一般当天），5058e8 偏慢（要拖两周），b1a812 明显更快"
3. 数据依据必须模糊化：用"约/大概/左右/一两天/拖两周"等
   ❌ "n=332, median_days=2.85"
   ✅ "发货大概要等两三天"
4. 查不到数据的卖家诚实说明"这家暂时没有相关数据"
5. 只基于查询结果对比，绝不编造
6. 决定权交回用户（"如果你…可以…"）
7. 适度使用语气词，保持亲切
8. 只答用户问的指标，不主动补充其他维度

{SAFETY_RULES}"""

AGGREGATE_ANSWER_PROMPT = f"""你是懂履约的购物助手，语气柔和、亲切、带一点俏皮。

以下是聚合排名数据：

{{data}}

回答规则：
1. 以"排行榜"形式回答，先给排名结论
2. 数据依据必须模糊化：用"约/大概/左右"
   ❌ "avg_freight=49.58, n=176"
   ✅ "运费大概 50 雷亚尔左右"
3. 品类名翻译成中文（如 office_furniture→办公家具、food_drink→食品饮料、fashion_shoes→时尚鞋靴）
4. 路线格式："XX 州→YY 州"
5. 只基于查询结果，不编造未查到的品类/路线
6. 如果有样本量信息，可提一句"样本较多，数据比较靠谱"
7. 决定权交回用户（"如果你想看更多排名…""如果你想知道某个品类的详情…"）
8. 适度使用语气词，保持亲切

【排序方向一致性——违反即失败】
9. 数据是按升序排列（值小的在前）→ 用"最快/最便宜/最短"描述
10. 数据是按降序排列（值大在前）→ 用"最慢/最贵/最长"描述
11. 排行榜标题必须和实际排序方向一致：
    - 数据排了最慢的在前 → 说"发货最慢的品类是…"，不能说"最快的"
    - 数据排了最贵的在前 → 说"运费最贵的品类是…"
12. 判断方向的方法：看数据中第一个条目的指标值是大还是小。值大→降序（最X），值小→升序（最Y）

{SAFETY_RULES}"""

_CHAT_PROMPTS = {
    "capability": _CAPABILITY_PROMPT,
    "methodology": _METHODOLOGY_PROMPT,
    "unsupported": _UNSUPPORTED_PROMPT,
    "other": _OTHER_PROMPT,
}


def _llm_generate(system_prompt: str, user_question: str) -> str:
    """通用 LLM 生成"""
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


def generate_answer_v2(
    user_question: str,
    intent_results: list,
    entities: dict,
    all_data: list,
) -> str:
    """
    V2 回答生成。
    intent_results: [{"operation", "dimension", "metric"}, ...]
    all_data: [{"intent": {...}, "data": {...}}, ...]
    """
    # ── 对话类意图（如果有 chat_intent）──
    # chat_intent 在 chat() 中单独处理，不会进到这里

    # ── 单意图 + 单数据 → 直接用 LLM 生成 ──
    if len(all_data) == 1:
        entry = all_data[0]
        data = entry["data"]
        intent = entry["intent"]

        if data is None:
            return "抱歉，这个数据暂时查不到呢，建议你直接联系卖家确认一下～"

        data_str = json.dumps(data, ensure_ascii=False, indent=2)

        # compare 数据 → 用对比专用 prompt
        if entry.get("compare"):
            prompt = COMPARE_ANSWER_PROMPT.format(data=data_str)
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

        # aggregate 数据 → 用排名专用 prompt
        if entry.get("aggregate"):
            prompt = AGGREGATE_ANSWER_PROMPT.format(data=data_str)
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

        prompt = ANSWER_V2_PROMPT.format(data=data_str)

        # recommend 返回 list → 特殊处理
        if isinstance(data, list):
            prompt = f"""你是懂履约的购物助手。以下是推荐数据：

{data_str}

回答规则：
1. 必须逐一列出具体卖家，每家说清楚：卖家ID缩写、所在城市/州、口碑好在哪里
   ✅ "推荐卖家 2059c39f（在 SP 州圣安德烈市），差评率低，评论量也不错"
   ❌ "帮你挑了几家靠谱的"（空话，没有具体信息）
2. 数据依据模糊化，用"差评率低/评论量够多"等口语，禁止输出 neg_rate、n_reviews 等精确数值
3. 用户给了收货地（buyer_state）→ 必须补充时效：数据里有 median_days 时，必须说"送到 XX 大概 Y 天"（模糊化），禁止说"时效没法确认/暂不确定"；数据里没有时效字段时，才可说"时效暂不确定"
   用户没给收货地 → 绝对不编造时效（禁止说"3-5天""一周左右"等无依据数字）
4. 卖家 ID 只显示前 8 位缩写（如 2059c39f），不要展示完整哈希
5. 强调仅供参考，决定权交回用户
6. 语气自然亲切，像朋友帮你挑店

{SAFETY_RULES}"""

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

    # ── 多意图 → 聚合数据后统一回答 ──
    data_block = {}
    has_compare = False
    for entry in all_data:
        if entry["data"] is not None:
            label = (
                f"{entry['intent']['operation']}×"
                f"{entry['intent']['dimension']}×"
                f"{entry['intent']['metric']}"
            )
            data_block[label] = entry["data"]
            if entry.get("compare"):
                has_compare = True

    if not data_block:
        return "抱歉，这些数据暂时查不到呢，建议你直接联系卖家确认一下～"

    data_str = json.dumps(data_block, ensure_ascii=False, indent=2)
    # 混合意图（含 compare）→ 用通用 prompt 即可，LLM 会根据数据结构自行判断
    prompt = ANSWER_V2_PROMPT.format(data=data_str)

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=0.7,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════
#  compare 辅助
# ═══════════════════════════════════════════════════════

# 指标→排序取值字段（全部越小越好）
_SORT_KEY_MAP = {
    "ship_time": "median_days",
    "transit_time": "median_days",
    "total_time": None,  # 特殊处理：ship_time.median_days + transit_time.median_days
    "freight": "avg_freight",
    "price": "avg_total",
    "neg_rate": "neg_rate",
}


def _extract_sort_value(entry: dict, metric: str) -> float:
    """从 compare 结果条目中提取排序用的数值，无数据排最后"""
    data = entry.get("data")
    if data is None:
        return float("inf")

    if metric == "total_time":
        # total_time 需要拆算：ship.median + transit.median
        ship = data.get("ship_time", {})
        transit = data.get("transit_time", {})
        ship_val = ship.get("median_days", 0) if ship else 0
        transit_val = transit.get("median_days", 0) if transit else 0
        return ship_val + transit_val

    key = _SORT_KEY_MAP.get(metric, "median_days")
    return data.get(key, float("inf"))


# ═══════════════════════════════════════════════════════
#  完整流程
# ═══════════════════════════════════════════════════════

# 全部操作已实现
_PENDING_OPS = set()


def chat(user_question: str, history=None):
    """
    V2 完整流程：意图识别 → 参数校验 → 数据查询 → 回答生成

    返回: (intent_result, entities, all_data, answer, trace)
    """
    trace = []

    # ── Step 1: 三元组意图识别 ──
    intent_result = classify_intent_v2(user_question, history)
    trace.append({"step": "①意图识别", "content": json.dumps(intent_result, ensure_ascii=False)})

    # ── 对话类意图 → 直接生成回答 ──
    if "chat_intent" in intent_result:
        chat_type = intent_result["chat_intent"]
        prompt = _CHAT_PROMPTS.get(chat_type)
        if prompt:
            answer = _llm_generate(prompt, user_question)
        else:
            answer = "这个我还真帮不上，不过时效、价格、卖家风险这几样我拿手，要不要试试？"
        trace.append({"step": "②回答生成", "content": answer[:100]})
        return intent_result, {}, [], answer, trace

    # ── 业务意图 ──
    intents = intent_result.get("intents", [])
    entities = intent_result.get("entities", {})
    trace.append({"step": "②参数提取", "content": json.dumps(entities, ensure_ascii=False)})

    # ── 州名校验 ──
    for key in ("buyer_state", "seller_state"):
        val = entities.get(key)
        if val and val.upper() not in VALID_STATES:
            msg = (
                f"「{val}」不是有效的巴西州缩写哦。巴西的州缩写是 2 位大写字母，"
                f"比如 SP（圣保罗）、MG（米纳斯吉拉斯）、RJ（里约）这些。"
                f"你是不是想写 MG 或 MS？告诉我正确的州名，我帮你查～"
            )
            trace.append({"step": "③州名校验", "content": f"❌ {val} 不合法"})
            return intent_result, entities, [], msg, trace

    # ── 遍历每个意图：校验参数 → 查询数据 ──
    all_data = []
    missing_hints = []

    for intent in intents:
        op = intent.get("operation", "query")
        dim = intent.get("dimension")
        metric = intent.get("metric")
        label = f"{op}×{dim}×{metric}"

        # unsupported → 引导回已有能力
        if op == "unsupported":
            guidance_map = {
                ("category", "ship_time"): "发货是商家行为，不同品类没有统一发货规则哦。你可以告诉我某家店，我帮你查它的发货速度；或者我推荐发货快的卖家给你～",
            }
            hint = guidance_map.get((dim, metric), "这个我还不具备呢，但我可以帮你查时效、价格、卖家风险，或者推荐靠谱卖家～")
            missing_hints.append(hint)
            trace.append({"step": f"③查询·{label}", "content": "→ 引导回已有能力"})
            continue

        # aggregate → 待实现
        if op in _PENDING_OPS:
            all_data.append({"intent": intent, "data": None, "note": f"{op} 操作待实现"})
            trace.append({"step": f"③查询·{label}", "content": "⏳ 待实现"})
            continue

        # compare → 多卖家对比（委托 dispatch_query 统一处理）
        if op == "compare":
            sids = entities.get("seller_ids") or []
            if len(sids) < 2:
                hint = "需要至少两个卖家才能对比哦，再告诉我一个卖家 ID 吧～"
                missing_hints.append(hint)
                trace.append({"step": f"③查询·{label}", "content": "⚠ 卖家不足 2 个"})
                continue

            # 参数校验（复用 query 规则）
            query_intent = {"operation": "query", "dimension": dim, "metric": metric}
            hint = check_required_params(query_intent, entities)
            if hint:
                missing_hints.append(hint)
                trace.append({"step": f"③查询·{label}", "content": f"⚠ 缺参数"})
                continue

            # 委托 dispatch_query（内含多卖家循环逻辑）
            compare_results = dispatch_query(query_intent, entities)
            if compare_results is None:
                all_data.append({"intent": intent, "data": None, "note": f"{dim}×{metric} 不支持对比"})
                trace.append({"step": f"③查询·{label}", "content": "⚠ 不支持此组合"})
                continue

            all_data.append({
                "intent": intent,
                "data": compare_results,
                "compare": True,
            })
            trace.append({
                "step": f"③查询·{label}",
                "content": f"对比 {len(sids)} 个卖家，{sum(1 for r in compare_results if r.get('data'))} 个有数据",
            })
            continue

        # aggregate → 调用 query_v2 聚合函数
        if op == "aggregate":
            fn = QUERY_DISPATCH.get((op, dim, metric))
            if fn is None:
                all_data.append({"intent": intent, "data": None, "note": f"{dim}×{metric} 不支持聚合"})
                trace.append({"step": f"③查询·{label}", "content": "⚠ 不支持此聚合组合"})
                continue
            # sort_direction 从 intent 传入 entities（LLM 放在 intent 里）
            agg_entities = {**entities}
            if "sort_direction" in intent:
                agg_entities["sort_direction"] = intent["sort_direction"]
            try:
                result = fn(agg_entities)
            except Exception:
                result = None
            if result:
                all_data.append({"intent": intent, "data": result, "aggregate": True})
                trace.append({"step": f"③查询·{label}", "content": f"聚合返回 {len(result)} 条"})
            else:
                all_data.append({"intent": intent, "data": None})
                trace.append({"step": f"③查询·{label}", "content": "无结果"})
            continue

        # recommend → 特殊处理（不需要 dimension×metric 的参数校验）
        if op == "recommend":
            hint = check_required_params({"operation": "recommend", "dimension": "category"}, entities)
            if hint:
                missing_hints.append(hint)
                trace.append({"step": f"③查询·{label}", "content": f"⚠ 缺参数"})
                continue
            data = dispatch_query({"operation": "recommend", "dimension": "category"}, entities)
            all_data.append({"intent": intent, "data": data})
            trace.append({"step": f"③查询·{label}", "content": json.dumps(str(data), ensure_ascii=False)[:80] if data else "无结果"})
            continue

        # 参数校验
        hint = check_required_params(intent, entities)
        if hint:
            missing_hints.append(hint)
            trace.append({"step": f"③查询·{label}", "content": f"⚠ 缺参数"})
            continue

        # 数据查询（dispatch_query 内含多卖家循环逻辑）
        data = dispatch_query(intent, entities)

        # 多卖家对比列表 → 标记 compare（驱动回答层用对比 prompt）
        is_compare = isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "seller_id" in data[0]

        all_data.append({"intent": intent, "data": data, **({"compare": True} if is_compare else {})})
        trace.append({
            "step": f"③查询·{label}",
            "content": json.dumps(str(data), ensure_ascii=False)[:80] if data else "无结果",
        })

    # ── 全部缺参数 → 合并反问 ──
    if missing_hints and not all_data:
        answer = missing_hints[0]  # 合并后的引导话术
        trace.append({"step": "④回答生成", "content": answer[:100]})
        return intent_result, entities, all_data, answer, trace

    # ── 部分缺参数 → 标记 ──
    if missing_hints:
        for hint in missing_hints:
            all_data.append({"intent": {"note": "缺参数"}, "data": None, "missing_hint": hint})

    # ── Step 4: 回答生成 ──
    # 过滤掉 note=待实现 和 缺参数 的条目，只保留有数据的
    valid_data = [d for d in all_data if d.get("data") is not None]

    if not valid_data:
        # 全部无数据
        if missing_hints:
            answer = missing_hints[0]
        else:
            answer = "抱歉，这些数据暂时查不到呢，建议你直接联系卖家确认一下～"
        trace.append({"step": "④回答生成", "content": answer[:100]})
        return intent_result, entities, all_data, answer, trace

    answer = generate_answer_v2(user_question, intents, entities, valid_data)

    # 如果有待实现的操作，追加提示
    pending = [d for d in all_data if d.get("note") == "待实现"]
    if pending:
        pending_ops = set(d["intent"]["operation"] for d in pending)
        answer += f"\n\n另外，{'、'.join(pending_ops)} 功能还在开发中，目前先用 query 帮你查啦～"

    # 如果有缺参数的意图，追加引导
    if missing_hints:
        answer += "\n\n" + "\n".join(missing_hints)

    trace.append({"step": "④回答生成", "content": answer[:100] + "..." if len(answer) > 100 else answer})
    return intent_result, entities, all_data, answer, trace


# ═══════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════

def self_test():
    print("\n" + "▓" * 60)
    print("  履约 AI 助手 V2.0 — query + compare + aggregate + recommend 验收")
    print("▓" * 60)

    tests = [
        # ── query 原有验收（回归）──
        {
            "q": "b1a812 多久能发货",
            "expect": "query×seller×ship_time",
            "note": "只答发货时长，不脑补运输/总时长",
            "banned": ["运输", "总共", "清关", "转运"],
        },
        {
            "q": "买书架送到 SP 要多久",
            "expect": "query×route×transit_time",
            "note": "识别为 transit_time（不是 total_time）",
            "banned": ["清关", "转运", "分拣"],
        },
        {
            "q": "咖啡运费一般多少",
            "expect": "query×category×freight",
            "note": "\"一般多少\"→ query，不是 aggregate",
            "banned": [],
        },
        {
            "q": "这家店靠谱吗",
            "expect": "引导提供卖家",
            "note": "引导话术应为\"靠谱程度\"",
            "banned": ["发货速度"],
        },
        {
            "q": "b1a812 发货快不快？送到 RN 要多久？",
            "expect": "query×seller×ship_time + query×seller×transit_time",
            "note": "多意图",
            "banned": [],
        },
        {
            "q": "你是谁？能做什么？",
            "expect": "capability",
            "note": "对话类意图",
            "banned": [],
        },
        # ── compare 验收 ──
        {
            "q": "b1a812 和 5058e8 谁发货快",
            "expect": "compare×seller×ship_time",
            "note": "两卖家对比发货时长",
            "banned": [],
        },
        {
            "q": "282f23 和 a3dd39 哪个靠谱",
            "expect": "compare×seller×neg_rate",
            "note": "两卖家对比差评率",
            "banned": [],
        },
        {
            "q": "b1a812 和 5058e8 哪个便宜，送到 RN，书架",
            "expect": "compare×seller×price",
            "note": "对比价格（需品类+收货地）",
            "banned": [],
        },
        {
            "q": "b1a812 对比发货速度",
            "expect": "引导至少两个卖家",
            "note": "明确 compare 但只给一个卖家 → 引导补充",
            "banned": [],
        },
        # ── aggregate 验收 ──
        {
            "q": "哪些品类运费最贵",
            "expect": "aggregate×category×freight",
            "note": "品类运费排名（desc，最贵在前）",
            "banned": [],
        },
        {
            "q": "各品类发货速度排名",
            "expect": "引导卖家维度",
            "note": "ship_time 只支持 seller 维度，应引导回卖家",
            "banned": [],
        },
        {
            "q": "哪些品类发货最慢",
            "expect": "引导卖家维度",
            "note": "ship_time 只支持 seller 维度，应引导回卖家",
            "banned": [],
        },
        {
            "q": "哪些路线运费最贵",
            "expect": "aggregate×route×freight",
            "note": "路线运费排名（desc，最贵在前）",
            "banned": [],
        },
        # ── recommend 验收 ──
        {
            "q": "我想买书，推荐个商家",
            "expect": "recommend×category",
            "note": "逐一列出卖家（含城市/州），无收货地不编造时效",
            "banned": ["3-5天", "一周左右"],
        },
        {
            "q": "推荐个卖咖啡的，送到 RN",
            "expect": "recommend×category",
            "note": "有收货地，可拼时效",
            "banned": [],
        },
    ]

    history = []
    all_pass = True
    for i, tc in enumerate(tests, 1):
        q = tc["q"]
        print(f"\n{'=' * 60}")
        print(f"  测试 {i}: {q}")
        print(f"  期望: {tc['expect']}")
        print(f"  说明: {tc['note']}")
        print("=" * 60)

        intent_result, entities, all_data, answer, trace = chat(q, history or None)

        print(f"\n📌 意图: {json.dumps(intent_result, ensure_ascii=False)}")
        print(f"\n💬 回答:\n{answer}")
        print(f"\n🔍 Trace:")
        for t in trace:
            print(f"  {t['step']}: {t['content'][:100]}")

        # 检查禁止词
        banned = tc.get("banned", [])
        violations = [kw for kw in banned if kw in answer]
        if violations:
            print(f"\n⚠️  违规: 回答中出现禁止词 {violations}")
            all_pass = False
        else:
            print(f"\n✅ 未出现禁止词")

        # 检查必须包含的词
        must_contain = tc.get("must_contain", [])
        if must_contain:
            missing_must = [kw for kw in must_contain if kw not in answer]
            if missing_must:
                print(f"⚠️  缺少必须包含的词: {missing_must}")
                all_pass = False
            else:
                print(f"✅ 包含必须词: {must_contain}")

        # 检查缺参话术
        if tc["expect"] == "引导提供卖家":
            if "靠谱程度" in answer:
                print("✅ 引导话术正确（靠谱程度）")
            elif "发货速度" in answer:
                print("⚠️  引导话术错误：应为\"靠谱程度\"而非\"发货速度\"")
                all_pass = False
            else:
                print(f"ℹ️  引导话术: {answer[:60]}")

        if tc["expect"] == "引导至少两个卖家":
            if "至少" in answer or "两个" in answer or "再告诉" in answer:
                print("✅ 引导补充卖家正确")
            else:
                print(f"⚠️  未引导补充卖家: {answer[:60]}")
                all_pass = False

        # ship_time 引导验收：不应返回 aggregate 数据，应引导回卖家维度
        if tc["expect"] == "引导卖家维度":
            intents_list = intent_result.get("intents", [])
            has_agg_ship = any(
                i.get("operation") == "aggregate" and i.get("metric") == "ship_time"
                for i in intents_list
            )
            if has_agg_ship:
                print("⚠️  仍识别为 aggregate×ship_time，未引导回卖家")
                all_pass = False
            elif "卖家" in answer or "商家" in answer or "店铺" in answer:
                print("✅ 引导回卖家维度")
            else:
                print(f"ℹ️  回答: {answer[:80]}")

        # compare 验收：检查回答是否包含对比内容
        if "compare" in tc["expect"]:
            intents_list = intent_result.get("intents", [])
            is_compare = any(i.get("operation") == "compare" for i in intents_list)
            if is_compare:
                print("✅ 识别为 compare 操作")
            else:
                print(f"⚠️  未识别为 compare: {intents_list}")
                all_pass = False

        # aggregate 验收：检查是否识别为 aggregate 操作
        if "aggregate" in tc["expect"]:
            intents_list = intent_result.get("intents", [])
            is_aggregate = any(i.get("operation") == "aggregate" for i in intents_list)
            if is_aggregate:
                print("✅ 识别为 aggregate 操作")
            else:
                print(f"⚠️  未识别为 aggregate: {intents_list}")
                all_pass = False

        # recommend 验收：检查是否识别为 recommend 操作
        if "recommend" in tc["expect"]:
            intents_list = intent_result.get("intents", [])
            is_recommend = any(i.get("operation") == "recommend" for i in intents_list)
            if is_recommend:
                print("✅ 识别为 recommend 操作")
            else:
                print(f"⚠️  未识别为 recommend: {intents_list}")
                all_pass = False

        print()

        history.append(f"用户：{q}")
        history.append(f"AI：{answer}")
        history = history[-10:]

    print("=" * 60)
    if all_pass:
        print("  ✅ 全部验收通过")
    else:
        print("  ⚠️  部分验收未通过，请检查上方输出")
    print("=" * 60)


if __name__ == "__main__":
    self_test()
