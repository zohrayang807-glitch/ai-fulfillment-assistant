#!/usr/bin/env python3
"""
履约 AI 助手 V2.0 — 第一阶段骨架
核心升级：意图识别从「单标签」改为「三元组」（操作 × 维度 × 指标）

⚠️ 硬约束：不改动 agent.py（V1）和 query.py（数据层），只 import 不重写。
"""

import sys, os, json
from typing import Optional
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import yaml
import db

# ── 加载环境变量 ──
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── 配置目录 ──
_CONFIG_DIR = Path(__file__).resolve().parent / "config"
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)


def _load_prompts() -> dict:
    """从 config/prompts.yaml 加载提示词，注入 SAFETY_RULES"""
    with open(_CONFIG_DIR / "prompts.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    safety = raw["safety"]
    prompts = {}
    for key, val in raw.items():
        if key == "safety":
            prompts[key] = val
        else:
            prompts[key] = val.replace("{SAFETY_RULES}", safety)
    return prompts


def _load_model_config() -> dict:
    """从 config/model.yaml 加载模型配置"""
    with open(_CONFIG_DIR / "model.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_filters() -> dict:
    """从 config/filters.yaml 加载干预开关配置"""
    filters_path = _CONFIG_DIR / "filters.yaml"
    if not filters_path.exists():
        return {"disabled_combinations": [], "disabled_states": [], "disabled_categories": []}
    with open(filters_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # 确保三个 key 都存在
    return {
        "disabled_combinations": data.get("disabled_combinations", []),
        "disabled_states": data.get("disabled_states", []),
        "disabled_categories": data.get("disabled_categories", []),
    }


PROMPTS = _load_prompts()
MODEL_CFG = _load_model_config()
FILTERS = _load_filters()


def _load_toggles() -> dict:
    """从 logs/toggles.json 加载开关配置"""
    toggles_path = _LOG_DIR / "toggles.json"
    if not toggles_path.exists():
        return {}
    with open(toggles_path, encoding="utf-8") as f:
        return json.load(f)


def reload_config():
    """热重载配置（admin 保存后调用）"""
    global PROMPTS, MODEL_CFG, FILTERS
    PROMPTS = _load_prompts()
    MODEL_CFG = _load_model_config()
    FILTERS = _load_filters()


def _check_filters(intent: dict, entities: dict) -> Optional[str]:
    """检查意图是否命中过滤规则。返回拦截消息或 None。"""
    op = intent.get("operation", "")
    dim = intent.get("dimension", "")
    metric = intent.get("metric", "")
    combo = f"{op}×{dim}×{metric}"

    # 1. 禁用组合
    if combo in FILTERS.get("disabled_combinations", []):
        return f"「{combo}」功能已暂停使用。你可以试试：查时效、查运费、对比卖家、推荐品类～"

    # 2. 禁用州
    buyer_state = (entities.get("buyer_state") or "").upper()
    if buyer_state and buyer_state in FILTERS.get("disabled_states", []):
        return f"收货州 {buyer_state} 暂不支持查询。你可以换个州试试，或者让我推荐其他地区的卖家～"

    # 3. 禁用品类
    category = entities.get("category")
    if category and category in FILTERS.get("disabled_categories", []):
        return f"品类「{category}」暂不支持查询。你可以试试其他品类，或者让我推荐靠谱卖家～"

    return None


def _log_token_usage(model: str, prompt_tokens: int, completion_tokens: int, caller: str = "unknown"):
    """记录每次 API 调用的 token 用量"""
    entry = {
        "ts": datetime.now().isoformat(),
        "caller": caller,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    log_path = _LOG_DIR / "token_usage.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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

# INTENT_V2_PROMPT 已迁移到 config/prompts.yaml，通过 PROMPTS["intent"] 读取
INTENT_V2_PROMPT = PROMPTS["intent"]  # 保留变量名供内部引用


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
        model=MODEL_CFG["model"],
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=MODEL_CFG["temperature_intent"],
        max_tokens=MODEL_CFG["max_tokens_intent"],
    )
    _log_token_usage(MODEL_CFG["model"], resp.usage.prompt_tokens, resp.usage.completion_tokens, caller="intent")
    text = (resp.choices[0].message.content or "").strip()
    # 去掉可能的 markdown 代码块包裹
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # 兜底：LLM 偶发返回空/非 JSON 时，重试一次；仍失败返回安全默认，不崩
    if not text:
        resp = client.chat.completions.create(
            model=MODEL_CFG["model"],
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_question}],
            temperature=MODEL_CFG["temperature_intent"],
            max_tokens=MODEL_CFG["max_tokens_intent"],
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # 解析失败时返回安全默认（归类为 other，避免崩溃）
        return {"chat_intent": "other"}


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

# SAFETY_RULES 及所有 prompt 已迁移到 config/prompts.yaml，通过 PROMPTS[...] 读取
SAFETY_RULES = PROMPTS["safety"]
ANSWER_V2_PROMPT = PROMPTS["answer"]
_CAPABILITY_PROMPT = PROMPTS["capability"]
_METHODOLOGY_PROMPT = PROMPTS["methodology"]
_UNSUPPORTED_PROMPT = PROMPTS["unsupported"]
_OTHER_PROMPT = PROMPTS["other"]
COMPARE_ANSWER_PROMPT = PROMPTS["compare_answer"]
AGGREGATE_ANSWER_PROMPT = PROMPTS["aggregate_answer"]

_CHAT_PROMPTS = {
    "capability": PROMPTS["capability"],
    "methodology": PROMPTS["methodology"],
    "unsupported": PROMPTS["unsupported"],
    "other": PROMPTS["other"],
}


def _llm_generate(system_prompt: str, user_question: str, caller: str = "chat") -> str:
    """通用 LLM 生成"""
    resp = client.chat.completions.create(
        model=MODEL_CFG["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=MODEL_CFG["temperature_answer"],
        max_tokens=MODEL_CFG["max_tokens_answer"],
    )
    _log_token_usage(MODEL_CFG["model"], resp.usage.prompt_tokens, resp.usage.completion_tokens, caller=caller)
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
                model=MODEL_CFG["model"],
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_question},
                ],
                temperature=MODEL_CFG["temperature_answer"],
                max_tokens=MODEL_CFG["max_tokens_answer"],
            )
            _log_token_usage(MODEL_CFG["model"], resp.usage.prompt_tokens, resp.usage.completion_tokens, caller="compare_answer")
            return resp.choices[0].message.content.strip()

        # aggregate 数据 → 用排名专用 prompt
        if entry.get("aggregate"):
            prompt = AGGREGATE_ANSWER_PROMPT.format(data=data_str)
            resp = client.chat.completions.create(
                model=MODEL_CFG["model"],
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_question},
                ],
                temperature=MODEL_CFG["temperature_answer"],
                max_tokens=MODEL_CFG["max_tokens_answer"],
            )
            _log_token_usage(MODEL_CFG["model"], resp.usage.prompt_tokens, resp.usage.completion_tokens, caller="aggregate_answer")
            return resp.choices[0].message.content.strip()

        prompt = ANSWER_V2_PROMPT.format(data=data_str)

        # recommend 返回 list → 特殊处理
        if isinstance(data, list):
            prompt = PROMPTS["recommend_answer"].format(data=data_str)

        resp = client.chat.completions.create(
            model=MODEL_CFG["model"],
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question},
            ],
            temperature=MODEL_CFG["temperature_answer"],
            max_tokens=MODEL_CFG["max_tokens_answer"],
        )
        _log_token_usage(MODEL_CFG["model"], resp.usage.prompt_tokens, resp.usage.completion_tokens, caller="answer")
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
        model=MODEL_CFG["model"],
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=MODEL_CFG["temperature_answer"],
        max_tokens=MODEL_CFG["max_tokens_multi_answer"],
    )
    _log_token_usage(MODEL_CFG["model"], resp.usage.prompt_tokens, resp.usage.completion_tokens, caller="multi_answer")
    return resp.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════
#  双层评审
# ═══════════════════════════════════════════════════════

# 禁词列表（框架评审用）
_BANNED_WORDS = [
    "一定买", "必须买", "赶紧下单", "限时抢", "错过就没",
    "100%", "绝对好", "零风险", "无副作用",
]


def _framework_review(question: str, answer: str, all_data: list) -> dict:
    """第一层·框架评审（代码，客观）
    检查：数据一致性、拆段、模糊化、禁词
    """
    issues = []

    # 1. 禁词检查
    for w in _BANNED_WORDS:
        if w in answer:
            issues.append(f"包含禁词：{w}")

    # 2. 数据一致性：关键数字是否出现在回答中
    for entry in all_data:
        data = entry.get("data")
        if data is None:
            continue
        if isinstance(data, dict):
            # 检查关键数值字段
            for key in ("median_days", "avg_freight", "avg_total", "neg_rate", "ontime_rate"):
                val = data.get(key)
                if val is not None and isinstance(val, (int, float)):
                    val_str = f"{val:.1f}" if isinstance(val, float) else str(val)
                    # 如果数据有值但回答中完全没提到该指标的任何数字
                    # （宽松检查：只要回答里有数字就算通过）

    # 3. total_time 拆段检查
    for entry in all_data:
        intent = entry.get("intent", {})
        if intent.get("metric") == "total_time":
            data = entry.get("data", {})
            if isinstance(data, dict):
                has_ship = "ship_time" in data
                has_transit = "transit_time" in data
                if has_ship and has_transit:
                    # 检查回答是否同时提到发货和运输
                    if "发货" not in answer and "运输" not in answer and "快递" not in answer:
                        issues.append("total_time 有拆段数据但回答未分别说明发货/运输")

    return {
        "pass": len(issues) == 0,
        "issues": issues,
    }


def _model_judge_review(question: str, answer: str, all_data: list = None) -> dict:
    """第二层·模型评审（裁判模型，主观）
    调用 judge_model 打分（1-10）+ 评语
    传入 all_data 让裁判核对回答数字 vs 真实查询数据
    """
    judge_prompt = """你是一个严格的回答质量评审员。请对以下回答进行评分。

【评审标准】
1. 准确性（1-10）：核对「助手回答」中的数字是否与「真实查询数据」一致。一致给高分（8-10），数字有偏差给中分（5-7），凭空编造给低分（1-4）
2. 完整性（1-10）：是否完整回答了用户问题，是否有遗漏
3. 语气人设（1-10）：是否柔和、亲切、俏皮，不能生硬
4. 防幻觉（1-10）：回答是否基于真实数据，而非编造。如果回答中的数字都能在真实数据中找到依据，给高分

【重要】以下「真实查询数据」是系统真实查询的结果，回答必须基于这些数据。如果回答中的数字与真实数据一致，说明回答有据可依，不是编造。

【输出格式】
只输出 JSON，不要其他内容：
{"scores": {"accuracy": 8, "completeness": 7, "tone": 9, "anti_hallucination": 8}, "overall": 8, "comment": "简短评语"}"""

    # 构造数据摘要传给裁判
    data_summary = ""
    if all_data:
        data_items = []
        for entry in all_data:
            d = entry.get("data")
            if d is not None:
                intent = entry.get("intent", {})
                label = f"{intent.get('operation', '?')}×{intent.get('dimension', '?')}×{intent.get('metric', '?')}"
                data_items.append(f"[{label}] {json.dumps(d, ensure_ascii=False)}")
        if data_items:
            data_summary = "\n\n【真实查询数据】\n" + "\n".join(data_items)

    user_msg = f"【用户问题】{question}\n\n【助手回答】{answer}{data_summary}"

    try:
        resp = client.chat.completions.create(
            model=MODEL_CFG.get("judge_model", "deepseek-v4-pro"),
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=MODEL_CFG.get("judge_max_tokens", 500),
        )
        _log_token_usage(
            MODEL_CFG.get("judge_model", "deepseek-v4-pro"),
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            caller="judge",
        )
        text = resp.choices[0].message.content.strip()
        # 去掉可能的 markdown 包裹
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return {
            "scores": result.get("scores", {}),
            "overall": result.get("overall", 0),
            "comment": result.get("comment", ""),
        }
    except Exception as e:
        return {
            "scores": {},
            "overall": 0,
            "comment": f"评审异常：{str(e)[:100]}",
        }


def _run_dual_review(question: str, answer: str, all_data: list, user: str = "我") -> dict:
    """执行双层评审，返回合并结果"""
    framework = _framework_review(question, answer, all_data)
    model_judge = _model_judge_review(question, answer, all_data)
    return {
        "ts": datetime.now().isoformat(),
        "user": user,
        "question": question,
        "answer": answer[:500],
        "framework": framework,
        "model_judge": model_judge,
    }


def _save_evaluation(review: dict):
    """将评审结果保存到数据库"""
    db.insert_evaluation(
        ts=review.get("ts", ""),
        question=review.get("question", ""),
        answer=review.get("answer", ""),
        scores=review.get("scores", {}),
        overall=review.get("overall", 0),
        comment=review.get("comment", ""),
        user=review.get("user", ""),
    )


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


def chat(user_question: str, history=None, user: str = "我"):
    """
    V2 完整流程：意图识别 → 参数校验 → 数据查询 → 回答生成

    返回: (intent_result, entities, all_data, answer, trace)
    """
    reload_config()  # 每次对话重新加载配置（支持热更新）
    trace = []

    # ── Step 1: 三元组意图识别 ──
    intent_result = classify_intent_v2(user_question, history)
    trace.append({"step": "①意图识别", "content": json.dumps(intent_result, ensure_ascii=False)})

    # ── 对话类意图 → 直接生成回答 ──
    if "chat_intent" in intent_result:
        chat_type = intent_result["chat_intent"]
        prompt = _CHAT_PROMPTS.get(chat_type)
        if prompt:
            answer = _llm_generate(prompt, user_question, caller=chat_type)
        else:
            answer = "这个我还真帮不上，不过时效、价格、卖家风险这几样我拿手，要不要试试？"
        trace.append({"step": "②回答生成", "content": answer[:100]})
        # 双层评审（对话类意图也评审，仅在开关开启时）
        toggles = _load_toggles()
        if toggles.get("dual_review_enabled", False):
            try:
                review = _run_dual_review(user_question, answer, [], user=user)
                _save_evaluation(review)
            except Exception:
                pass
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
        # ── 干预过滤：命中禁用规则则跳过 ──
        filter_msg = _check_filters(intent, entities)
        if filter_msg:
            missing_hints.append(filter_msg)
            label = f"{intent.get('operation')}×{intent.get('dimension')}×{intent.get('metric')}"
            trace.append({"step": f"③查询·{label}", "content": "🚫 命中过滤规则"})
            continue
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
        # 双层评审（仅在开关开启时）
        toggles = _load_toggles()
        if toggles.get("dual_review_enabled", False):
            try:
                review = _run_dual_review(user_question, answer, [], user=user)
                _save_evaluation(review)
            except Exception:
                pass
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
        # 双层评审（仅在开关开启时）
        toggles = _load_toggles()
        if toggles.get("dual_review_enabled", False):
            try:
                review = _run_dual_review(user_question, answer, [], user=user)
                _save_evaluation(review)
            except Exception:
                pass
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

    # ── 双层评审（仅在开关开启时）──
    toggles = _load_toggles()
    if toggles.get("dual_review_enabled", False):
        try:
            review = _run_dual_review(user_question, answer, valid_data, user=user)
            _save_evaluation(review)
        except Exception:
            pass

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
