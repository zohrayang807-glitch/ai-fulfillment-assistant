#!/usr/bin/env python3
"""
履约知识库查询接口
从 knowledge_base/ 的 CSV 读取数据，供 AI 购物助手工具层调用。

每个函数做层级回退：精确匹配不到时，退到更粗粒度。
"""

import pandas as pd
from pathlib import Path

KB = Path(__file__).resolve().parent

# ── 懒加载 delivery_vs_promise ──
_delivery_promise = None


def _load_delivery_promise():
    global _delivery_promise
    if _delivery_promise is None:
        _delivery_promise = pd.read_csv(KB / "delivery_vs_promise.csv")
    return _delivery_promise


# ── 懒加载 category_baseline ──
_category_baseline = None


def _load_category_baseline():
    global _category_baseline
    if _category_baseline is None:
        _category_baseline = pd.read_csv(KB / "category_baseline.csv")
    return _category_baseline

# ── 懒加载 review_reason ──
_review_reason = None


def _load_review_reason():
    global _review_reason
    if _review_reason is None:
        _review_reason = pd.read_csv(KB / "review_reason.csv")
    return _review_reason


# ── 懒加载：首次调用时读 CSV，之后复用 ──
_route_timing = None
_seller_risk = None
_seller_cost = None


def _load_route_timing():
    global _route_timing
    if _route_timing is None:
        _route_timing = pd.read_csv(KB / "route_timing.csv")
    return _route_timing


def _load_seller_risk():
    global _seller_risk
    if _seller_risk is None:
        _seller_risk = pd.read_csv(KB / "seller_risk.csv")
    return _seller_risk


def _load_seller_cost():
    global _seller_cost
    if _seller_cost is None:
        _seller_cost = pd.read_csv(KB / "seller_cost.csv")
    return _seller_cost


def query_timing(seller_state, buyer_state):
    """
    查询某路线配送时效分布。

    返回 dict: {n, median_days, p90_days, pct_in_10d}
    查不到返回 None。

    回退逻辑：
    1. 精确匹配：seller_state × buyer_state
    2. 回退：同 buyer_state 的全卖家平均
    """
    df = _load_route_timing()

    # 精确匹配
    row = df[(df["seller_state"] == seller_state) & (df["customer_state"] == buyer_state)]
    if len(row) > 0 and row["n"].iloc[0] > 0:
        r = row.iloc[0]
        return {
            "n": int(r["n"]),
            "median_days": r["median_days"],
            "p90_days": r["p90_days"],
            "pct_in_10d": r["pct_in_10d"],
            "source": f"route {seller_state}→{buyer_state}",
        }

    # 回退：同买家州全卖家平均
    fallback = df[df["customer_state"] == buyer_state]
    if len(fallback) > 0:
        n_total = fallback["n"].sum()
        if n_total > 0:
            weighted_median = (fallback["median_days"] * fallback["n"]).sum() / n_total
            weighted_p90 = (fallback["p90_days"] * fallback["n"]).sum() / n_total
            weighted_pct = (fallback["pct_in_10d"] * fallback["n"]).sum() / n_total
            return {
                "n": int(n_total),
                "median_days": round(weighted_median, 2),
                "p90_days": round(weighted_p90, 2),
                "pct_in_10d": round(weighted_pct, 2),
                "source": f"all sellers→{buyer_state} (fallback)",
            }

    return None


def query_promise(seller_state, buyer_state):
    """
    查询承诺 vs 实际偏差（route 粒度）。

    返回 dict: {n, avg_promise, avg_actual, avg_gap, ontime_rate, source}
    查不到返回 None。

    回退逻辑：
    1. 精确匹配：seller_state × buyer_state
    2. 回退：同 buyer_state 全部（all→buyer_state）
    """
    df = _load_delivery_promise()

    # 精确匹配
    row = df[(df["seller_state"] == seller_state) & (df["customer_state"] == buyer_state)]
    if len(row) > 0 and row["n"].iloc[0] > 0:
        r = row.iloc[0]
        return {
            "n": int(r["n"]),
            "avg_promise": r["avg_promise"],
            "avg_actual": r["avg_actual"],
            "avg_gap": r["avg_gap"],
            "ontime_rate": r["ontime_rate"],
            "source": f"route {seller_state}→{buyer_state}",
        }

    # 回退：同买家州全部
    fallback = df[df["customer_state"] == buyer_state]
    if len(fallback) > 0:
        n_total = fallback["n"].sum()
        if n_total > 0:
            w = fallback["n"]
            return {
                "n": int(n_total),
                "avg_promise": round((fallback["avg_promise"] * w).sum() / n_total, 2),
                "avg_actual": round((fallback["avg_actual"] * w).sum() / n_total, 2),
                "avg_gap": round((fallback["avg_gap"] * w).sum() / n_total, 2),
                "ontime_rate": round((fallback["ontime_rate"] * w).sum() / n_total, 4),
                "source": f"all sellers→{buyer_state} (fallback)",
            }

    return None


def _unique_seller(df, seller_id):
    """前缀匹配后做唯一性检查。返回 (mask, error_msg)。"""
    if len(seller_id) < 40:
        mask = df["seller_id"].str.startswith(seller_id)
        matched = df.loc[mask, "seller_id"].unique()
        if len(matched) == 0:
            return mask, "no_match"
        if len(matched) > 1:
            return mask, f"卖家前缀 {seller_id} 不唯一，匹配到 {len(matched)} 个卖家，请提供更长的 ID。"
        return mask, None
    else:
        mask = df["seller_id"] == seller_id
        if mask.sum() == 0:
            return mask, "no_match"
        return mask, None


def query_seller_state(seller_id):
    """从 seller_risk.csv 反查卖家发货州。返回 (seller_state, error_msg)。"""
    df = _load_seller_risk()
    mask, err = _unique_seller(df, seller_id)
    if err == "no_match":
        return None, f"未找到卖家 {seller_id}"
    if err:
        return None, err
    row = df[mask].iloc[0]
    return row.get("seller_state"), None


def query_seller_risk(seller_id, category=None):
    """
    查询卖家退货/差评风险信号。

    返回 dict: {n_reviews, neg_rate, return_kw_rate, cat_avg_return_kw}
    查不到返回 None。

    两种粒度：
    1. 有 category → 查该卖家在该品类的表现
    2. 无 category → 查该卖家所有品类聚合的表现
    """
    df = _load_seller_risk()

    seller_mask, err = _unique_seller(df, seller_id)
    if err == "no_match":
        return None
    if err:
        return {"error": err}

    # ── 有品类：精确匹配 seller × category ──
    if category:
        mask = seller_mask & (df["category_en"] == category)
        row = df[mask]
        if len(row) > 0 and row["n_reviews"].iloc[0] > 0:
            r = row.iloc[0]
            return {
                "n_reviews": int(r["n_reviews"]),
                "neg_rate": r["neg_rate"],
                "return_kw_rate": r["return_kw_rate"],
                "cat_avg_return_kw": r["cat_avg_return_kw"],
                "source": f"seller {seller_id[:10]}.. / {category}",
            }
        # 回退：同 category 全卖家平均
        fallback = df[df["category_en"] == category]
        if len(fallback) > 0:
            n_reviews = fallback["n_reviews"].sum()
            if n_reviews > 0:
                weighted_neg = (fallback["neg_rate"] * fallback["n_reviews"]).sum() / n_reviews
                weighted_kw = (fallback["return_kw_rate"] * fallback["n_reviews"]).sum() / n_reviews
                cat_avg = fallback["cat_avg_return_kw"].mean()
                return {
                    "n_reviews": int(n_reviews),
                    "neg_rate": round(weighted_neg, 2),
                    "return_kw_rate": round(weighted_kw, 2),
                    "cat_avg_return_kw": round(cat_avg, 2),
                    "source": f"all sellers / {category} (fallback)",
                }
        return None

    # ── 无品类：聚合该卖家所有品类 ──
    seller_rows = df[seller_mask]
    if len(seller_rows) == 0:
        return None

    n_reviews = seller_rows["n_reviews"].sum()
    if n_reviews == 0:
        return None

    weighted_neg = (seller_rows["neg_rate"] * seller_rows["n_reviews"]).sum() / n_reviews
    weighted_kw = (seller_rows["return_kw_rate"] * seller_rows["n_reviews"]).sum() / n_reviews
    cat_avg = seller_rows["cat_avg_return_kw"].mean()
    cats = seller_rows["category_en"].nunique()

    return {
        "n_reviews": int(n_reviews),
        "neg_rate": round(weighted_neg, 2),
        "return_kw_rate": round(weighted_kw, 2),
        "cat_avg_return_kw": round(cat_avg, 2),
        "n_categories": cats,
        "source": f"seller {seller_id[:10]}.. / all {cats} categories (aggregated)",
    }


def query_seller_categories(seller_id):
    """
    查询卖家跨品类表现：该卖家覆盖了哪些品类，每个品类的差评率相对基线如何。

    返回 dict:
      {categories: [{category_en, n_reviews, neg_rate, baseline_neg, vs_baseline}, ...],
       best: {category_en, neg_rate, ...},
       worst: {category_en, neg_rate, ...}}
    查不到返回 None。
    """
    df = _load_seller_risk()
    baseline = _load_category_baseline()

    seller_mask, err = _unique_seller(df, seller_id)
    if err == "no_match":
        return None
    if err:
        return {"error": err}

    rows = df[seller_mask].copy()
    if rows.empty:
        return None

    # 合并基线
    base_map = baseline.set_index("category_en")["neg_rate"].to_dict()
    cats = []
    for _, r in rows.iterrows():
        cat = r["category_en"]
        bl = base_map.get(cat)
        vs = None
        if bl is not None:
            if r["neg_rate"] < bl * 0.7:
                vs = "好于平均"
            elif r["neg_rate"] > bl * 1.3:
                vs = "差于平均"
            else:
                vs = "接近平均"
        cats.append({
            "category_en": cat,
            "n_reviews": int(r["n_reviews"]),
            "neg_rate": round(r["neg_rate"], 2),
            "baseline_neg": round(bl, 2) if bl is not None else None,
            "vs_baseline": vs,
        })

    # 按 neg_rate 排序
    cats.sort(key=lambda x: x["neg_rate"])
    best = cats[0]
    worst = cats[-1]

    return {
        "seller_id": seller_id,
        "n_categories": len(cats),
        "categories": cats,
        "best": best,
        "worst": worst,
    }


def query_review_reason(category=None, seller_id=None):
    """
    查询差评原因 top3。

    两种模式：
    1. 品类模式（category 有值）→ 返回该品类差评原因 top3
    2. 卖家模式（seller_id 有值）→ 返回该卖家差评原因 top3
       若卖家差评样本太少（<10 条），标注 "样本少" 并回退到该卖家主要品类

    返回 dict: {reasons: [{reason, n, pct}, ...], total: int, note?: str}
    查不到返回 None。
    """
    df = _load_review_reason()

    if seller_id:
        # ── 卖家模式 ──
        seller_mask, err = _unique_seller(df, seller_id)
        if err == "no_match":
            return None
        if err:
            return {"error": err}

        sub = df[seller_mask]
        if sub.empty:
            return None

        total = sub["n"].sum()

        # 样本太少 → 尝试回退到该卖家主要品类
        if total < 10:
            # 找该卖家在 review_reason 中出现最多的品类
            cat_total = sub.groupby("category_en")["n"].sum().sort_values(ascending=False)
            if cat_total.empty:
                return {"reasons": [], "total": int(total), "note": "差评样本太少，无法归纳主要原因"}
            main_cat = cat_total.index[0]
            # 回退到品类模式
            cat_result = query_review_reason(category=main_cat)
            if cat_result:
                cat_result["note"] = f"卖家差评仅 {total} 条（样本少），以下为该卖家主要品类 {main_cat} 的差评原因"
                return cat_result
            return {"reasons": [], "total": int(total), "note": "差评样本太少，无法归纳主要原因"}

        # 正常：按 reason 聚合 top3
        agg = sub.groupby("reason")["n"].sum().sort_values(ascending=False).head(3)
        reasons = []
        for reason, n in agg.items():
            reasons.append({"reason": reason, "n": int(n), "pct": round(n / total * 100, 1)})

        return {"reasons": reasons, "total": int(total)}

    if category:
        # ── 品类模式 ──
        sub = df[df["category_en"] == category]
        if sub.empty:
            return None

        total = sub["n"].sum()
        agg = sub.groupby("reason")["n"].sum().sort_values(ascending=False).head(3)
        reasons = []
        for reason, n in agg.items():
            reasons.append({"reason": reason, "n": int(n), "pct": round(n / total * 100, 1)})

        return {"reasons": reasons, "total": int(total), "category": category}

    return None


def query_cost(seller_id, category, buyer_state):
    """
    查询卖家到手价（标价+运费）。

    返回 dict: {n, avg_price, avg_freight, avg_total, median_days, neg_rate}
    查不到返回 None。

    回退逻辑：
    1. 精确匹配：seller_id × category × buyer_state
    2. 回退：seller_id × category（全买家州平均）
    3. 再回退：category × buyer_state（全卖家平均）
    """
    df = _load_seller_cost()

    # 精确匹配（支持前缀，含唯一性检查）
    seller_mask, err = _unique_seller(df, seller_id)
    if err == "no_match":
        return None
    if err:
        return {"error": err}

    mask = seller_mask & (df["category_en"] == category) & (df["customer_state"] == buyer_state)
    row = df[mask]
    if len(row) > 0 and row["n"].iloc[0] > 0:
        r = row.iloc[0]
        return {
            "n": int(r["n"]),
            "avg_price": r["avg_price"],
            "avg_freight": r["avg_freight"],
            "avg_total": r["avg_total"],
            "median_days": r["median_days"],
            "neg_rate": r["neg_rate"],
            "source": f"seller {seller_id[:10]}.. / {category} / {buyer_state}",
        }

    # 回退 1：seller × category（全买家州）
    mask2 = seller_mask & (df["category_en"] == category)
    row2 = df[mask2]
    if len(row2) > 0 and row2["n"].sum() > 0:
        n_total = row2["n"].sum()
        return {
            "n": int(n_total),
            "avg_price": round((row2["avg_price"] * row2["n"]).sum() / n_total, 2),
            "avg_freight": round((row2["avg_freight"] * row2["n"]).sum() / n_total, 2),
            "avg_total": round((row2["avg_total"] * row2["n"]).sum() / n_total, 2),
            "median_days": round((row2["median_days"] * row2["n"]).sum() / n_total, 2),
            "neg_rate": round((row2["neg_rate"] * row2["n"]).sum() / n_total, 2),
            "source": f"seller {seller_id[:10]}.. / {category} / all states (fallback)",
        }

    # 回退 2：category × buyer_state（全卖家）
    mask3 = (df["category_en"] == category) & (df["customer_state"] == buyer_state)
    row3 = df[mask3]
    if len(row3) > 0 and row3["n"].sum() > 0:
        n_total = row3["n"].sum()
        return {
            "n": int(n_total),
            "avg_price": round((row3["avg_price"] * row3["n"]).sum() / n_total, 2),
            "avg_freight": round((row3["avg_freight"] * row3["n"]).sum() / n_total, 2),
            "avg_total": round((row3["avg_total"] * row3["n"]).sum() / n_total, 2),
            "median_days": round((row3["median_days"] * row3["n"]).sum() / n_total, 2),
            "neg_rate": round((row3["neg_rate"] * row3["n"]).sum() / n_total, 2),
            "source": f"all sellers / {category} / {buyer_state} (fallback)",
        }

    return None


def query_cost_baseline(category, buyer_state=None):
    """
    品类级价格基线（无需指定卖家）。

    从 seller_cost.csv 聚合该品类的平均价格/运费/到手价。
    有 buyer_state → 品类×买家州；否则 → 品类全量。

    Returns: {avg_price, avg_freight, avg_total, n, scope}
    """
    df = _load_seller_cost()
    sub = df[df["category_en"] == category]
    if sub.empty:
        return None

    if buyer_state:
        sub2 = sub[sub["customer_state"] == buyer_state]
        n = sub2["n"].sum()
        if n >= 10:
            w = sub2["n"]
            return {
                "avg_price": round((sub2["avg_price"] * w).sum() / n, 2),
                "avg_freight": round((sub2["avg_freight"] * w).sum() / n, 2),
                "avg_total": round((sub2["avg_total"] * w).sum() / n, 2),
                "n": int(n),
                "scope": f"{category} → {buyer_state}",
            }

    # 回退：品类全量
    n_total = sub["n"].sum()
    if n_total > 0:
        w = sub["n"]
        return {
            "avg_price": round((sub["avg_price"] * w).sum() / n_total, 2),
            "avg_freight": round((sub["avg_freight"] * w).sum() / n_total, 2),
            "avg_total": round((sub["avg_total"] * w).sum() / n_total, 2),
            "n": int(n_total),
            "scope": f"{category} → all states",
        }

    return None


def query_value_score(seller_ids, category, buyer_state, weights=None):
    """
    多卖家性价比综合评分：价格(越低越好) + 时效(越低越好) + 风险(越低越好)。

    对每个卖家取 avg_total / median_days / neg_rate，
    在卖家集合内做 min-max 归一化，加权求和。

    Returns: list[dict] 按 value_score 降序，或 error string。
    """
    if len(seller_ids) < 2:
        return "需要至少两个卖家才能做性价比对比。"

    if weights is None:
        weights = {"price": 0.4, "time": 0.3, "risk": 0.3}

    # 收集每个卖家的三维数据
    sellers_data = []
    for sid in seller_ids:
        cost = query_cost(sid, category, buyer_state)
        if cost is None:
            sellers_data.append({
                "seller_id": sid, "avg_total": None, "median_days": None,
                "neg_rate": None, "missing": "该卖家在该品类无价格数据",
            })
            continue

        # 时效：先查发货州再查 timing
        median_days = cost.get("median_days")
        if median_days is None:
            ss, _ = query_seller_state(sid)
            if ss:
                timing = query_timing(ss, buyer_state)
                if timing:
                    median_days = timing.get("median_days")

        # 风险：从 seller_risk 取
        risk = query_seller_risk(sid, category)
        neg_rate = risk.get("neg_rate") if risk else None

        sellers_data.append({
            "seller_id": sid,
            "avg_total": cost.get("avg_total"),
            "median_days": median_days,
            "neg_rate": neg_rate,
            "neg_rate_source": "seller_risk",  # 标注来源，避免与 query_cost 返回的 neg_rate 混淆
        })

    # 评分
    valid = [s for s in sellers_data if s.get("avg_total") is not None]
    if len(valid) < 2:
        return "有效数据不足，无法做性价比对比。"

    def ratio_score(values):
        """
        相对比例评分：score = min_value / value。
        越小越好的维度，min 的那个得 1.0，其他按比例递减。
        保留绝对差距（如价格差 3.7 倍 → 0.27 vs 1.0）。
        value <= 0 时按中性 0.5 处理。
        """
        mn = min(v for v in values if v > 0) if any(v > 0 for v in values) else 0
        if mn == 0:
            return [0.5] * len(values), 0
        scores = []
        for v in values:
            if v <= 0:
                scores.append(0.5)
            else:
                scores.append(round(mn / v, 4))
        return scores, mn

    # 提取有效值（None 用中性 0.5 填充）
    totals = [s["avg_total"] for s in valid]
    days = [s["median_days"] if s["median_days"] is not None else None for s in valid]
    negs = [s["neg_rate"] if s["neg_rate"] is not None else None for s in valid]

    # 价格评分（越低越好）
    price_scores, min_price = ratio_score(totals)

    # 时效评分（越低越好）— 缺失值用 0.5
    days_clean = [d if d is not None else None for d in days]
    days_valid = [d for d in days_clean if d is not None]
    if days_valid:
        day_scores_all, min_days = ratio_score(days_valid)
        day_scores = []
        idx = 0
        for d in days_clean:
            if d is not None:
                day_scores.append(day_scores_all[idx])
                idx += 1
            else:
                day_scores.append(0.5)
    else:
        day_scores = [0.5] * len(valid)
        min_days = 0

    # 风险评分（越低越好）— 缺失值用 0.5
    negs_clean = [n if n is not None else None for n in negs]
    negs_valid = [n for n in negs_clean if n is not None]
    if negs_valid:
        neg_scores_all, min_neg = ratio_score(negs_valid)
        neg_scores = []
        idx = 0
        for n in negs_clean:
            if n is not None:
                neg_scores.append(neg_scores_all[idx])
                idx += 1
            else:
                neg_scores.append(0.5)
    else:
        neg_scores = [0.5] * len(valid)
        min_neg = 0

    # 加权求和 + 写入 min 值供回答层解释差距
    for i, s in enumerate(valid):
        s["price_score"] = round(price_scores[i], 4)
        s["time_score"] = round(day_scores[i], 4)
        s["risk_score"] = round(neg_scores[i], 4)
        s["value_score"] = round(
            weights["price"] * price_scores[i]
            + weights["time"] * day_scores[i]
            + weights["risk"] * neg_scores[i],
            4,
        )
        # 标注各维度基准值（便于回答解释差距）
        s["min_price"] = round(min_price, 2)
        s["min_days"] = round(min_days, 2) if min_days else None
        s["min_neg"] = round(min_neg, 2) if min_neg else None
        # 标注缺失维度
        missing_dims = []
        if s.get("median_days") is None:
            missing_dims.append("时效")
        if s.get("neg_rate") is None:
            missing_dims.append("风险")
        if missing_dims:
            s["missing_dims"] = missing_dims

    valid.sort(key=lambda x: x["value_score"], reverse=True)
    return valid


def query_freight_estimate(category, buyer_state=None):
    """
    品类级运费参考（无需指定卖家）。

    从 seller_cost.csv 聚合该品类的平均运费。
    有 buyer_state 且样本足 → 品类×买家州
    否则 → 品类全量

    Returns: {avg_freight, n, scope, note}
    """
    df = _load_seller_cost()
    cat_rows = df[df["category_en"] == category]
    if cat_rows.empty:
        return None

    note = "运费还受重量、距离、物流商影响，这是品类参考值。"

    # 优先：品类×买家州
    if buyer_state:
        sub = cat_rows[cat_rows["customer_state"] == buyer_state]
        n = sub["n"].sum() if "n" in sub.columns else len(sub)
        if n >= 10:
            w = sub["n"]
            avg_freight = round((sub["avg_freight"] * w).sum() / w.sum(), 2)
            return {
                "avg_freight": avg_freight,
                "n": int(n),
                "scope": f"{category} → {buyer_state}",
                "note": note,
            }

    # 回退：品类全量
    n_total = cat_rows["n"].sum() if "n" in cat_rows.columns else len(cat_rows)
    if n_total > 0:
        w = cat_rows["n"]
        avg_freight = round((cat_rows["avg_freight"] * w).sum() / w.sum(), 2)
        return {
            "avg_freight": avg_freight,
            "n": int(n_total),
            "scope": f"{category} → all states",
            "note": note,
        }

    return None


def query_recommend(category: str, buyer_state: str = None, top_n: int = 3):
    """
    推荐：同一品类中口碑好且有一定订单量的卖家。
    过滤条件：
      - 品类精确匹配
      - n_reviews ≥ 10（保证统计意义）
      - neg_rate < 该品类基线 neg_rate（好于平均水平）
    排序：neg_rate 升序（好评越多越靠前）
    附加：如有 buyer_state，通过 query_timing 追加每个卖家到该州的时效。

    Returns: list[dict] 或 error string
    """
    df = _load_seller_risk()
    baseline = _load_category_baseline()

    # 品类基线 neg_rate
    base_row = baseline[baseline["category_en"] == category]
    if base_row.empty:
        return f"未找到该品类：{category}"
    base_neg = base_row.iloc[0]["neg_rate"]

    # 筛选该品类、有足够评价、好于基线
    sub = df[(df["category_en"] == category) & (df["n_reviews"] >= 10)]
    if sub.empty:
        return f"品类 {category} 暂无有足够评价量的卖家数据。"

    good = sub[sub["neg_rate"] < base_neg].sort_values("neg_rate").head(top_n)

    if good.empty:
        return f"品类 {category} 暂无明显好于平均水平的卖家。"

    results = []
    for _, row in good.iterrows():
        entry = {
            "seller_id": row["seller_id"],
            "neg_rate": round(row["neg_rate"], 4),
            "n_reviews": int(row["n_reviews"]),
        }
        results.append(entry)

    # 附加时效（先反查卖家发货州，再查时效）
    if buyer_state:
        for entry in results:
            seller_state, err = query_seller_state(entry["seller_id"])
            if not seller_state:
                continue  # 反查失败（未找到/前缀不唯一），跳过时效，不阻塞推荐
            timing = query_timing(seller_state, buyer_state)
            if isinstance(timing, dict):
                entry["median_days"] = timing.get("median_days")
                entry["p90_days"] = timing.get("p90_days")
                entry["timing_source"] = timing.get("source", f"{seller_state}→{buyer_state}")

    return results


# ── 测试 ──
if __name__ == "__main__":
    print("=" * 50)
    print("  测试 query_timing('SP', 'RN')")
    print("=" * 50)
    r = query_timing("SP", "RN")
    print(r)
    assert r is not None, "返回 None"
    assert r["n"] == 332, f"期望 n=332，实际 n={r['n']}"
    print(f"✅ n={r['n']}, median={r['median_days']}, p90={r['p90_days']}, 10d={r['pct_in_10d']}%")

    print()
    print("=" * 50)
    print("  测试 query_timing 回退（不存在的卖家州）")
    print("=" * 50)
    r2 = query_timing("XX", "RN")
    print(r2)

    print()
    print("=" * 50)
    print("  测试 query_timing 返回 None（不存在的买家州）")
    print("=" * 50)
    r3 = query_timing("SP", "XX")
    print(r3)
