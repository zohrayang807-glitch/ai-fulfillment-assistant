#!/usr/bin/env python3
"""
履约知识库查询接口
从 knowledge_base/ 的 CSV 读取数据，供 AI 购物助手工具层调用。

每个函数做层级回退：精确匹配不到时，退到更粗粒度。
"""

import pandas as pd
from pathlib import Path

KB = Path(__file__).resolve().parent

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


def query_seller_risk(seller_id, category):
    """
    查询卖家退货/差评风险信号。

    返回 dict: {n_reviews, neg_rate, return_kw_rate, cat_avg_return_kw}
    查不到返回 None。

    回退逻辑：
    1. 精确匹配：seller_id × category
    2. 回退：同 category 的全卖家平均（cat_avg_return_kw 仍来自 category_baseline）
    """
    df = _load_seller_risk()

    # 精确匹配（支持前缀）
    if len(seller_id) < 40:
        seller_mask = df["seller_id"].str.startswith(seller_id)
    else:
        seller_mask = df["seller_id"] == seller_id
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

    # 精确匹配（支持前缀：10位hex → 匹配所有以此开头的卖家）
    if len(seller_id) < 40:
        seller_mask = df["seller_id"].str.startswith(seller_id)
    else:
        seller_mask = df["seller_id"] == seller_id
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
