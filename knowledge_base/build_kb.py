#!/usr/bin/env python3
"""
履约知识库生成脚本
从 Olist 数据集生成 4 张 CSV，供 AI 购物助手工具层调用。

关联规则（与 story1/story2/story3_provenance.py 完全一致）：
- 每订单取 order_item_id 最小的那条（first item per order）
- 只统计 order_status == "delivered" 且 delivery_days > 0 的订单
- 差评 = review_score ≤ 2
- 退货词 = devol/troca/reembolso/cancel/estorn（小写匹配 review_comment_message）
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "olist"
OUT = Path(__file__).resolve().parent

RETURN_KW = ["devol", "troca", "reembolso", "cancel", "estorn"]


def load_data():
    """加载并预处理，关联规则与 story*_provenance.py 一致"""
    orders = pd.read_csv(BASE / "olist_orders_dataset.csv")
    items = pd.read_csv(BASE / "olist_order_items_dataset.csv")
    products = pd.read_csv(BASE / "olist_products_dataset.csv")
    sellers = pd.read_csv(BASE / "olist_sellers_dataset.csv")
    reviews = pd.read_csv(BASE / "olist_order_reviews_dataset.csv")
    customers = pd.read_csv(BASE / "olist_customers_dataset.csv")
    trans = pd.read_csv(BASE / "product_category_name_translation.csv")

    # ── 1. 筛选已送达订单 + 配送天数 > 0（与 story1 一致）──
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(
        orders["order_delivered_customer_date"]
    )
    delivered = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
    ].copy()
    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    delivered = delivered[delivered["delivery_days"] > 0]

    # ── 2. 每订单取 order_item_id 最小的那条（first item per order）──
    first_item = (
        items.sort_values("order_item_id")
        .groupby("order_id")
        .first()
        .reset_index()[["order_id", "seller_id", "product_id", "price", "freight_value"]]
    )

    # ── 3. 关联产品类目（翻译优先，原名兜底）──
    products = products.merge(trans, on="product_category_name", how="left")
    products["category_en"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    )

    # ── 4. 组装主表（delivered × first_item × products × sellers × customers）──
    df = (
        delivered[["order_id", "customer_id", "delivery_days"]]
        .merge(first_item, on="order_id", how="inner")
        .merge(products[["product_id", "category_en"]], on="product_id", how="left")
        .merge(sellers[["seller_id", "seller_city", "seller_state"]], on="seller_id", how="left")
        .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    )

    # ── 5. 关联评论（一个订单可能有多条，去重取第一条，与 story3 一致）──
    order_reviews = reviews.drop_duplicates(subset="order_id")[
        ["order_id", "review_score", "review_comment_message"]
    ]
    df = df.merge(order_reviews, on="order_id", how="left")

    # ── 6. 标记字段 ──
    df["is_neg"] = df["review_score"].fillna(999) <= 2
    df["has_return_kw"] = (
        df["review_comment_message"]
        .fillna("")
        .str.lower()
        .apply(lambda x: any(kw in x for kw in RETURN_KW))
    )
    df["total_price"] = df["price"] + df["freight_value"]

    return df


def build_route_timing(df):
    """① route_timing.csv — 卖家州×买家州（仅 delivered + days>0）"""
    grp = df.groupby(["seller_state", "customer_state"])
    result = grp.agg(
        n=("order_id", "count"),
        mean_days=("delivery_days", "mean"),
        median_days=("delivery_days", "median"),
        p90_days=("delivery_days", lambda x: x.quantile(0.9)),
        pct_in_10d=("delivery_days", lambda x: (x <= 10).mean() * 100),
    ).reset_index()

    result = result.round({"mean_days": 2, "median_days": 2, "p90_days": 2, "pct_in_10d": 2})
    result.to_csv(OUT / "route_timing.csv", index=False)
    return result


def build_seller_risk(df):
    """② seller_risk.csv — 按类目卖家级（基线来自已送达订单）"""
    # 类目退货词基线
    cat_return = df.groupby("category_en")["has_return_kw"].mean().rename("cat_avg_return_kw")

    grp = df.groupby(["seller_id", "seller_city", "seller_state", "category_en"])
    result = grp.agg(
        n_reviews=("review_score", lambda x: x.notna().sum()),
        neg_rate=("is_neg", "mean"),
        return_kw_rate=("has_return_kw", "mean"),
    ).reset_index()

    result["neg_rate"] = (result["neg_rate"] * 100).round(2)
    result["return_kw_rate"] = (result["return_kw_rate"] * 100).round(2)

    cat_return_pct = (cat_return * 100).round(2).reset_index()
    cat_return_pct.columns = ["category_en", "cat_avg_return_kw"]
    result = result.merge(cat_return_pct, on="category_en", how="left")

    result.to_csv(OUT / "seller_risk.csv", index=False)
    return result


def build_seller_cost(df):
    """③ seller_cost.csv — 卖家×类目×买家州"""
    grp = df.groupby(["seller_id", "category_en", "customer_state"])
    result = grp.agg(
        n=("order_id", "count"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        avg_total=("total_price", "mean"),
        median_days=("delivery_days", "median"),
        neg_rate=("is_neg", "mean"),
    ).reset_index()

    result["avg_price"] = result["avg_price"].round(2)
    result["avg_freight"] = result["avg_freight"].round(2)
    result["avg_total"] = result["avg_total"].round(2)
    result["median_days"] = result["median_days"].round(2)
    result["neg_rate"] = (result["neg_rate"] * 100).round(2)

    result.to_csv(OUT / "seller_cost.csv", index=False)
    return result


def build_category_baseline(df):
    """④ category_baseline.csv — 每类目基线"""
    grp = df.groupby("category_en")
    result = grp.agg(
        n_orders=("order_id", "count"),
        n_reviews=("review_score", lambda x: x.notna().sum()),
        neg_rate=("is_neg", "mean"),
        return_kw_rate=("has_return_kw", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
    ).reset_index()

    result["neg_rate"] = (result["neg_rate"] * 100).round(2)
    result["return_kw_rate"] = (result["return_kw_rate"] * 100).round(2)
    result["avg_delivery_days"] = result["avg_delivery_days"].round(2)

    result.to_csv(OUT / "category_baseline.csv", index=False)
    return result


def main():
    print("加载数据（关联规则与 story*_provenance.py 一致）...")
    df = load_data()
    print(f"主表行数: {len(df)}（已送达 + 配送天数>0 + 每单首商品）")

    print("\n--- ① route_timing.csv ---")
    rt = build_route_timing(df)
    print(f"行数: {len(rt)}")
    print(rt.head(3).to_string(index=False))

    print("\n--- ② seller_risk.csv ---")
    sr = build_seller_risk(df)
    print(f"行数: {len(sr)}")
    print(sr.head(3).to_string(index=False))

    print("\n--- ③ seller_cost.csv ---")
    sc = build_seller_cost(df)
    print(f"行数: {len(sc)}")
    print(sc.head(3).to_string(index=False))

    print("\n--- ④ category_baseline.csv ---")
    cb = build_category_baseline(df)
    print(f"行数: {len(cb)}")
    print(cb.head(3).to_string(index=False))

    # ── 验证 SP→RN 应该 = 332 ──
    sp_rn = rt[(rt["seller_state"] == "SP") & (rt["customer_state"] == "RN")]
    if len(sp_rn) == 1:
        n_val = int(sp_rn["n"].iloc[0])
        pct_val = sp_rn["pct_in_10d"].iloc[0]
        print(f"\n✅ 验证: SP→RN = {n_val} 单, 10天内占比 = {pct_val:.0f}%")
        assert n_val == 332, f"❌ 期望 332，实际 {n_val}"
        print("✅ 与 story1_provenance.py 完全一致！")

    print("\n履约知识库生成完毕，输出目录:", OUT)


if __name__ == "__main__":
    main()
