#!/usr/bin/env python3
"""
履约知识库生成脚本
从 Olist 数据集生成 4 张 CSV，供 AI 购物助手工具层调用。

关联规则（与 story*_provenance.py 一致）：
- 每订单取 order_item_id 最小的那条
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
    """加载并预处理所有表"""
    orders = pd.read_csv(BASE / "olist_orders_dataset.csv")
    items = pd.read_csv(BASE / "olist_order_items_dataset.csv")
    products = pd.read_csv(BASE / "olist_products_dataset.csv")
    sellers = pd.read_csv(BASE / "olist_sellers_dataset.csv")
    reviews = pd.read_csv(BASE / "olist_order_reviews_dataset.csv")
    customers = pd.read_csv(BASE / "olist_customers_dataset.csv")

    # 翻译类目名
    trans = pd.read_csv(BASE / "product_category_name_translation.csv")
    products = products.merge(trans, on="product_category_name", how="left")
    products["category_en"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    )

    # 配送天数
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(
        orders["order_delivered_customer_date"]
    )
    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # 合并主表
    df = (
        items[["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"]]
        .merge(orders[["order_id", "customer_id", "order_status", "delivery_days"]], on="order_id", how="inner")
        .merge(products[["product_id", "category_en"]], on="product_id", how="left")
        .merge(sellers[["seller_id", "seller_city", "seller_state"]], on="seller_id", how="left")
        .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
        .merge(reviews[["order_id", "review_score", "review_comment_message"]], on="order_id", how="left")
    )

    # 每订单取 order_item_id 最小的那条
    df = df.sort_values("order_item_id").groupby("order_id", as_index=False).first()

    # 差评标记
    df["is_neg"] = df["review_score"].fillna(999) <= 2

    # 退货词标记
    df["has_return_kw"] = (
        df["review_comment_message"]
        .fillna("")
        .str.lower()
        .apply(lambda x: any(kw in x for kw in RETURN_KW))
    )

    # 到手价
    df["total_price"] = df["price"] + df["freight_value"]

    return df


def build_route_timing(df):
    """① route_timing.csv — 卖家州×买家州"""
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
    """② seller_risk.csv — 按类目卖家级"""
    # 退货词基线：按类目
    cat_return = df.groupby("category_en")["has_return_kw"].mean().rename("cat_avg_return_kw")

    grp = df.groupby(["seller_id", "seller_city", "seller_state", "category_en"])
    result = grp.agg(
        n_reviews=("review_score", lambda x: x.notna().sum()),
        neg_rate=("is_neg", "mean"),
        return_kw_rate=("has_return_kw", "mean"),
    ).reset_index()

    result["neg_rate"] = (result["neg_rate"] * 100).round(2)
    result["return_kw_rate"] = (result["return_kw_rate"] * 100).round(2)

    # 合并类目基线
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
    print("加载数据...")
    df = load_data()
    print(f"主表行数: {len(df)}")

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

    print("\n✅ 履约知识库生成完毕，输出目录:", OUT)


if __name__ == "__main__":
    main()
