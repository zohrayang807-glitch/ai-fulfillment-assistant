#!/usr/bin/env python3
"""
生成 V2 aggregate 操作所需的 3 张聚合表：
  1. category_freight.csv   — 各品类平均运费（加权）
  2. category_ship_time.csv — 各品类平均发货时长（中位数 + 均值）
  3. route_freight.csv      — 各路线平均运费（加权）

输出到 knowledge_base/ 目录。
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OLIST = ROOT / "olist"
KB = ROOT / "knowledge_base"


def build_category_freight():
    """① 各品类平均运费（从 seller_cost.csv 按 category_en 加权平均）"""
    df = pd.read_csv(KB / "seller_cost.csv")
    # 加权平均：sum(avg_freight * n) / sum(n)
    grouped = df.groupby("category_en").apply(
        lambda g: pd.Series({
            "n": g["n"].sum(),
            "avg_freight": np.average(g["avg_freight"], weights=g["n"]),
        })
    ).reset_index()
    grouped = grouped.sort_values("avg_freight", ascending=False)
    out = KB / "category_freight.csv"
    grouped.to_csv(out, index=False)
    print(f"✅ category_freight.csv: {len(grouped)} 个品类")
    print(f"   Top 5: {grouped.head(5)[['category_en','avg_freight','n']].to_string(index=False)}")
    return grouped


def build_category_ship_time():
    """② 各品类平均发货时长（从 olist 原始数据计算）"""
    # 读取 orders
    orders = pd.read_csv(OLIST / "olist_orders_dataset.csv",
                         usecols=["order_id", "order_status",
                                  "order_purchase_timestamp",
                                  "order_delivered_carrier_date"])
    # 读取 order_items（取每个 order 的第一个 item 的 seller 和 product）
    items = pd.read_csv(OLIST / "olist_order_items_dataset.csv",
                        usecols=["order_id", "product_id"])
    # 读取 products
    products = pd.read_csv(OLIST / "olist_products_dataset.csv",
                           usecols=["product_id", "product_category_name"])
    # 读取品类翻译
    trans = pd.read_csv(OLIST / "product_category_name_translation.csv")
    trans.columns = ["product_category_name", "category_en"]

    # 只取已交付且两个时间戳都非空的订单
    orders = orders[orders["order_status"] == "delivered"]
    orders = orders.dropna(subset=["order_purchase_timestamp", "order_delivered_carrier_date"])

    # 合并
    merged = orders.merge(items, on="order_id", how="inner")
    merged = merged.merge(products, on="product_id", how="inner")
    merged = merged.merge(trans, on="product_category_name", how="left")

    # 计算发货时长（天）
    merged["purchase"] = pd.to_datetime(merged["order_purchase_timestamp"])
    merged["carrier"] = pd.to_datetime(merged["order_delivered_carrier_date"])
    merged["ship_days"] = (merged["carrier"] - merged["purchase"]).dt.total_seconds() / 86400

    # 排除负值和异常值（>60天）
    merged = merged[(merged["ship_days"] >= 0) & (merged["ship_days"] <= 60)]

    # 品类为空的用 category_en 填充（已经是英文）
    merged["category_en"] = merged["category_en"].fillna(merged["product_category_name"])

    # 按品类聚合
    grouped = merged.groupby("category_en").agg(
        n=("ship_days", "count"),
        median_days=("ship_days", "median"),
        avg_days=("ship_days", "mean"),
    ).reset_index()

    # 过滤 n >= 30
    grouped = grouped[grouped["n"] >= 30]
    grouped = grouped.sort_values("median_days", ascending=True)
    grouped["median_days"] = grouped["median_days"].round(2)
    grouped["avg_days"] = grouped["avg_days"].round(2)

    out = KB / "category_ship_time.csv"
    grouped.to_csv(out, index=False)
    print(f"✅ category_ship_time.csv: {len(grouped)} 个品类（n≥30）")
    print(f"   最快 5: {grouped.head(5)[['category_en','median_days','n']].to_string(index=False)}")
    print(f"   最慢 5: {grouped.tail(5)[['category_en','median_days','n']].to_string(index=False)}")
    return grouped


def build_route_freight():
    """③ 各路线平均运费（seller_cost × sellers 反查 seller_state，加权平均）"""
    cost = pd.read_csv(KB / "seller_cost.csv")
    sellers = pd.read_csv(OLIST / "olist_sellers_dataset.csv",
                          usecols=["seller_id", "seller_state"])

    # 反查 seller_state
    merged = cost.merge(sellers, on="seller_id", how="left")
    merged = merged.dropna(subset=["seller_state"])

    # 按 (seller_state, customer_state) 加权平均
    grouped = merged.groupby(["seller_state", "customer_state"]).apply(
        lambda g: pd.Series({
            "n": g["n"].sum(),
            "avg_freight": np.average(g["avg_freight"], weights=g["n"]),
        })
    ).reset_index()

    # 过滤 n >= 10
    grouped = grouped[grouped["n"] >= 10]
    grouped = grouped.sort_values("avg_freight", ascending=False)
    grouped["avg_freight"] = grouped["avg_freight"].round(2)

    out = KB / "route_freight.csv"
    grouped.to_csv(out, index=False)
    print(f"✅ route_freight.csv: {len(grouped)} 条路线（n≥10）")
    print(f"   最贵 5: {grouped.head(5)[['seller_state','customer_state','avg_freight','n']].to_string(index=False)}")
    print(f"   最便宜 5: {grouped.tail(5)[['seller_state','customer_state','avg_freight','n']].to_string(index=False)}")
    return grouped


if __name__ == "__main__":
    print("=" * 60)
    print("  生成 V2 聚合表")
    print("=" * 60)
    build_category_freight()
    print()
    build_category_ship_time()
    print()
    build_route_freight()
    print()
    print("✅ 全部完成，CSV 已写入 knowledge_base/")
