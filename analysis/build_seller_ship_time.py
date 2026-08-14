#!/usr/bin/env python3
"""
seller_ship_time.csv 生成脚本
统计每个卖家的发货时效（下单→交给快递 的天数）。

数据源：orders × order_items × sellers
发货天数 = order_delivered_carrier_date − order_purchase_timestamp
只取 order_status='delivered' 且两个时间戳都非空；排除负值。
按 seller_id 聚合：n、median_days、avg_days
过滤 n >= 10

输出：knowledge_base/seller_ship_time.csv
字段：seller_id, seller_city, seller_state, n, median_days, avg_days
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "olist"
OUT = Path(__file__).resolve().parent.parent / "knowledge_base"


def main():
    print("加载 orders + items + sellers ...")
    orders = pd.read_csv(BASE / "olist_orders_dataset.csv")
    items = pd.read_csv(BASE / "olist_order_items_dataset.csv")
    sellers = pd.read_csv(BASE / "olist_sellers_dataset.csv")

    # ── 1. 筛选已送达且两个时间戳都非空 ──
    delivered = orders[
        (orders["order_status"] == "delivered")
        & orders["order_purchase_timestamp"].notna()
        & orders["order_delivered_carrier_date"].notna()
    ][["order_id", "order_purchase_timestamp", "order_delivered_carrier_date"]].copy()

    print(f"有效已送达订单: {len(delivered)}")

    # ── 2. 每订单取 order_item_id 最小的那条（与 build_kb 一致）──
    first_item = (
        items.sort_values("order_item_id")
        .groupby("order_id")
        .first()
        .reset_index()[["order_id", "seller_id"]]
    )

    df = delivered.merge(first_item, on="order_id", how="inner")
    print(f"关联 seller_id 后: {len(df)} 行")

    # ── 3. 计算发货天数 ──
    df["purchase"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["carrier"] = pd.to_datetime(df["order_delivered_carrier_date"])
    df["ship_days"] = (df["carrier"] - df["purchase"]).dt.total_seconds() / 86400

    # 排除负值（数据异常）
    before = len(df)
    df = df[df["ship_days"] >= 0]
    print(f"排除负值 {before - len(df)} 行，剩余 {len(df)}")

    # ── 4. 按 seller_id 聚合 ──
    agg = df.groupby("seller_id").agg(
        n=("ship_days", "count"),
        median_days=("ship_days", "median"),
        avg_days=("ship_days", "mean"),
    ).reset_index()

    # 过滤 n >= 10
    agg_all = len(agg)
    agg = agg[agg["n"] >= 10].copy()
    print(f"卖家总数: {agg_all}，n>=10: {len(agg)}")

    # 四舍五入
    agg["median_days"] = agg["median_days"].round(2)
    agg["avg_days"] = agg["avg_days"].round(2)

    # ── 5. 关联卖家城市/州 ──
    seller_loc = sellers[["seller_id", "seller_city", "seller_state"]]
    agg = agg.merge(seller_loc, on="seller_id", how="left")

    # 字段排序
    agg = agg[["seller_id", "seller_city", "seller_state", "n", "median_days", "avg_days"]]

    # ── 6. 输出 ──
    out_path = OUT / "seller_ship_time.csv"
    agg.to_csv(out_path, index=False)
    print(f"\n✅ 已写入 {out_path}（{len(agg)} 个卖家）")

    # 统计
    print(f"\n── 发货时效分布 ──")
    print(f"median_days: min={agg['median_days'].min()}, median={agg['median_days'].median():.2f}, max={agg['median_days'].max()}")
    print(f"avg_days:    min={agg['avg_days'].min()}, median={agg['avg_days'].median():.2f}, max={agg['avg_days'].max()}")

    # 示例
    print(f"\n── 示例 ──")
    for _, r in agg.head(5).iterrows():
        print(f"  {r['seller_id'][:10]}.. ({r['seller_city']}/{r['seller_state']}): n={r['n']}, median={r['median_days']}天, avg={r['avg_days']}天")


if __name__ == "__main__":
    main()
