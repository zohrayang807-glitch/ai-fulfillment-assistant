#!/usr/bin/env python3
"""
delivery_vs_promise.csv 生成脚本
聚合粒度：seller_state × customer_state（route 粒度，与 route_timing 对齐）
每单算：promise_days, actual_days, ontime
过滤 n >= 10
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "olist"
OUT = Path(__file__).resolve().parent.parent / "knowledge_base"


def main():
    print("加载 orders + sellers + customers ...")
    orders = pd.read_csv(BASE / "olist_orders_dataset.csv")
    items = pd.read_csv(BASE / "olist_order_items_dataset.csv")
    sellers = pd.read_csv(BASE / "olist_sellers_dataset.csv")
    customers = pd.read_csv(BASE / "olist_customers_dataset.csv")

    # ── 1. 筛选已送达 + 有承诺日期 + 有实际送达日期 ──
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

    delivered = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    delivered["promise_days"] = (
        delivered["order_estimated_delivery_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    delivered["actual_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    delivered = delivered[(delivered["promise_days"] > 0) & (delivered["actual_days"] > 0)]
    delivered["ontime"] = (delivered["actual_days"] <= delivered["promise_days"]).astype(int)

    # ── 2. 每订单取 order_item_id 最小的那条 ──
    first_item = (
        items.sort_values("order_item_id")
        .groupby("order_id")
        .first()
        .reset_index()[["order_id", "seller_id"]]
    )

    # ── 3. 关联卖家州 + 买家州 ──
    df = (
        delivered[["order_id", "customer_id", "promise_days", "actual_days", "ontime"]]
        .merge(first_item, on="order_id", how="inner")
        .merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
        .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    )

    print(f"主表行数: {len(df)}（已送达 + 有承诺 + 有实际 + 配送天数>0）")

    # ── 4. 按 route 聚合 ──
    grp = df.groupby(["seller_state", "customer_state"])
    result = grp.agg(
        n=("order_id", "count"),
        avg_promise=("promise_days", "mean"),
        avg_actual=("actual_days", "mean"),
        ontime_rate=("ontime", "mean"),
    ).reset_index()

    result["avg_gap"] = result["avg_promise"] - result["avg_actual"]
    result = result[result["n"] >= 10]
    result = result.round({"avg_promise": 2, "avg_actual": 2, "avg_gap": 2, "ontime_rate": 4})
    result = result[["seller_state", "customer_state", "n", "avg_promise", "avg_actual", "avg_gap", "ontime_rate"]]

    out_path = OUT / "delivery_vs_promise.csv"
    result.to_csv(out_path, index=False)
    print(f"\n输出: {out_path}")
    print(f"行数: {len(result)}")
    print(result.head(5).to_string(index=False))

    # ── 验证 SP→SP ──
    sp_sp = result[(result["seller_state"] == "SP") & (result["customer_state"] == "SP")]
    if len(sp_sp) == 1:
        r = sp_sp.iloc[0]
        print(f"\n✅ 验证 SP→SP: n={int(r['n'])}, avg_promise={r['avg_promise']}, avg_actual={r['avg_actual']}, ontime_rate={r['ontime_rate']}")
        assert 15 <= r["avg_promise"] <= 20, f"avg_promise 异常: {r['avg_promise']}"
        assert 5 <= r["avg_actual"] <= 10, f"avg_actual 异常: {r['avg_actual']}"
        assert 0.8 <= r["ontime_rate"] <= 1.0, f"ontime_rate 异常: {r['ontime_rate']}"
        print("✅ 验证通过！")

    print("\ndelivery_vs_promise.csv 生成完毕。")


if __name__ == "__main__":
    main()
