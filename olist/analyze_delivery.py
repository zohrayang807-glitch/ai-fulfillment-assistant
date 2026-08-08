"""Olist 巴西电商数据集 — 配送时长分析"""

import pandas as pd

DATA_DIR = "/Users/yangzhidong/ai-portfolio/olist"

# ── 1. 读取数据 ──────────────────────────────────────────────
orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")
order_items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
sellers = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")

# ── 2. 转换日期 & 计算配送时长 ───────────────────────────────
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])

orders["delivery_days"] = (
    orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400  # 秒 → 天

# 只保留已完成配送且时长为正的订单
delivered = orders[(orders["order_status"] == "delivered") & (orders["delivery_days"] > 0)].copy()
print(f"有效配送订单数: {len(delivered):,}\n")

# ── 3. 关联卖家所在州 ───────────────────────────────────────
# order_items 有 seller_id（一笔订单可能多个商品/卖家，取第一个）
seller_per_order = order_items.groupby("order_id")["seller_id"].first().reset_index()
delivered = delivered.merge(seller_per_order, on="order_id", how="left")
delivered = delivered.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")

# ── 4. 按州汇总平均配送时长 ──────────────────────────────────
state_stats = (
    delivered.groupby("seller_state")["delivery_days"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "avg_days", "count": "orders"})
    .sort_values("avg_days")
)

print("=" * 55)
print("  按卖家州汇总 — 平均配送时长（天）")
print("=" * 55)
print(f"{'州':<6} {'平均天数':>10} {'订单数':>10}")
print("-" * 55)
for state, row in state_stats.iterrows():
    print(f"{state:<6} {row['avg_days']:>10.2f} {int(row['orders']):>10,}")

print("\n── 最快 5 个州 ──")
fast5 = state_stats.head(5)
for state, row in fast5.iterrows():
    print(f"  {state}  →  {row['avg_days']:.2f} 天  ({int(row['orders']):,} 笔)")

print("\n── 最慢 5 个州 ──")
slow5 = state_stats.tail(5).iloc[::-1]
for state, row in slow5.iterrows():
    print(f"  {state}  →  {row['avg_days']:.2f} 天  ({int(row['orders']):,} 笔)")

# ── 5. 配送时长分布 ─────────────────────────────────────────
bins = [0, 3, 7, float("inf")]
labels = ["1-3 天", "4-7 天", "7 天以上"]
delivered["delivery_bucket"] = pd.cut(delivered["delivery_days"], bins=bins, labels=labels)

dist = delivered["delivery_bucket"].value_counts().reindex(labels)
total = dist.sum()

print("\n" + "=" * 40)
print("  配送时长分布")
print("=" * 40)
for label in labels:
    cnt = dist[label]
    pct = cnt / total * 100
    print(f"  {label:<10}  {cnt:>8,} 笔  ({pct:>5.1f}%)")

# ── 6. 缺失值报告 ───────────────────────────────────────────
missing = orders.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\n" + "=" * 50)
print("  olist_orders_dataset.csv 缺失值报告")
print("=" * 50)
for col, cnt in missing.items():
    pct = cnt / len(orders) * 100
    print(f"  {col:<40} {cnt:>7,} 条  ({pct:>5.2f}%)")
