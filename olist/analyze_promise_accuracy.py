"""Olist 巴西电商数据集 — 承诺配送准确性分析"""

import pandas as pd

DATA_DIR = "/Users/yangzhidong/ai-portfolio/olist"

# ── 1. 读取数据 ──────────────────────────────────────────────
orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")
order_items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
sellers = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")
products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv")
cat_trans = pd.read_csv(f"{DATA_DIR}/product_category_name_translation.csv")

# ── 2. 转换日期 & 筛选有效订单 ───────────────────────────────
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

delivered = orders[
    (orders["order_status"] == "delivered")
    & orders["order_delivered_customer_date"].notna()
    & orders["order_estimated_delivery_date"].notna()
].copy()

# 承诺偏差 = 实际送达 - 承诺日期（正 = 延迟，负 = 提前）
delivered["promise_diff_days"] = (
    delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

# 分类：提前(≤-1天) / 准时(±1天) / 延迟(≥1天)
def classify(diff):
    if diff < -1:
        return "提前送达"
    elif diff <= 1:
        return "准时送达"
    else:
        return "延迟送达"

delivered["promise_status"] = delivered["promise_diff_days"].apply(classify)

total = len(delivered)
print(f"有效订单数: {total:,}\n")

# ── 3. 整体承诺偏差分布 ─────────────────────────────────────
dist = delivered["promise_status"].value_counts().reindex(["提前送达", "准时送达", "延迟送达"])
print("=" * 55)
print("  整体承诺偏差分布（实际送达 vs 承诺日期）")
print("=" * 55)
print(f"{'类别':<12} {'订单数':>10} {'占比':>8} {'平均偏差(天)':>14}")
print("-" * 55)
for status in ["提前送达", "准时送达", "延迟送达"]:
    subset = delivered[delivered["promise_status"] == status]
    cnt = len(subset)
    pct = cnt / total * 100
    avg = subset["promise_diff_days"].mean()
    print(f"  {status:<10} {cnt:>10,} {pct:>7.1f}% {avg:>12.1f}")

overall_avg = delivered["promise_diff_days"].mean()
print(f"\n  整体平均偏差: {overall_avg:.1f} 天（负值 = 平均提前送达）")
print(f"  中位数偏差:   {delivered['promise_diff_days'].median():.1f} 天")

# ── 4. 关联商品类目 & 卖家州 ─────────────────────────────────
# order_items → product_id → product_category_name → 英文类目
item_detail = order_items[["order_id", "product_id", "seller_id"]].copy()
item_detail = item_detail.merge(
    products[["product_id", "product_category_name"]], on="product_id", how="left"
)
item_detail = item_detail.merge(
    cat_trans, on="product_category_name", how="left"
)
# 用英文名，缺失的用葡文名兜底
item_detail["category"] = (
    item_detail["product_category_name_english"]
    .fillna(item_detail["product_category_name"])
)

# 关联卖家州
item_detail = item_detail.merge(
    sellers[["seller_id", "seller_state"]], on="seller_id", how="left"
)

# 合并到 delivered（一笔订单取第一个商品的类目和州）
first_item = item_detail.groupby("order_id").first().reset_index()
delivered = delivered.merge(first_item[["order_id", "category", "seller_state"]], on="order_id", how="left")

# ── 5. 按商品类目统计延迟率 ─────────────────────────────────
cat_stats = (
    delivered.groupby("category")
    .agg(
        total_orders=("order_id", "count"),
        late_rate=("promise_status", lambda x: (x == "延迟送达").mean() * 100),
        avg_diff=("promise_diff_days", "mean"),
    )
    .query("total_orders >= 50")  # 至少 50 笔才有统计意义
    .sort_values("late_rate", ascending=False)
)

print("\n" + "=" * 70)
print("  承诺最不准的商品类目（延迟率最高，≥50 笔订单）")
print("=" * 70)
print(f"{'类目':<35} {'订单数':>8} {'延迟率':>8} {'平均偏差(天)':>14}")
print("-" * 70)
for cat, row in cat_stats.head(10).iterrows():
    print(f"  {cat:<33} {int(row['total_orders']):>8,} {row['late_rate']:>7.1f}% {row['avg_diff']:>12.1f}")

print("\n" + "=" * 70)
print("  承诺最准的商品类目（延迟率最低，≥50 笔订单）")
print("=" * 70)
print(f"{'类目':<35} {'订单数':>8} {'延迟率':>8} {'平均偏差(天)':>14}")
print("-" * 70)
for cat, row in cat_stats.tail(5).iloc[::-1].iterrows():
    print(f"  {cat:<33} {int(row['total_orders']):>8,} {row['late_rate']:>7.1f}% {row['avg_diff']:>12.1f}")

# ── 6. 按卖家州统计延迟率 ───────────────────────────────────
state_stats = (
    delivered.groupby("seller_state")
    .agg(
        total_orders=("order_id", "count"),
        late_rate=("promise_status", lambda x: (x == "延迟送达").mean() * 100),
        avg_diff=("promise_diff_days", "mean"),
    )
    .sort_values("late_rate", ascending=False)
)

print("\n" + "=" * 70)
print("  按卖家州统计 — 延迟率排行")
print("=" * 70)
print(f"{'州':<6} {'订单数':>8} {'延迟率':>8} {'平均偏差(天)':>14}  {'评价'}")
print("-" * 70)
for state, row in state_stats.iterrows():
    if row["late_rate"] >= 20:
        tag = "⚠️  较差"
    elif row["late_rate"] >= 10:
        tag = "🔶 一般"
    else:
        tag = "✅ 良好"
    print(f"  {state:<4} {int(row['total_orders']):>8,} {row['late_rate']:>7.1f}% {row['avg_diff']:>12.1f}  {tag}")
