"""Olist 巴西电商数据集 — 时效偏差分布 & 分位数分析"""

import pandas as pd
import numpy as np

DATA_DIR = "/Users/yangzhidong/ai-portfolio/olist"

# ── 1. 读取 & 清洗 ──────────────────────────────────────────
orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")
order_items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
sellers = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

delivered = orders[
    (orders["order_status"] == "delivered")
    & orders["order_delivered_customer_date"].notna()
    & orders["order_estimated_delivery_date"].notna()
].copy()

# 实际配送天数
delivered["delivery_days"] = (
    delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400
delivered = delivered[delivered["delivery_days"] > 0].copy()

# 时效偏差 = 承诺日期 − 实际送达（正 = 提前，负 = 延迟）
delivered["promise_diff"] = (
    delivered["order_estimated_delivery_date"] - delivered["order_delivered_customer_date"]
).dt.total_seconds() / 86400

# 关联卖家州
seller_per_order = order_items.groupby("order_id")["seller_id"].first().reset_index()
delivered = delivered.merge(seller_per_order, on="order_id", how="left")
delivered = delivered.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")

total = len(delivered)
print(f"有效订单数: {total:,}\n")

# ══════════════════════════════════════════════════════════════
#  2. 时效偏差：中位数 vs 平均值
# ══════════════════════════════════════════════════════════════
mean_diff = delivered["promise_diff"].mean()
median_diff = delivered["promise_diff"].median()
std_diff = delivered["promise_diff"].std()
skew_ratio = abs(mean_diff - median_diff) / abs(median_diff) * 100

print("=" * 65)
print("  ① 时效偏差统计（承诺日期 − 实际送达，正=提前，负=延迟）")
print("=" * 65)
print(f"  平均值:   {mean_diff:>8.1f} 天")
print(f"  中位数:   {median_diff:>8.1f} 天")
print(f"  标准差:   {std_diff:>8.1f} 天")
print(f"  最小值:   {delivered['promise_diff'].min():>8.1f} 天")
print(f"  最大值:   {delivered['promise_diff'].max():>8.1f} 天")
print()

if skew_ratio > 30:
    print(f"  ⚠️  中位数与平均值差异 {skew_ratio:.1f}%（>30%），分布明显偏斜")
    direction = "右偏（少量极端延迟订单拉高均值）" if mean_diff > median_diff else "左偏"
    print(f"     方向: {direction}")
else:
    print(f"  ✅ 中位数与平均值差异 {skew_ratio:.1f}%（≤30%），分布相对对称")

# ══════════════════════════════════════════════════════════════
#  3. 实际配送天数 — 整体分位数
# ══════════════════════════════════════════════════════════════
p50 = delivered["delivery_days"].quantile(0.50)
p75 = delivered["delivery_days"].quantile(0.75)
p90 = delivered["delivery_days"].quantile(0.90)

print("\n" + "=" * 65)
print("  ② 实际配送天数 — 整体分位数")
print("=" * 65)
print(f"  平均值:   {delivered['delivery_days'].mean():>8.1f} 天")
print(f"  P50:      {p50:>8.1f} 天  （一半订单在此天数内送达）")
print(f"  P75:      {p75:>8.1f} 天  （75% 订单在此天数内送达）")
print(f"  P90:      {p90:>8.1f} 天  （90% 订单在此天数内送达）")

# ══════════════════════════════════════════════════════════════
#  4. 各州分位数
# ══════════════════════════════════════════════════════════════
state_pct = (
    delivered.groupby("seller_state")["delivery_days"]
    .agg(
        订单数="count",
        平均值="mean",
        P50=lambda x: x.quantile(0.50),
        P75=lambda x: x.quantile(0.75),
        P90=lambda x: x.quantile(0.90),
    )
    .sort_values("P50")
    .round(1)
)

print("\n" + "=" * 75)
print("  ③ 各卖家州 — 配送天数分位数")
print("=" * 75)
print(f"{'州':<5} {'订单数':>7} {'平均值':>7} {'P50':>7} {'P75':>7} {'P90':>7}  {'P90-P50':>8}")
print("-" * 75)
for state, row in state_pct.iterrows():
    spread = row["P90"] - row["P50"]
    print(
        f"  {state:<3} {int(row['订单数']):>7,} "
        f"{row['平均值']:>7.1f} {row['P50']:>7.1f} {row['P75']:>7.1f} {row['P90']:>7.1f}"
        f"  {spread:>7.1f}"
    )

# ══════════════════════════════════════════════════════════════
#  5. 各州偏差中位数 vs 平均值
# ══════════════════════════════════════════════════════════════
state_skew = (
    delivered.groupby("seller_state")["promise_diff"]
    .agg(订单数="count", 平均值="mean", 中位数="median")
    .query("订单数 >= 50")
    .assign(差异百分比=lambda df: ((df["平均值"] - df["中位数"]).abs() / df["中位数"].abs() * 100).round(1))
    .assign(是否偏斜=lambda df: df["差异百分比"].apply(lambda x: "⚠️ 偏斜" if x > 30 else "✅ 正常"))
    .sort_values("差异百分比", ascending=False)
)

print("\n" + "=" * 75)
print("  ④ 各州时效偏差 — 中位数 vs 平均值（≥50 笔订单）")
print("=" * 75)
print(f"{'州':<5} {'订单数':>7} {'平均值':>8} {'中位数':>8} {'差异%':>7}  {'判断'}")
print("-" * 75)
for state, row in state_skew.iterrows():
    print(
        f"  {state:<3} {int(row['订单数']):>7,} "
        f"{row['平均值']:>8.1f} {row['中位数']:>8.1f} {row['差异百分比']:>6.1f}%  {row['是否偏斜']}"
    )
