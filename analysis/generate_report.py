"""Olist 巴西电商数据集 — 生成完整 Markdown 分析报告"""

import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = "/Users/yangzhidong/ai-portfolio/olist"
REPORT_PATH = f"{DATA_DIR}/REPORT.md"

# ── 读取数据 ─────────────────────────────────────────────────
orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")
order_items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
sellers = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")
products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv")
cat_trans = pd.read_csv(f"{DATA_DIR}/product_category_name_translation.csv")

# ── 数据清洗 ─────────────────────────────────────────────────
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

delivered = orders[
    (orders["order_status"] == "delivered")
    & orders["order_delivered_customer_date"].notna()
    & orders["order_estimated_delivery_date"].notna()
].copy()

delivered["delivery_days"] = (
    delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400
delivered["promise_diff"] = (
    delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400
delivered = delivered[delivered["delivery_days"] > 0].copy()

def classify(diff):
    if diff < -1: return "提前送达"
    elif diff <= 1: return "准时送达"
    else: return "延迟送达"

delivered["promise_status"] = delivered["promise_diff"].apply(classify)

# 关联卖家州
seller_per_order = order_items.groupby("order_id")["seller_id"].first().reset_index()
delivered = delivered.merge(seller_per_order, on="order_id", how="left")
delivered = delivered.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")

# 关联商品类目
item_detail = order_items[["order_id", "product_id"]].copy()
item_detail = item_detail.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
item_detail = item_detail.merge(cat_trans, on="product_category_name", how="left")
item_detail["category"] = item_detail["product_category_name_english"].fillna(item_detail["product_category_name"])
first_item = item_detail.groupby("order_id").first().reset_index()
delivered = delivered.merge(first_item[["order_id", "category"]], on="order_id", how="left")

total = len(delivered)

# ══════════════════════════════════════════════════════════════
#  生成 Markdown 报告
# ══════════════════════════════════════════════════════════════

lines = []
w = lines.append

w("# Olist 巴西电商 — 配送时效分析报告")
w(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
w(f"> 有效订单数: **{total:,}** 笔\n")
w("---\n")

# ── 第一部分：配送时长 ───────────────────────────────────────
w("## 1. 配送时长概览\n")
w("| 指标 | 值 |")
w("|---|---|")
w(f"| 平均配送天数 | {delivered['delivery_days'].mean():.1f} 天 |")
w(f"| P50（中位数） | {delivered['delivery_days'].quantile(0.50):.1f} 天 |")
w(f"| P75 | {delivered['delivery_days'].quantile(0.75):.1f} 天 |")
w(f"| P90 | {delivered['delivery_days'].quantile(0.90):.1f} 天 |")

bins_time = [0, 3, 7, float("inf")]
labels_time = ["1-3 天", "4-7 天", "7 天以上"]
delivered["delivery_bucket"] = pd.cut(delivered["delivery_days"], bins=bins_time, labels=labels_time)
dist_time = delivered["delivery_bucket"].value_counts().reindex(labels_time)

w("\n### 配送时长分布\n")
w("| 区间 | 订单数 | 占比 |")
w("|---|---|---|")
for label in labels_time:
    cnt = dist_time[label]
    pct = cnt / total * 100
    w(f"| {label} | {cnt:,} | {pct:.1f}% |")

# ── 第二部分：各州分位数 ─────────────────────────────────────
state_pct = (
    delivered.groupby("seller_state")["delivery_days"]
    .agg(订单数="count", 平均值="mean",
         P50=lambda x: x.quantile(0.50),
         P75=lambda x: x.quantile(0.75),
         P90=lambda x: x.quantile(0.90))
    .sort_values("P50")
    .round(1)
)

w("\n## 2. 各州配送天数分位数\n")
w("| 州 | 订单数 | 平均值 | P50 | P75 | P90 | P90-P50 |")
w("|---|---|---|---|---|---|---|")
for state, row in state_pct.iterrows():
    spread = row["P90"] - row["P50"]
    w(f"| {state} | {int(row['订单数']):,} | {row['平均值']:.1f} | {row['P50']:.1f} | {row['P75']:.1f} | {row['P90']:.1f} | {spread:.1f} |")

w("\n**最快 5 州（P50）:** " + ", ".join(
    f"{s} ({r['P50']:.1f}天)" for s, r in state_pct.head(5).iterrows()
))
w("\n**最慢 5 州（P50）:** " + ", ".join(
    f"{s} ({r['P50']:.1f}天)" for s, r in state_pct.tail(5).iloc[::-1].iterrows()
))

# ── 第三部分：承诺准确度 ─────────────────────────────────────
dist_promise = delivered["promise_status"].value_counts().reindex(["提前送达", "准时送达", "延迟送达"])

w("\n## 3. 承诺准确度（实际 vs 预计送达）\n")
w("| 类别 | 订单数 | 占比 | 平均偏差 |")
w("|---|---|---|---|")
for status in ["提前送达", "准时送达", "延迟送达"]:
    subset = delivered[delivered["promise_status"] == status]
    cnt = len(subset)
    pct = cnt / total * 100
    avg = subset["promise_diff"].mean()
    w(f"| {status} | {cnt:,} | {pct:.1f}% | {avg:+.1f} 天 |")

# ── 第四部分：偏差分布偏斜 ───────────────────────────────────
mean_diff = delivered["promise_diff"].mean()
median_diff = delivered["promise_diff"].median()
skew_ratio = abs(mean_diff - median_diff) / abs(median_diff) * 100

w("\n## 4. 时效偏差分布：中位数 vs 平均值\n")
w("| 指标 | 值 |")
w("|---|---|")
w(f"| 平均值 | {mean_diff:+.1f} 天 |")
w(f"| 中位数 | {median_diff:+.1f} 天 |")
w(f"| 标准差 | {delivered['promise_diff'].std():.1f} 天 |")
w(f"| 差异百分比 | {skew_ratio:.1f}% |")

if skew_ratio > 30:
    w(f"\n> ⚠️ **分布偏斜警报**：中位数与平均值差异 {skew_ratio:.1f}%（>30%），存在极端值拉偏。")
else:
    w(f"\n> ✅ 分布对称：中位数与平均值差异 {skew_ratio:.1f}%（≤30%），均值可靠。")

# 各州偏斜
state_skew = (
    delivered.groupby("seller_state")["promise_diff"]
    .agg(订单数="count", 平均值="mean", 中位数="median")
    .query("订单数 >= 50")
    .assign(差异百分比=lambda df: ((df["平均值"] - df["中位数"]).abs() / df["中位数"].abs() * 100).round(1))
    .assign(判断=lambda df: df["差异百分比"].apply(lambda x: "⚠️ 偏斜" if x > 30 else "✅ 正常"))
    .sort_values("差异百分比", ascending=False)
)

w("\n### 各州偏差偏斜检查（≥50 笔）\n")
w("| 州 | 订单数 | 平均值 | 中位数 | 差异% | 判断 |")
w("|---|---|---|---|---|---|")
for state, row in state_skew.iterrows():
    w(f"| {state} | {int(row['订单数']):,} | {row['平均值']:+.1f} | {row['中位数']:+.1f} | {row['差异百分比']:.1f}% | {row['判断']} |")

# ── 第五部分：类目延迟率 ─────────────────────────────────────
cat_stats = (
    delivered.groupby("category")
    .agg(total_orders=("order_id", "count"),
         late_rate=("promise_status", lambda x: (x == "延迟送达").mean() * 100),
         avg_diff=("promise_diff", "mean"))
    .query("total_orders >= 50")
    .sort_values("late_rate", ascending=False)
)

w("\n## 5. 商品类目延迟率排行\n")
w("### 延迟率最高的类目（≥50 笔订单）\n")
w("| 类目 | 订单数 | 延迟率 | 平均偏差 |")
w("|---|---|---|---|")
for cat, row in cat_stats.head(10).iterrows():
    w(f"| {cat} | {int(row['total_orders']):,} | {row['late_rate']:.1f}% | {row['avg_diff']:+.1f} 天 |")

w("\n### 延迟率最低的类目\n")
w("| 类目 | 订单数 | 延迟率 | 平均偏差 |")
w("|---|---|---|---|")
for cat, row in cat_stats.tail(5).iloc[::-1].iterrows():
    w(f"| {cat} | {int(row['total_orders']):,} | {row['late_rate']:.1f}% | {row['avg_diff']:+.1f} 天 |")

# ── 第六部分：缺失值 ─────────────────────────────────────────
missing = orders.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

w("\n## 6. 原始数据缺失值\n")
w("| 列名 | 缺失数 | 占比 |")
w("|---|---|---|")
for col, cnt in missing.items():
    pct = cnt / len(orders) * 100
    w(f"| {col} | {cnt:,} | {pct:.2f}% |")

w("\n---")
w("\n*分析脚本: `analyze_delivery.py` / `analyze_promise_accuracy.py` / `analyze_percentiles.py`*")

# ── 写入文件 ─────────────────────────────────────────────────
report = "\n".join(lines)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"✅ 报告已生成: {REPORT_PATH}")
print(f"   共 {len(lines)} 行")
