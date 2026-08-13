"""故事 2 验证脚本 — 办公家具退货投诉分析"""

import pandas as pd

DATA_DIR = "/Users/yangzhidong/ai-portfolio/olist"

# ── 读取数据 ─────────────────────────────────────────────────
reviews = pd.read_csv(f"{DATA_DIR}/olist_order_reviews_dataset.csv")
order_items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv")
sellers = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")
cat_trans = pd.read_csv(f"{DATA_DIR}/product_category_name_translation.csv")
orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")

# 只取已送达订单（与 build_kb 一致）
delivered_ids = orders[orders["order_status"] == "delivered"]["order_id"]

# ── 关联规则 ─────────────────────────────────────────────────
# 只取已送达订单的 items（与 build_kb 一致）
delivered_items = order_items[order_items["order_id"].isin(delivered_ids)]
# 每个订单取 order_item_id 最小的那条，得到 seller_id 和 product_id
first_item = (
    delivered_items.sort_values("order_item_id")
    .groupby("order_id")
    .first()
    .reset_index()[["order_id", "seller_id", "product_id"]]
)

# 评论 → 订单 → 首商品 → 商品类目（翻译为英文）
products_en = products.merge(cat_trans, on="product_category_name", how="left")
products_en["category_en"] = products_en["product_category_name_english"].fillna(
    products_en["product_category_name"]
)

# 评论去重（一个订单只取一条评论，与 build_kb 一致）+ 只取已送达订单
order_reviews = reviews.drop_duplicates(subset="order_id")[
    ["order_id", "review_score", "review_comment_message"]
]
order_reviews = order_reviews[order_reviews["order_id"].isin(delivered_ids)]
df = order_reviews.copy()
df = df.merge(first_item, on="order_id", how="left")
df = df.merge(products_en[["product_id", "category_en"]], on="product_id", how="left")
df = df.merge(sellers[["seller_id", "seller_city", "seller_state"]], on="seller_id", how="left")

# ── 过滤：办公家具 ──────────────────────────────────────────
df = df[df["category_en"] == "office_furniture"].copy()

# ── 标记 ────────────────────────────────────────────────────
df["is_low"] = df["review_score"] <= 2
msg = df["review_comment_message"].fillna("").str.lower()
df["has_return_kw"] = msg.str.contains(r"devol|troca|reembol|cancel|estorn", regex=True)

# ── 输出 ① 类目整体 ─────────────────────────────────────────
n_total = len(df)
n_low = df["is_low"].sum()
n_low_return = (df["is_low"] & df["has_return_kw"]).sum()

print("=" * 60)
print("  ① 类目整体（office_furniture）")
print("=" * 60)
print(f"  总评论数 (n):             {n_total:,}")
print(f"  差评数 (n):               {n_low:,}")
print(f"  差评率:                   {n_low/n_total*100:.1f}%")
print(f"  差评含退货词评论数 (n):   {n_low_return:,}")
print(f"  占总评论比例:             {n_low_return/n_total*100:.2f}%")

# ── 输出 ② 问题卖家 a7f13822ce ──────────────────────────────
bad = df[df["seller_id"].str.startswith("a7f13822ce")].copy()
n_bad = len(bad)
n_bad_low = bad["is_low"].sum()
n_bad_ret = (bad["is_low"] & bad["has_return_kw"]).sum()

print("\n" + "=" * 60)
print("  ② 问题卖家（a7f13822ce..）")
print("=" * 60)
print(f"  城市/州:                  {bad['seller_city'].iloc[0]}/{bad['seller_state'].iloc[0]}")
print(f"  总评论数 (n):             {n_bad}")
print(f"  差评数 (n):               {n_bad_low}")
print(f"  差评率:                   {n_bad_low/n_bad*100:.1f}%")
print(f"  差评含退货词占比:         {n_bad_ret/n_bad_low*100:.1f}%")

# ── 输出 ③ 推荐卖家 f8db351d ────────────────────────────────
good = df[df["seller_id"].str.startswith("f8db351d")].copy()
n_good = len(good)
n_good_low = good["is_low"].sum()
n_good_ret = (good["is_low"] & good["has_return_kw"]).sum()

print("\n" + "=" * 60)
print("  ③ 推荐卖家（f8db351d..）")
print("=" * 60)
print(f"  城市/州:                  {good['seller_city'].iloc[0]}/{good['seller_state'].iloc[0]}")
print(f"  总评论数 (n):             {n_good}")
print(f"  差评数 (n):               {n_good_low}")
print(f"  差评率:                   {n_good_low/n_good*100:.1f}%")
if n_good_low > 0:
    print(f"  差评含退货词占比:         {n_good_ret/n_good_low*100:.1f}%")
else:
    print(f"  差评含退货词占比:         N/A（无差评）")

# ── 输出 ④ 对比 ─────────────────────────────────────────────
bad_rate = n_bad_low / n_bad * 100
good_rate = n_good_low / n_good * 100
cat_ret_rate = n_low_return / n_total * 100
# 全量退货词率（与 build_kb 对齐，分母=全部评论，不仅是差评）
cat_kw_rate = df["has_return_kw"].sum() / n_total * 100
bad_kw_rate = bad["has_return_kw"].sum() / n_bad * 100

print("\n" + "=" * 60)
print("  ④ 对比结论")
print("=" * 60)
if good_rate > 0:
    print(f"  问题卖家差评率 {bad_rate:.1f}%，是推荐卖家 {good_rate:.1f}% 的 {bad_rate/good_rate:.1f} 倍")
else:
    print(f"  问题卖家差评率 {bad_rate:.1f}%，推荐卖家差评率 0%（17 笔评论 0 差评）")
print(f"  问题卖家退货词占比(占总评论) {bad_kw_rate:.2f}%，是类目平均 {cat_kw_rate:.2f}% 的 {bad_kw_rate/cat_kw_rate:.1f} 倍")
