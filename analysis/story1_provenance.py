"""验证四个关键结论的独立脚本"""

import pandas as pd

DATA_DIR = "/Users/yangzhidong/ai-portfolio/olist"

# ── 公共数据加载 ──────────────────────────────────────────────
orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")
order_items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
sellers = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")
customers = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset.csv")
products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv")
cat_trans = pd.read_csv(f"{DATA_DIR}/product_category_name_translation.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])

delivered = orders[
    (orders["order_status"] == "delivered")
    & orders["order_delivered_customer_date"].notna()
].copy()
delivered["actual_days"] = (
    delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400
delivered = delivered[delivered["actual_days"] > 0]

delivered = delivered.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
seller_per_order = order_items.groupby("order_id")["seller_id"].first().reset_index()
delivered = delivered.merge(seller_per_order, on="order_id", how="left")
delivered = delivered.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")

item_detail = order_items[["order_id", "product_id"]].copy()
item_detail = item_detail.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
item_detail = item_detail.merge(cat_trans, on="product_category_name", how="left")
item_detail["category"] = item_detail["product_category_name_english"].fillna(item_detail["product_category_name"])
first_item = item_detail.groupby("order_id").first().reset_index()
delivered = delivered.merge(first_item[["order_id", "category"]], on="order_id", how="left")

audio = delivered[delivered["category"] == "audio"]

# ══════════════════════════════════════════════════════════════
#  结论 1：SP 占比 75%
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("  结论 1：SP 占比 75%")
print("=" * 60)

audio_seller_by_state = audio.groupby("seller_state")["seller_id"].nunique().sort_values(ascending=False)
total_audio_sellers = audio_seller_by_state.sum()
sp_sellers = audio_seller_by_state.get("SP", 0)
pct = sp_sellers / total_audio_sellers * 100

print(f"  audio 卖家总数: {total_audio_sellers}")
print(f"  SP 卖家数:      {sp_sellers}")
print(f"  SP 占比:        {pct:.1f}%")
print()
print("  各州分布:")
for state, cnt in audio_seller_by_state.items():
    print(f"    {state}: {cnt} 个 ({cnt/total_audio_sellers*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
#  结论 2：SP→RN 均值 20.5 / 中位数 17.8
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  结论 2：SP→RN 实际配送 均值 20.5 / 中位数 17.8")
print("=" * 60)

sp_to_rn = delivered[
    (delivered["seller_state"] == "SP") & (delivered["customer_state"] == "RN")
]
n = len(sp_to_rn)
mean_val = sp_to_rn["actual_days"].mean()
median_val = sp_to_rn["actual_days"].median()

pct_10 = (sp_to_rn["actual_days"] <= 10).mean() * 100
cnt_10 = (sp_to_rn["actual_days"] <= 10).sum()

print(f"  订单数:      {n}")
print(f"  均值:        {mean_val:.1f} 天")
print(f"  中位数:      {median_val:.1f} 天")
print(f"  10 天内到货: {pct_10:.0f}%（{cnt_10} 单）")

# ══════════════════════════════════════════════════════════════
#  结论 3：RN 本地 5.0 天 / 100%
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  结论 3：RN 本地配送 中位数 5.0 天 / 10 天内 100%")
print("=" * 60)

rn_local = delivered[
    (delivered["seller_state"] == "RN") & (delivered["customer_state"] == "RN")
]
n = len(rn_local)
median_val = rn_local["actual_days"].median()
pct_10 = (rn_local["actual_days"] <= 10).mean() * 100

print(f"  订单数:      {n}")
print(f"  均值:        {rn_local['actual_days'].mean():.1f} 天")
print(f"  中位数:      {median_val:.1f} 天")
print(f"  10 天内到货: {pct_10:.0f}%")

# ══════════════════════════════════════════════════════════════
#  结论 4：PE→RN 7.2 天 / 75% / 2 个卖家
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  结论 4：PE→RN 中位数 7.2 天 / 10 天内 75% / 2 个卖家")
print("=" * 60)

pe_to_rn = delivered[
    (delivered["seller_state"] == "PE") & (delivered["customer_state"] == "RN")
]
n = len(pe_to_rn)
median_val = pe_to_rn["actual_days"].median()
mean_val = pe_to_rn["actual_days"].mean()
pct_10 = (pe_to_rn["actual_days"] <= 10).mean() * 100
n_sellers = pe_to_rn["seller_id"].nunique()

print(f"  订单数:        {n}")
print(f"  均值:          {mean_val:.1f} 天")
print(f"  中位数:        {median_val:.1f} 天")
print(f"  10 天内到货:   {pct_10:.0f}%")
print(f"  卖家数:        {n_sellers}")
