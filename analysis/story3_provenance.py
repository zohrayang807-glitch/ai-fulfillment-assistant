"""
story3_provenance.py — 故事3 数据复现：手表/礼品品类，两家卖家对比

关联规则（修正版）：
  1) 一个订单只要包含该卖家的 relogios_presentes 商品，就计入该卖家的样本
  2) 价格和运费取该订单中该卖家手表商品的值（不取"每单第一条"）
  3) product_id 关联 products 表得到 product_category_name
  4) 只统计类目 relogios_presentes（手表/礼品）
  5) 只看两个卖家：seller_id 以 b33e7c5544 或 d650b663c3 开头
  6) 差评 = review_score <= 2

输出：
  ① 全量买家对比
  ② 买家=SP 对比
  ③ 结论语句
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'olist')

# ---------- 加载 ----------
orders = pd.read_csv(os.path.join(DATA_DIR, 'olist_orders_dataset.csv'))
order_items = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_items_dataset.csv'))
products = pd.read_csv(os.path.join(DATA_DIR, 'olist_products_dataset.csv'))
reviews = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_reviews_dataset.csv'))
customers = pd.read_csv(os.path.join(DATA_DIR, 'olist_customers_dataset.csv'))

# ---------- 新关联规则：保留所有 order_items 行 ----------
# 关联产品类目
items = order_items.merge(
    products[['product_id', 'product_category_name']],
    on='product_id', how='left'
)

# 筛选：类目 = relogios_presentes
items = items[items['product_category_name'] == 'relogios_presentes'].copy()

# 筛选：两个目标卖家
TARGET_SELLERS = ['b33e7c5544', 'd650b663c3']
items = items[items['seller_id'].str.startswith(tuple(TARGET_SELLERS))].copy()
items['seller_short'] = items['seller_id'].str[:10]

# 关联订单时间（算配送天数）+ customer_id
items = items.merge(
    orders[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date',
            'customer_id']],
    on='order_id', how='left'
)

# 关联买家州
items = items.merge(
    customers[['customer_id', 'customer_state']],
    on='customer_id', how='left'
)

# 关联评论（一个订单可能有多条评论，去重取第一条）
order_reviews = reviews.drop_duplicates(subset='order_id')[['order_id', 'review_score']]
items = items.merge(order_reviews, on='order_id', how='left')

# ---------- 计算字段 ----------
items['order_purchase_timestamp'] = pd.to_datetime(items['order_purchase_timestamp'])
items['order_delivered_customer_date'] = pd.to_datetime(items['order_delivered_customer_date'])
items['delivery_days'] = (
    (items['order_delivered_customer_date'] - items['order_purchase_timestamp'])
    .dt.total_seconds() / 86400
)
items['total_price'] = items['price'] + items['freight_value']
items['is_bad'] = items['review_score'] <= 2


def seller_stats(df, label):
    """计算单个卖家的统计指标

    差评率 = 差评数 ÷ 该卖家手表订单的评论数（分母是有评论的订单数，不是总订单数）
    """
    n = len(df)
    avg_price = df['price'].mean()
    avg_freight = df['freight_value'].mean()
    avg_total = df['total_price'].mean()
    median_delivery = df['delivery_days'].dropna().median()
    # 有评论的订单
    reviewed = df[df['review_score'].notna()]
    bad_count = (reviewed['review_score'] <= 2).sum()
    n_reviewed = len(reviewed)
    bad_rate = bad_count / n_reviewed * 100 if n_reviewed > 0 else 0
    return {
        'label': label,
        'n': n,
        'avg_price': avg_price,
        'avg_freight': avg_freight,
        'avg_total': avg_total,
        'median_delivery': median_delivery,
        'bad_count': int(bad_count),
        'bad_rate': bad_rate,
    }


def print_comparison(stats_a, stats_b, title):
    """打印两个卖家的对比表"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'指标':<20} {stats_a['label']:>20} {stats_b['label']:>20}")
    print(f"  {'-' * 60}")
    rows = [
        ('订单数 (n)',       f"{stats_a['n']}",           f"{stats_b['n']}"),
        ('平均标价',         f"R${stats_a['avg_price']:.0f}",  f"R${stats_b['avg_price']:.0f}"),
        ('平均运费',         f"R${stats_a['avg_freight']:.0f}", f"R${stats_b['avg_freight']:.0f}"),
        ('平均到手总价',     f"R${stats_a['avg_total']:.0f}",   f"R${stats_b['avg_total']:.0f}"),
        ('配送天数中位数',   f"{stats_a['median_delivery']:.1f} 天", f"{stats_b['median_delivery']:.1f} 天"),
        ('差评数',           f"{stats_a['bad_count']}",   f"{stats_b['bad_count']}"),
        ('差评率',           f"{stats_a['bad_rate']:.0f}%", f"{stats_b['bad_rate']:.0f}%"),
    ]
    for label, va, vb in rows:
        print(f"  {label:<20} {va:>20} {vb:>20}")


# ---------- ① 全量买家对比 ----------
all_stats = {}
for sid_short in TARGET_SELLERS:
    sub = items[items['seller_short'] == sid_short]
    all_stats[sid_short] = seller_stats(sub, sid_short)

print_comparison(all_stats[TARGET_SELLERS[0]], all_stats[TARGET_SELLERS[1]],
                 "① 全量买家对比（relogios_presentes）")

# ---------- ② 买家=SP 对比 ----------
sp = items[items['customer_state'] == 'SP'].copy()

sp_stats = {}
for sid_short in TARGET_SELLERS:
    sub = sp[sp['seller_short'] == sid_short]
    sp_stats[sid_short] = seller_stats(sub, sid_short)

print_comparison(sp_stats[TARGET_SELLERS[0]], sp_stats[TARGET_SELLERS[1]],
                 "② 买家=SP 对比（relogios_presentes）")

# ---------- ③ 结论 ----------
sa = sp_stats[TARGET_SELLERS[0]]
sb = sp_stats[TARGET_SELLERS[1]]
price_diff = abs(sa['avg_total'] - sb['avg_total'])
delivery_diff = abs(sa['median_delivery'] - sb['median_delivery'])
bad_diff = abs(sa['bad_rate'] - sb['bad_rate'])
cheaper = TARGET_SELLERS[0] if sa['avg_total'] < sb['avg_total'] else TARGET_SELLERS[1]
faster = TARGET_SELLERS[0] if sa['median_delivery'] < sb['median_delivery'] else TARGET_SELLERS[1]
better = TARGET_SELLERS[0] if sa['bad_rate'] < sb['bad_rate'] else TARGET_SELLERS[1]

print(f"\n{'=' * 70}")
print(f"  ③ 结论（买家=SP）")
print(f"{'=' * 70}")
print(f"  {TARGET_SELLERS[0]} vs {TARGET_SELLERS[1]}：")
print(f"    到手价差 R${price_diff:.0f}（{cheaper} 更便宜）")
print(f"    配送差 {delivery_diff:.1f} 天（{faster} 更快）")
print(f"    差评率差 {bad_diff:.0f} 个百分点（{better} 更低）")
print()
