#!/usr/bin/env python3
"""
review_reason.csv 生成脚本
对差评文本（score≤2 且有文本）做葡语关键词规则分类，
按 category_en × seller_id × reason 聚合计数。

输出：knowledge_base/review_reason.csv
字段：category_en, seller_id, reason, n
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "olist"
OUT = Path(__file__).resolve().parent.parent / "knowledge_base"

# ── 差评原因关键词规则（葡语，优先级从上到下，首次匹配即停）──
REASON_RULES = [
    ("物流慢", re.compile(r"atraso|demora|atrasou|demorou|prazo|chegou tarde|demorado", re.I)),
    ("货不对板", re.compile(r"diferente|errado|gato por lebre|nao e o|não é o|outra cor|recebi outro", re.I)),
    ("质量差", re.compile(r"quebr|defeito|ruim|qualidade|veio quebrado|estragad|pessimo|péssimo", re.I)),
    ("包装差", re.compile(r"embalag|amassad|veio aberto|danificad", re.I)),
    ("尺寸不符", re.compile(r"tamanho|menor|maior|apert|grande demais|pequeno", re.I)),
    ("客服差", re.compile(r"atendimento|suporte|resposta|contato", re.I)),
    ("退款/纠纷", re.compile(r"devolu|reembolso|estorno|reclama", re.I)),
    ("与描述不符", re.compile(r"descricao|descrição|propaganda|anuncio|anúncio|engan", re.I)),
]


def classify_reason(text):
    """对一条差评文本做原因分类，返回原因标签。"""
    if not isinstance(text, str) or not text.strip():
        return "其他"
    for reason, pattern in REASON_RULES:
        if pattern.search(text):
            return reason
    return "其他"


def main():
    print("加载 reviews + items + products + translations ...")
    reviews = pd.read_csv(BASE / "olist_order_reviews_dataset.csv")
    items = pd.read_csv(BASE / "olist_order_items_dataset.csv")
    products = pd.read_csv(BASE / "olist_products_dataset.csv")
    trans = pd.read_csv(BASE / "product_category_name_translation.csv")

    # ── 1. 筛选差评且有文本 ──
    bad = reviews[
        (reviews["review_score"] <= 2)
        & reviews["review_comment_message"].notna()
    ][["order_id", "review_comment_message"]].copy()

    print(f"差评有文本: {len(bad)} 条")

    # ── 2. 每订单取 order_item_id 最小的那条（与 build_kb 一致）──
    first_item = (
        items.sort_values("order_item_id")
        .groupby("order_id")
        .first()
        .reset_index()[["order_id", "seller_id", "product_id"]]
    )

    # ── 3. 关联品类 ──
    products_en = products.merge(
        trans, on="product_category_name", how="left"
    )[["product_id", "product_category_name_english"]].rename(
        columns={"product_category_name_english": "category_en"}
    )

    df = bad.merge(first_item, on="order_id", how="inner")
    df = df.merge(products_en, on="product_id", how="left")
    df = df.dropna(subset=["category_en", "seller_id"])

    print(f"关联后有效行: {len(df)}")

    # ── 4. 分类差评原因 ──
    df["reason"] = df["review_comment_message"].apply(classify_reason)

    # 原因分布
    reason_dist = df["reason"].value_counts()
    print(f"\n差评原因分布:")
    for reason, cnt in reason_dist.items():
        print(f"  {reason}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # ── 5. 按 category_en × seller_id × reason 聚合 ──
    result = (
        df.groupby(["category_en", "seller_id", "reason"])
        .agg(n=("order_id", "count"))
        .reset_index()
    )

    out_path = OUT / "review_reason.csv"
    result.to_csv(out_path, index=False)
    print(f"\n输出: {out_path}")
    print(f"行数: {len(result)}")

    # ── 验证 ──
    # 按品类统计
    cat_counts = result.groupby("category_en")["n"].sum().sort_values(ascending=False)
    print(f"\n品类数: {len(cat_counts)}")
    print(f"Top 5 品类:")
    for cat, total in cat_counts.head(5).items():
        print(f"  {cat}: {total} 条")

    # 按卖家统计
    seller_counts = result.groupby("seller_id")["n"].sum().sort_values(ascending=False)
    print(f"\n卖家数: {len(seller_counts)}")
    print(f"Top 3 卖家:")
    for sid, total in seller_counts.head(3).items():
        print(f"  {sid[:12]}..: {total} 条")

    print("\nreview_reason.csv 生成完毕。")


if __name__ == "__main__":
    main()
