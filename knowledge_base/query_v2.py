"""
V2 聚合查询层 —— 只读 CSV，返回排序列表。
不改动 query.py（V1 数据层）。
"""

from pathlib import Path
from typing import Optional
import pandas as pd

_KB = Path(__file__).resolve().parent

# ── 懒加载 ──
_category_freight = None
_category_ship_time = None
_route_freight = None


def _load_category_freight():
    global _category_freight
    if _category_freight is None:
        _category_freight = pd.read_csv(_KB / "category_freight.csv")
    return _category_freight


def _load_category_ship_time():
    global _category_ship_time
    if _category_ship_time is None:
        _category_ship_time = pd.read_csv(_KB / "category_ship_time.csv")
    return _category_ship_time


def _load_route_freight():
    global _route_freight
    if _route_freight is None:
        _route_freight = pd.read_csv(_KB / "route_freight.csv")
    return _route_freight


def query_category_freight(top_n: int = 5, ascending: bool = False) -> list:
    """
    各品类运费排名。
    ascending=True  → 最便宜；ascending=False → 最贵（默认）
    返回: [{"category_en", "n", "avg_freight"}, ...]
    """
    df = _load_category_freight()
    df = df.sort_values("avg_freight", ascending=ascending).head(top_n)
    return df.to_dict("records")


def query_category_ship_time(top_n: int = 5, ascending: bool = True) -> list:
    """
    各品类发货时长排名。
    ascending=True  → 最快（默认）；ascending=False → 最慢
    返回: [{"category_en", "n", "median_days", "avg_days"}, ...]
    """
    df = _load_category_ship_time()
    df = df.sort_values("median_days", ascending=ascending).head(top_n)
    return df.to_dict("records")


def query_route_freight_single(seller_state: str, buyer_state: str) -> Optional[dict]:
    """
    查询单条路线运费（精确匹配 seller_state + customer_state）。
    返回: {"n", "avg_freight"} 或 None（查不到）。
    """
    df = _load_route_freight()
    mask = (df["seller_state"] == seller_state) & (df["customer_state"] == buyer_state)
    if mask.sum() == 0:
        return None
    row = df[mask].iloc[0]
    return {"n": int(row["n"]), "avg_freight": float(row["avg_freight"])}


def query_route_freight(top_n: int = 5, ascending: bool = False) -> list:
    """
    各路线运费排名。
    ascending=True  → 最便宜；ascending=False → 最贵（默认）
    返回: [{"seller_state", "customer_state", "n", "avg_freight"}, ...]
    """
    df = _load_route_freight()
    df = df.sort_values("avg_freight", ascending=ascending).head(top_n)
    return df.to_dict("records")


if __name__ == "__main__":
    print("=== 品类运费 Top 5 最贵 ===")
    for r in query_category_freight(5, ascending=False):
        print(f"  {r['category_en']}: {r['avg_freight']:.1f} (n={r['n']})")

    print("\n=== 品类发货 Top 5 最快 ===")
    for r in query_category_ship_time(5, ascending=True):
        print(f"  {r['category_en']}: {r['median_days']:.1f}天 (n={r['n']})")

    print("\n=== 路线运费 Top 5 最贵 ===")
    for r in query_route_freight(5, ascending=False):
        print(f"  {r['seller_state']}→{r['customer_state']}: {r['avg_freight']:.1f} (n={r['n']})")
