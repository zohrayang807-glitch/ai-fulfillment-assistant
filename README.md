# AI 购物助手作品集

用 Olist 巴西电商公开数据验证"懂履约的购物助手"分析框架 — 通过历史配送数据，给出比平台承诺更真实的时效预期。

## 背景

目标岗位：抖音电商 AI 购物助手方向产品经理。

平台承诺时效普遍保守失真，用户无法据此安排收货。AI 购物助手的核心价值是：基于历史履约数据，为用户提供更准确的到货预期，而非复述平台的保守承诺。

本项目用 Olist 数据集完成分析框架验证，产品落地场景为抖音电商。

## 数据来源

[Kaggle: Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

- 时间区间：2016 年 9 月 — 2018 年 10 月
- 订单量：约 9.6 万笔已完成配送订单
- 覆盖 22 个卖家州、50+ 商品类目

## 复现步骤

```bash
# 1. 下载数据集，解压到 olist/ 目录（9 个 CSV 文件）
# 2. 安装依赖
pip install pandas

# 3. 运行验证脚本
python analysis/story1_provenance.py
```

## 目录结构

```
ai-portfolio/
├── REPORT.md                              # 完整分析报告（Markdown 格式）
├── README.md                              # 本文件
├── analysis/
│   ├── analyze_delivery.py                # 配送时长分析（按州汇总、时长分布）
│   ├── analyze_promise_accuracy.py        # 承诺准确度分析（提前/准时/延迟、类目延迟率）
│   ├── analyze_percentiles.py             # 分位数分析（P50/P75/P90、中位数vs均值偏斜检测）
│   ├── generate_report.py                 # 汇总生成 REPORT.md
│   └── story1_provenance.py               # 四项关键结论的独立数据验证
├── notes/
│   ├── opportunity-statement.md           # 机会陈述
│   └── story-1.md                         # 故事线笔记
└── olist/                                 # 原始数据（9 个 CSV，不入库）
```

## 关键数据发现

| 指标 | 数值 | 说明 |
|---|---|---|
| 配送超 7 天占比 | **73.0%** | 巴西物流整体偏慢 |
| 承诺提前送达率 | **90.4%** | 平均比承诺早到 11.2 天，承诺严重保守 |
| 实际配送 P50 | **10.2 天** | 一半订单 10 天内送达 |
| 实际配送 P90 | **23.1 天** | 尾部体验显著恶化 |
| SP→RN 10 天内到货率 | **11%** | 332 笔中仅 35 笔，跨区配送时效不可控 |
| 承诺模型驱动因素 | **卖家-买家距离** | 同州 17.6 天 vs 跨州 27.2 天，类目几乎无影响 |
