# Olist 数据集 · 数据字典

> 数据来源：Kaggle Brazilian E-Commerce Public Dataset by Olist（2016-2018）
> 用途：面试前速查每张表的字段含义、表间关系

---

## 1. 客户表（olist_customers_dataset.csv）— 99,441 行

| 中文 | 英文字段 | 含义 |
|---|---|---|
| 客户ID | customer_id | 每个订单一个ID（一人可多个） |
| 客户唯一ID | customer_unique_id | 去重后的"真人"ID |
| 邮编前缀 | customer_zip_code_prefix | 收货邮编前5位 |
| 城市 | customer_city | 葡萄牙语城市名 |
| 州 | customer_state | 巴西州缩写（SP/RJ/RN…） |

---

## 2. 卖家表（olist_sellers_dataset.csv）— 3,095 行

| 中文 | 英文字段 | 含义 |
|---|---|---|
| 卖家ID | seller_id | 卖家唯一ID |
| 邮编前缀 | seller_zip_code_prefix | 发货地邮编 |
| 城市 | seller_city | 葡萄牙语 |
| 州 | seller_state | 州缩写 |

---

## 3. 商品表（olist_products_dataset.csv）— 32,951 行

| 中文 | 英文字段 | 含义 |
|---|---|---|
| 商品ID | product_id | 商品唯一ID |
| 品类名 | product_category_name | **葡萄牙语**品类 |
| 商品名长度 | product_name_lenght | ⚠️ 原表拼写错误，应为 length |
| 描述长度 | product_description_lenght | ⚠️ 同样拼写错误 |
| 图片数 | product_photos_qty | 商品图片数量 |
| 重量 | product_weight_g | 克 |
| 长/高/宽 | product_length_cm / _height_cm / _width_cm | 厘米 |

---

## 4. 订单表（olist_orders_dataset.csv）— 99,441 行（**最核心**）

| 中文 | 英文字段 | 含义 |
|---|---|---|
| 订单ID | order_id | 订单唯一ID |
| 客户ID | customer_id | 关联客户表 |
| 订单状态 | order_status | delivered=已送达 / canceled=取消 / unavailable=不可用 等 |
| 下单时间 | order_purchase_timestamp | 用户下单 |
| 审核时间 | order_approved_at | 平台/商家审核 |
| 交承运商时间 | order_delivered_carrier_date | 交给物流 |
| 送达时间 | order_delivered_customer_date | 客户签收 |
| 预计送达 | order_estimated_delivery_date | 平台承诺日期 |

**时效分析的来源**：5 个时间戳构成完整履约时间线。

---

## 5. 订单商品表（olist_order_items_dataset.csv）— 112,650 行

| 中文 | 英文字段 | 含义 |
|---|---|---|
| 订单ID | order_id | 关联订单表 |
| 商品序号 | order_item_id | 一个订单里第几个商品（1开始） |
| 商品ID | product_id | 关联商品表 |
| 卖家ID | seller_id | 关联卖家表 |
| 发货截止 | shipping_limit_date | 最晚发货时间 |
| 价格 | price | 商品单价（雷亚尔） |
| 运费 | freight_value | 运费（雷亚尔） |

**"多对多"坑的来源**：一个 order_id 可有多条记录、多个卖家。

---

## 6. 支付表（olist_order_payments_dataset.csv）— 103,886 行

| 中文 | 英文字段 | 含义 |
|---|---|---|
| 订单ID | order_id | 关联订单 |
| 支付序号 | payment_sequential | 一个订单第几笔支付 |
| 支付方式 | payment_type | credit_card=信用卡 / boleto=票据 / voucher=券 / debit_card=借记卡 |
| 分期数 | payment_installments | 分期期数 |
| 支付金额 | payment_value | 金额（雷亚尔） |

---

## 7. 评价表（olist_order_reviews_dataset.csv）— 99,224 行

| 中文 | 英文字段 | 含义 |
|---|---|---|
| 评价ID | review_id | 评价唯一ID |
| 订单ID | order_id | 关联订单 |
| 评分 | review_score | 1-5 分（1-2 分算差评） |
| 评价标题 | review_comment_title | 可能为空 |
| 评价内容 | review_comment_message | **退货词分析来源** |
| 评价时间 | review_creation_date | 创建时间 |
| 回复时间 | review_answer_timestamp | 商家回复时间 |

**注意**：`review_comment_message` 大量为 NaN（空）——"没评论=未知"的数据证据。

---

## 8. 地理位置表（olist_geolocation_dataset.csv）— 1,000,163 行

| 中文 | 英文字段 | 含义 |
|---|---|---|
| 邮编前缀 | geolocation_zip_code_prefix | 邮编 |
| 纬度 | geolocation_lat | 纬度 |
| 经度 | geolocation_lng | 经度 |
| 城市 | geolocation_city | 城市 |
| 州 | geolocation_state | 州 |

**"算真实距离"的原料**：有经纬度，可算卖家→买家直线距离。

---

## 9. 品类翻译表（product_category_name_translation.csv）— 71 行

| 中文 | 英文字段 | 含义 |
|---|---|---|
| 品类名（葡语） | product_category_name | 原始品类 |
| 品类名（英语） | product_category_name_english | 翻译后 |

**"类目命名坑"的根源**：葡语和英语两套名字。

---

## 表间关系

```
订单表(order_id) ─┬─→ 订单商品表(order_id, product_id, seller_id)
                  ├─→ 支付表(order_id)
                  └─→ 评价表(order_id)
                        │
              ┌─────────┴─────────┐
        商品表(product_id)    卖家表(seller_id)
```

- **订单是核心**，商品/支付/评价都挂在订单上；
- **商品和卖家**通过"订单商品表"关联（所以有"一单多卖家"的坑）；
- **时效**来自订单表 5 个时间戳；
- **退货风险**来自评价表 `review_comment_message`。
