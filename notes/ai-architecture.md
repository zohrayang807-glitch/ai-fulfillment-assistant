# AI 架构设计 — 懂履约的 AI 购物助手 V2.0

> 版本：v2.0（已上线）｜反映当前实际架构

---

## 一、整体架构（四层）

```
用户提问
  ↓
【意图识别层】三元组解析（操作 × 维度 × 指标）
  ↓
【参数校验层】必填参数校验 + 缺参兜底
  ↓
【数据查询层】按映射表调 Supabase 数据
  ↓
【回答生成层】LLM 把数据组织成人话 + 双层评审
```

**核心原则：数字只由数据层计算，LLM 只负责"听懂 + 说人话"，不参与任何计算。**

---

## 二、意图识别：三元组模型（V2 核心升级）

从"单标签意图"升级为**三元组**（operation × dimension × metric）：

| 轴 | 取值 |
|---|---|
| **操作** operation | query 查询 / compare 对比 / aggregate 聚合 / recommend 推荐 |
| **维度** dimension | seller 商家 / category 品类 / route 路线 |
| **指标** metric | ship_time 发货 / transit_time 运输 / total_time 总时长 / freight 运费 / price 价格 / neg_rate 差评 / ontime_rate 准时 / promise_gap 承诺偏差 |

**输出结构**（LLM 报意图，代码查映射表路由）：

```json
{"intents": [{"operation": "query", "dimension": "seller", "metric": "ship_time"}],
 "entities": {"seller_ids": ["b1a812"], "category": null, "buyer_state": null}}
```

**为什么不用原生 function calling**：LLM 只报意图（理解），代码查 `QUERY_DISPATCH` 映射表决定调哪个函数（精确路由）——函数路由由代码决定、可审计，避免 LLM 调错工具。

---

## 三、模型选型（实测确认）

| 用途 | 模型 | 行为 | 理由 |
|---|---|---|---|
| **主模型**（回答用户） | `deepseek-chat` | 对话模式，不显式思考 | 快、省、稳，有推理和上下文能力 |
| **裁判模型**（评审） | `deepseek-v4-pro` | 推理模型，先思考再答 | 评审需要深思，更强 |

**关键坑（实测验证）**：
- `deepseek-chat` 不是壳/别名，是真实"对话模式"入口（reasoning_tokens=None）
- `deepseek-v4-flash`/`pro` 是推理模型，思考占 token，**max_tokens 必须给足（≥2000）**，否则思考吃光、输出为空
- 配置在 `config/model.yaml`，后台可切换

---

## 四、数据层：Supabase（部署后实际）

从"本地 JSONL 文件"迁移到 **Supabase 云端数据库**（多人共享、数据持久）：

| 表 | 用途 |
|---|---|
| conversations | 对话日志 |
| evaluations | 评审记录（框架 + 裁判） |
| cases | Eval 用例 |
| bug_feedback | BUG 反馈单 |
| prompt_versions | 提示词版本 |

**db.py 数据访问层**：所有读写封装，Supabase 失败时 fallback 到本地 JSONL。

**知识库 CSV**：route_timing / seller_risk / seller_cost / category_baseline / delivery_vs_promise / route_freight 等，预计算查询表。

---

## 五、回答生成与双层评审

### 防幻觉铁律（贯穿）

1. 数字由数据层查询，LLM 不计算
2. 缺失维度禁编造（没查价格不说"综合推荐"）
3. 只答用户问的指标，不脑补没问的维度
4. 数据模糊化（约/大概/九成），禁 n=/P50/P90

### 双层评审

- **框架评审**（代码规则）：数据一致性、拆段、防编造
- **模型评审**（裁判 v4-pro）：准确性/完整性/语气/防幻觉打分
- **评审开关**：默认关闭对话评审，Eval 时才评审（省 token）

---

## 六、坏例闭环与 Eval

```
标记坏例（填原因）
  ↓
normalize_case 规范化（字段统一）
  ↓
写入 Supabase cases
  ↓
Eval 并发跑（读 Supabase）→ 32 条 20 秒
  ↓
通过率写入 evaluations → 报表可视化
```

**Eval 并发**：ThreadPoolExecutor(6)，32 条 5.4× 提速。

---

## 七、错误处理与降级

| 场景 | 兜底 |
|---|---|
| Supabase 写入失败 | fallback 本地 JSONL + 打印错误 |
| LLM 返回空/非 JSON | 重试一次 + 返回安全默认（不崩） |
| Eval 用例格式异常 | normalize_case 统一字段 |
| 缺参数 | 全缺合并反问 / 部分缺单独提示 |

---

## 八、可观测性

- **指标看板**：对话数、模型调用、token、成本、Eval 通过率、坏例数
- **Token 明细**：每次调用（模型、token、成本）
- **评审记录**：每条对话的框架 + 裁判评分
- **历史记录**：Eval 通过率趋势

---

## 九、技术选型与理由

| 问题 | 选择 | 为什么 |
|---|---|---|
| 数据是精确数值 | **结构化查询**，不用 RAG | RAG 检索文本有数字漂移，精确数值必须确定性查询 |
| 意图→工具调用 | **prompt + 代码路由**，不用原生 FC | 函数由代码决定、可审计 |
| 能力组合 | **三元组正交架构** | 加指标 = 三操作自动点亮 |
| 多人共享 | **Supabase** | 数据持久、共享访问 |
| 部署 | **Streamlit Cloud** | 免费、自动拉 GitHub 代码 |
