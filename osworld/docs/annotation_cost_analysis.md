# 视频标注开销分析

## 标注流程概述

每个视频任务的标注分为三个阶段：

1. **帧级别标注（Frame Annotation）**：逐帧对比相邻截图（两两配对），判断操作是否有意义并提取动作描述
2. **Planning 生成**：汇总所有有效帧标注，生成任务级别规划文本
3. **Grounding 生成**：汇总所有有效帧标注，生成带坐标的操作轨迹文本

---

## Token 计算依据

### 数据来源

基于 `videos/chrome/.../OmniParser_Pic/` 实际文件测量：

- 每视频总帧文件数：**30 帧**（png + parsed.txt 各 15 对）
- 实际 API 调用：帧两两配对 → **15 次调用/视频**
- 有效帧比例：约 **73%**（约 11 次有效标注，3–4 次 meaningless）

**OmniParser 解析文件（parsed.txt）实测大小及帧类型：**

| 帧 | 文件大小 | 类型 |
|----|---------|------|
| frame_000 | 0 chars | **非GUI帧**（开场动画/Logo） |
| frame_001 | 199 chars | **非GUI帧**（转场过渡） |
| frame_002 | 13,107 chars | GUI-无效（Meaningful: False） |
| frame_003 | 13,436 chars | GUI-无效（Meaningful: False） |
| frame_004 | 12,600 chars | GUI-有效 |
| frame_005 | 13,865 chars | GUI-有效 |
| frame_006 | 18,653 chars | GUI-有效 |
| frame_007 | 19,721 chars | GUI-有效 |
| frame_008 | 19,706 chars | GUI-有效 |
| frame_009 | 20,490 chars | GUI-有效 |
| frame_010 | 23,297 chars | GUI-有效 |
| frame_011 | 20,682 chars | GUI-有效 |
| frame_012 | 20,212 chars | GUI-有效 |
| frame_013 | 8,453 chars | GUI-有效 |
| frame_014 | 10,110 chars | GUI-有效 |

**关键观察**：视频关键帧中存在**非 GUI 帧**（开场动画、转场、加载画面等），此类帧的 OmniParser 无法提取到有效交互元素，输出几乎为空（0–200 chars）。这类帧往往集中在 meaningless 调用中。按类型分层的平均大小：

| 帧类型 | 数量 | 平均大小 |
|--------|------|---------|
| 非GUI帧（OmniParser≈空） | ~2–4 帧 | **~100 chars** |
| GUI-有效帧（实测，复杂浏览器设置页） | 11 帧 | **~17,072 chars** |
| GUI-有效帧（典型OSWorld单应用任务） | 11 帧 | **~10,000 chars**（估算） |

### 图像 Token 计算（OpenAI high-detail 模式）

图像分辨率：1920 × 1080

```
tiles_w = ceil(1920 / 512) = 4
tiles_h = ceil(1080 / 512) = 3
image_tokens = 85 + 4 × 3 × 170 = 2,125 tokens/张
```

每次帧标注传入 **2 张图像**：
```
image_tokens_per_call = 2 × 2,125 = 4,250 tokens
```

### 帧标注 Prompt 文本构成

`generate_vlm_action_prompt(json_file_1, json_file_2, task_description, thought)` 包含：

```
静态模板文本（指令+动作空间+输出格式）：~2,200 chars = ~550 tokens
OmniParser JSON 1（frame N）：        ~14,302 chars = ~3,576 tokens
OmniParser JSON 2（frame N+2）：       ~14,302 chars = ~3,576 tokens
----------------------------------------------
每次帧标注 Input 合计：
  4,250 (图像) + 7,151 (OmniParser×2) + 550 (文本) = 11,951 tokens
```

### Consolidated 标注文件（Planning/Grounding 输入）

实测（`_thou_and_action.txt` 汇总文件）：
```
Consolidated annotation: 10,914 chars = ~2,728 tokens
```

---

## 每视频 Token 消耗（实测）

### 阶段一：帧标注（15 次 API 调用）

**关键：有效帧与无效帧的输入 token 差异取决于 OmniParser 输出大小。** 非 GUI 的无效帧因 OmniParser 几乎为空，每次调用 input 显著减少；而 GUI-无效帧 OmniParser 与有效帧相近，差异仅在输出。

```
# 有效帧调用（11次）
valid_call_input = 2,125×2 (图像) + 2×17,072/4 (OmniParser) + 550 (prompt) = 13,336 tokens

# 非GUI无效帧调用（4次）——OmniParser几乎为空
invalid_call_input = 2,125×2 (图像) + 2×100/4 (OmniParser≈0) + 550 (prompt) = 4,850 tokens

frame_input = 11 × 13,336 + 4 × 4,850 = 146,696 + 19,400 = 166,096 tokens

# 输出（按有效/无效区分）
有效帧（11次）：thou_and_action ~896字符 + thought ~804字符 + actions ~100字符 = ~450 tokens/次
无效帧（4次）： 模型输出 Thought + "Meaningful: False" + 空 actions  ≈ 350 tokens/次
  （代码仅写入 "Meaningful: False" 到 labeled.txt，不生成 thou_and_action.txt）

frame_output = 11 × 450 + 4 × 350 = 4,950 + 1,400 = 6,350 tokens
```

### 阶段二：Planning 生成（1 次调用）

consolidated 文件只包含 **11 个有效帧**的 thou_and_action 内容（无效帧不写入）：

```
planning_input  = 2,728 (consolidated，11帧内容) + 450 (静态模板) = 3,178 tokens
planning_output = 546 tokens（实测：2,185 chars）
```

### 阶段三：Grounding 生成（1 次调用）

```
grounding_input  = 3,178 tokens（同 planning，输入同一 consolidated 文件）
grounding_output = 1,650 tokens（实测：6,598 chars）
```

### 汇总（含非GUI帧修正，复杂GUI场景）

| 阶段 | API 调用次数 | Input Tokens | Output Tokens | 小计 |
|------|------------|-------------|--------------|------|
| 帧标注—有效 (11次, 复杂GUI) | 11 | 146,696 | 4,950 | 151,646 |
| 帧标注—无效 (4次, 非GUI≈空) | 4 | 19,400 | 1,400 | 20,800 |
| Planning 生成 | 1 | 3,178 | 546 | 3,724 |
| Grounding 生成 | 1 | 3,178 | 1,650 | 4,828 |
| **合计** | **17** | **172,452** | **8,546** | **~181,000** |

**每视频约 ~181,000 tokens（~172K input + ~8.5K output）**

---

## 费用计算

### 模型定价

> doubao-seed-1.8 / qwen3vl-8b-instruct 官方定价为人民币，按 **1 USD = 7.2 CNY** 换算为美元。

| 模型 | Input ($/1M tokens) | Output ($/1M tokens) | Cached Input ($/1M tokens) |
|------|--------------------|--------------------|--------------------------|
| gpt-5.1-2025-11-13 | $1.2500 | $10.0000 | $0.1250 |
| gpt-4.1-mini | $0.2000 | $1.6000 | $0.0500 |
| doubao-seed-1.8 ¹ | $0.1111 | $1.1111 | — |
| qwen3vl-8b-instruct | $0.0694 | $0.2778 | — |

> ¹ doubao-seed-1.8 分档计价（原价 CNY）：所有标注调用均落在 `[0, 32K] 输入 + (0.2K, ∞) 输出` 档（¥0.80/M 输入，¥8.00/M 输出），其余档位（32K–256K 输入）不适用。

---

### 各场景 Token 总量

两种场景差异来源于有效帧 OmniParser 数据量（屏幕 UI 元素复杂程度）：

| 场景 | 总 Input | 总 Output | 说明 |
|------|---------|---------|------|
| A：复杂GUI | 172,452 | 8,546 | 浏览器设置页，OmniParser 实测均值 ~17K chars/帧 |
| B：典型任务 | 133,556 | 8,546 | 单应用专注操作，OmniParser 估算均值 ~10K chars/帧 |

---

### 单视频费用计算

```
# 场景A (172,452 input + 8,546 output)
gpt-5.1:        172,452/1M × $1.2500 + 8,546/1M × $10.0000 = $0.2156 + $0.0855 = $0.30
gpt-4.1-mini:   172,452/1M × $0.2000 + 8,546/1M ×  $1.6000 = $0.0345 + $0.0137 = $0.048
doubao-seed-1.8:172,452/1M × $0.1111 + 8,546/1M ×  $1.1111 = $0.0192 + $0.0095 = $0.029
qwen3vl-8b:     172,452/1M × $0.0694 + 8,546/1M ×  $0.2778 = $0.0120 + $0.0024 = $0.014

# 场景B (133,556 input + 8,546 output)
gpt-5.1:        133,556/1M × $1.2500 + 8,546/1M × $10.0000 = $0.1669 + $0.0855 = $0.25
gpt-4.1-mini:   133,556/1M × $0.2000 + 8,546/1M ×  $1.6000 = $0.0267 + $0.0137 = $0.040
doubao-seed-1.8:133,556/1M × $0.1111 + 8,546/1M ×  $1.1111 = $0.0148 + $0.0095 = $0.024
qwen3vl-8b:     133,556/1M × $0.0694 + 8,546/1M ×  $0.2778 = $0.0093 + $0.0024 = $0.012
```

### 四模型综合对比

| 模型 | 场景A (复杂GUI) | 场景B (典型任务) | 相对 gpt-5.1 (场景B) |
|------|--------------|----------------|-------------------|
| gpt-5.1 | **~$0.30** | **~$0.25** | 1× |
| gpt-4.1-mini | **~$0.048** | **~$0.040** | ~6× cheaper |
| doubao-seed-1.8 | **~$0.029** | **~$0.024** | ~10× cheaper |
| qwen3vl-8b-instruct | **~$0.014** | **~$0.012** | ~21× cheaper |

### 批量费用估算（场景B 典型任务）

| 规模 | gpt-5.1 | gpt-4.1-mini | doubao-seed-1.8 | qwen3vl-8b-instruct |
|------|---------|-------------|----------------|---------------------|
| 1 视频 | **~$0.25** | **~$0.040** | **~$0.024** | **~$0.012** |
| 50 视频 | **~$13** | **~$2.0** | **~$1.2** | **~$0.6** |
| 100 视频 | **~$25** | **~$4.0** | **~$2.4** | **~$1.2** |
| 1,000 视频 | **~$252** | **~$40** | **~$24** | **~$12** |

---

## 简化估算公式

```
N   = 视频帧对数（典型值 15）
V   = 有效帧数（典型值 11，即 73% × 15）
J_v = 有效帧 OmniParser 字符数/帧（复杂GUI ~17,000 | 典型任务 ~10,000）
J_i = 无效帧 OmniParser 字符数/帧（非GUI帧 ~100，几乎为空）

# 帧标注：有效/无效调用分别计算
valid_call_input   = 2,125×2 + 2×J_v/4 + 550
invalid_call_input = 2,125×2 + 2×J_i/4 + 550 ≈ 4,850（非GUI时）

frame_input  = V × valid_call_input + (N-V) × invalid_call_input
frame_output = V × 450 + (N-V) × 350

total_input  = frame_input + 2 × 3,178
total_output = frame_output + 546 + 1,650

cost ($) = total_input / 1e6 × price_in + total_output / 1e6 × price_out
```

代入 N=15，V=11，J_i=100，total_output=8,546：

| 场景 | J_v | total_input | gpt-5.1 | gpt-4.1-mini | doubao-seed-1.8 | qwen3vl-8b |
|------|-----|------------|---------|-------------|----------------|-----------|
| A 复杂GUI | 17,000 | 172,452 | $0.30 | $0.048 | $0.029 | $0.014 |
| B 典型任务 | 10,000 | 133,556 | $0.25 | $0.040 | $0.024 | $0.012 |

---

## 附：各来源 Token 占比

| 来源 | Input Tokens | 占比 |
|------|-------------|------|
| OmniParser JSON（30文件） | 107,265 | 58% |
| 图像（30张 × 2,125） | 63,750 | 34% |
| 静态Prompt文本（15次 × 550） | 8,250 | 4% |
| Planning/Grounding文本 | 6,356 | 3% |
| **合计** | **185,621** | **100%** |

> 实测关键数据：OmniParser parsed.txt 平均 **14,302 chars/帧**，是 input token 的最大来源，占 58%。

---

## 附：实测文件尺寸汇总

| 文件类型 | 平均大小 | 对应 Tokens |
|---------|---------|------------|
| OmniParser parsed.txt（每帧） | 14,302 chars | 3,576 tokens |
| 帧标注输出 thou_and_action.txt | 896 chars | 224 tokens |
| 帧标注输出 thought.txt | 804 chars | 201 tokens |
| Consolidated 汇总文件 | 10,914 chars | 2,728 tokens |
| Planning 输出 | 2,185 chars | 546 tokens |
| Grounding 输出 | 6,598 chars | 1,649 tokens |

---

*生成时间：2026-03-06，基于 `videos/chrome/How to browse Civil Division f/` 实际文件测量*
