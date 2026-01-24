这是一个非常专业且有趣的任务。基于您提供的 `df_matches`（赔率与概率数据）和 `df_leagues`（联赛宏观统计数据），我们可以构建不同维度的策略。

现有的策略01（保守/正路）和策略02（博冷/高赔）已经覆盖了基础两端。接下来的10个策略将尝试利用**数据挖掘**、**联赛特性**、**庄家心理**和**数学特征**来填充中间地带，旨在寻找不同的风险收益比（Sharpe Ratio）。

以下是为您设计的10个新策略逻辑：

---

### 第一类：统计与联赛特性驱动 (利用 `df_leagues`)

这两个策略的核心在于：**不只看当场比赛，更看该联赛的历史“性格”。**

#### Strategy 03: 联赛红利策略 (The League Reliability Strategy)

利用 `df_leagues` 中的 `P值` 和 `卡方统计量`。P值越低，说明该联赛越“正路”（符合理论概率）；P值越高，说明该联赛越“妖”（冷门多）。

```python
def _strategy_03(self, df: pd.DataFrame, df_leagues: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_03 核心逻辑：【基于联赛稳定性的降维打击】
    1. 关联数据：将 df_matches 与 df_leagues 按 '赛事' 合并。
    2. 单选稳胆 (i 场)：在 P值 < 0.05 (显著稳定) 的联赛中，选取主胜概率最高的场次，投注 3。
    3. 全选防冷 (l 场)：在 P值 > 0.5 (高波动) 的联赛中，选取赔率方差最小（最难判断）的场次，投注 310。
    4. 双选平 (j 场)：在“真实频率_平” > “理论概率_平”的联赛中（如法乙、意乙），选取平赔最低的场次，投注 31/10。
    5. 双选分胜负 (k 场)：剩余名额，选主胜/主负概率差值大的，投注 30。
    """
    # 代码逻辑实现区...
    return selection

```

#### Strategy 04: 偏差修正策略 (The Statistical Correction)

利用 `df_leagues` 中 `真实频率` 与 `理论概率` 的偏差。如果某联赛的主胜真实发生率远低于赔率暗示的理论概率，说明庄家在该联赛长期高估主队，我们反向操作。

```python
def _strategy_04(self, df: pd.DataFrame, df_leagues: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_04 核心逻辑：【理论与现实的偏差套利】
    1. 计算偏差：Delta = 真实频率_胜 - 理论概率_胜。
    2. 单选博冷 (i 场)：选取 Delta 最负（主胜被严重高估）且主胜SP < 1.5 的场次，反向投注 1 或 0（根据平负概率大小）。
    3. 双选补漏 (j+k 场)：选取 Delta 正值最大（主胜被低估）的联赛场次，优先保 3。
    4. 全选 (l 场)：针对卡方统计量异常高的联赛，全包。
    """
    # 代码逻辑实现区...
    return selection

```

---

### 第二类：赔率结构与庄家意图 (利用 `df_matches`)

这两个策略关注微观赔率结构，特别是庄家抽水（Vig）和赔率离散度。

#### Strategy 05: 抽水异常策略 (The Vig Alert Strategy)

`庄家抽水` 是庄家的利润保护。如果某场比赛抽水极低，说明庄家对结果把握大，敢于降低利润空间吸筹；如果抽水极高，说明庄家在通过高水位混乱视听。

```python
def _strategy_05(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_05 核心逻辑：【跟随庄家风控级别】
    1. 单选 (i 场)：选取“庄家抽水”最低（<5%）且赔率方差大的场次。庄家敢低水说明结果在射程内，选概率最大项。
    2. 全选 (l 场)：选取“庄家抽水”最高（>11%）的场次。高水意味着高风险和不确定性，直接 310。
    3. 双选 (j+k 场)：中间水位的比赛，按常规概率优势选 31 或 30。
    """
    return selection

```

#### Strategy 06: 赔率价值策略 (The EV Hunter)

不看胜率，看**期望值（EV）**。如果 （实际上因为抽水很少见，这里指相对值），说明该选项具有高赔付比。

```python
def _strategy_06(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_06 核心逻辑：【寻找被市场遗忘的高赔】
    1. 计算 EV：对每个选项计算 Value = SP值 * 对应概率。
    2. 单选 (i 场)：选取非热门项（概率<40%）但 EV 值全场最高的选项（通常是强队爆冷的平局或弱队胜）。
    3. 双选 (j/k 场)：选取两个 EV 值相加最高的组合。
    4. 目的：此策略命中率低，但一旦中奖，奖金极高（火锅奖绝缘体）。
    """
    return selection

```

---

### 第三类：对冲与形态策略 (特定数学逻辑)

#### Strategy 07: 熵增避险策略 (The Low Entropy Sniper)

与 Strategy 01 类似但逻辑相反。01 是高熵全包，07 是**只玩低熵**。任选9场的优势是可以“放弃”5场比赛。我们只选最容易看懂的。

```python
def _strategy_07(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_07 核心逻辑：【只打顺风球】
    1. 排序：计算每场比赛的信息熵，按熵值从小到大排序（最清晰到最混乱）。
    2. 弃赛：直接放弃熵值最高的5场比赛（不选）。
    3. 单选 (i 场)：在留下的9场中，熵值最低的前 i 场直接单选概率最大项。
    4. 双选 (j+k 场)：剩下的场次双选防守。
    5. 特点：极度追求命中率，容易中火锅奖，适合作为复式单的基础。
    """
    return selection

```

#### Strategy 08: 30两头堵策略 (The Decisive Battle)

专门针对“不爱平局”的联赛（如德甲、荷甲）或特定赔率组合。

```python
def _strategy_08(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_08 核心逻辑：【暴力分胜负】
    1. 筛选：选取“主平概率” < 25% 的比赛。
    2. 双选 (k 场)：直接双选 30，完全放弃平局。
    3. 单选 (i 场)：若主平概率 < 20% 且一方胜率 > 50%，直接单选 3 或 0。
    4. 适用场景：针对球风开放的联赛，或者保级生死战（平局对双方无意义）。
    """
    return selection

```

#### Strategy 09: 主场堡垒策略 (The Home Fortress)

利用足彩中的“主场优势”偏因。在势均力敌的比赛中，无脑偏向主队。

```python
def _strategy_09(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_09 核心逻辑：【主场加权法】
    1. 定义：在赔率接近（如主胜SP 2.3 vs 客胜SP 2.6）时，强制判定主队不败。
    2. 单选 (i 场)：主胜概率 > 45%，直接单选 3。
    3. 双选 (j 场)：主胜概率在 30%-45% 之间，选 31。
    4. 弃赛：只有当客胜概率 > 60% 时，才考虑放弃该场或全包。
    """
    return selection

```

---

### 第四类：极值与对冲 (Risk Hedging)

#### Strategy 10: 凯利准则变体 (Kelly Criterion Variant)

利用凯利公式思想寻找庄家赔付风险最高的地方。

```python
def _strategy_10(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_10 核心逻辑：【凯利值找冷】
    1. 假设：如果某项结果的赔率显著高于其应当具备的概率（即凯利指数 > 1.05），可能是诱盘，也可能是庄家在阻盘。
    2. 策略：找出凯利指数异常高的选项（通常是冷门）。
    3. 投注：将这些高风险项放入全包 (l) 或双选 (j) 范围，而不是单选。
    4. 稳胆：只在凯利指数平稳（0.90-0.95）的选项中选稳胆。
    """
    return selection

```

#### Strategy 11: 赔率断层策略 (The Odds Gap)

观察 SP 值之间的断层。

```python
def _strategy_11(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_11 核心逻辑：【赔率梯队】
    1. 1.30以下区间：设为稳胆 (i)。
    2. 1.30-1.70区间：设为防平双选 (31)。
    3. 1.70-2.30区间：死亡区间，设为全包 (l) 或 30 (k)。
    4. 2.30以上区间：放弃入选9场名单，或者如果必须选，则全包。
    """
    return selection

```

#### Strategy 12: 动态回撤策略 (Dynamic Drawdown)

这是最复杂的策略，试图模拟人类高手的“缩单”过程。

```python
def _strategy_12(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
    """
    策略 strategy_12 核心逻辑：【最优解逼近】
    1. 初始：先按 Strategy_01 选出 12 场候选。
    2. 缩减：计算这 12 场中，每场去除一个选项后的“安全系数损失”。
    3. 剔除：剔除损失最小的3场比赛，保留 9 场。
    4. 优化：对剩下的 9 场，如果总资金允许，将最不确定的单选升级为双选。
    """
    return selection

```

---

### 总结与建议

| 策略风格 | 推荐策略ID | 适用场景 | 预期ROI | 预期命中率 |
| --- | --- | --- | --- | --- |
| **稳健收米型** | 01, 07, 09 | 强弱分明的比赛周，防火锅 | 低 | 高 |
| **博取大奖型** | 02, 06, 10 | 豪门状态不佳，冷门频出周 | 极高 | 低 |
| **数据极客型** | 03, 04, 12 | 赛季中后期，联赛数据样本充足时 | 中高 | 中 |
| **特殊形态型** | 05, 08 | 德甲/荷甲专场，或赔率异常周 | 波动大 | 波动大 |