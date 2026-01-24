这是一个基于你提供的**任选9场（Ren Xuan 9）**规则、数据以及概率定义的量化套利系统架构设计。

该系统被命名为 **"RX9-Alpha"（任九阿尔法）**。其核心理念不再是单纯的预测胜负，而是寻找**赔率与实际概率的偏差**，并结合**奖池周期**进行资金配置，以实现长期正收益。

---

# RX9-Alpha 量化套利系统架构

该系统由四个核心模块组成：**数据清洗与特征工程层**、**策略量化引擎**、**组合优化器**、以及**回测与参数寻优系统**。

---

### 一、 数据清洗与特征工程层 (Data ETL & Feature Engineering)

这一层负责将原始赔率和历史数据转化为机器可理解的指标。

#### 1. 基础概率标准化 (Based on User's Definition)

根据你提供的“最终概率”定义，系统首先处理当期14场比赛的赔率。

* **输入**：欧赔（胜/平/负）、亚盘水位、凯利指数。
* **处理**：
1. **去水（De-vig）**：计算 。
2. **归一化（Normalization）**：。
3. **输出**：得到每场比赛的 `主胜最终概率`、`平局最终概率`、`主负最终概率`。



#### 2. 联赛特征画像 (League Profiling)

利用 `df_final_prob` 数据，给每场比赛打上“联赛属性标签”。

* **波动率计算**：读取该联赛的 `Std_最终概率`。
* *逻辑*：如果某场比赛属于“英超”，其平局标准差仅为 0.0356，说明英超平局概率稳定；若属于“非洲杯”，客胜标准差高达 0.2506，说明非洲杯极易出现由于客队表现波动导致的冷门。


* **量化指标**：生成 `Volatility_Index` (波动指数)。波动指数越高，该场比赛越适合作为“博冷”对象，而非“稳胆”。

#### 3. 周期状态判定 (Cycle Detection)

利用 `df_bonus` 数据，判断当前处于什么周期。

* **规则算法**：读取最近 3 期奖金。
* `If (Last_3_Prizes < 1000)` -> 标记为 **"反弹窗口期"** (High Expected Value)。
* `If (Last_Prize > 50000)` -> 标记为 **"回落调整期"** (Low Expected Value)。


* **输出**：`Cycle_Factor` (周期系数，范围 0.8 - 1.5)，用于调整后续的投注资金量。

---

### 二、 策略量化引擎 (Strategy Engine)

这是系统的“大脑”，负责根据用户参数生成具体的选单策略。

#### 1. 比赛分级算法 (Match Grading)

系统将当期14场比赛进行评分排序。

* **稳胆得分 (Safety Score)**：
* *解释*：胜率越高且所属联赛越稳定的比赛，稳胆分越高。


* **博冷得分 (Value Score)**：
* *解释*：赔率回报高且所属联赛波动大的比赛，博冷分越高。



#### 2. 动态筛选逻辑 (Dynamic Selection)

根据用户设定的风格，从14场中筛选目标9场（或更多场次进行容错）。

| 风格参数 | 筛选逻辑 (Python 伪代码) | 适用场景 |
| --- | --- | --- |
| **保守型 (稳收益)** | `Select top 9 matches by Safety_Score` | 奖池平稳期，目标是中几百元回血。 |
| **激进型 (博大奖)** | `Select 5 High Safety + 4 High Value` | “火锅奖”连出3期后，利用非洲杯/法乙博冷。 |
| **均衡型 (智能)** | `Select 7 Safety + 2 Medium Value` | 常规操作。 |

---

### 三、 组合优化器 (Portfolio Optimizer)

这一层解决“怎么买最划算”的问题，运用你提到的**胆拖**和**矩阵过滤**技术。

#### 1. 智能胆拖生成 (Smart Dan-Tuo)

* **定胆 (Banker Selection)**：
* 选取 `Safety_Score` 最高的  场比赛作为“胆”。
* *系统约束*：若 ，系统发出警告，建议减少胆码数量。


* **补拖 (Tuo Selection)**：
* 针对“模糊场次”（如 `df_final_prob` 中平局概率高的法乙），自动生成双选（31或10）。



#### 2. 矩阵降本算法 ("Choose 5 Use 4" Implementation)

实现你提到的“选五用四”逻辑来降低成本。

* **输入**：5场不确定的比赛作为“预备胆”。
* **处理**：生成  组方案。
* **输出**：5张单子，只要这5场里错1场以内，就能保证有一张单子胆码全对。

#### 3. 资金分配 (Kelly Criterion Adaptation)

* **基准注额**：。
* 如果当前是“反弹窗口期” (`Cycle_Factor` = 1.5)，系统会自动建议将预算从 200元 提升至 300元。

---

### 四、 回测与参数寻优系统 (Backtesting & Optimization)

这是系统的验证环节，利用 `df_outcome_freq` 和 `df_outcome_prob` 进行历史模拟。

#### 1. 回测模拟器 (Simulator)

* **输入**：过去 100 期的赔率数据。
* **模拟过程**：
1. 系统根据设定的参数（如：只选五大联赛稳胆）生成虚拟投注单。
2. 对比实际赛果。
3. **计算收益**：如果中奖，根据当期 `df_bonus` 中的奖金计算回报；如果未中，计入成本损失。


* **关键处理**：在模拟“超级冷”期号（如25193期）时，利用 `df_outcome_prob` 数据修正命中率。例如，在超级冷周期，系统会强制降低强队胜出的权重。

#### 2. 性能指标 (KPIs)

* **ROI (投资回报率)**：总奖金 / 总投入。
* **Hit Rate (命中率)**：中奖期数 / 总期数。
* **Max Drawdown (最大回撤)**：连续未中奖造成的最大资金损耗。

#### 3. 参数自动调优 (Grid Search)

系统自动遍历参数组合，寻找最佳配置。例如：

* *参数A*：胆码数量 (2, 3, 4)
* *参数B*：双选场次数量 (3, 4, 5)
* *参数C*：止损线
* **结果**：系统可能会发现，“在‘比较冷’的周期（数据源于 `df_outcome_freq`），使用 **2胆7拖 + 全双选** 的策略 ROI 最高”。

---

### 五、 系统代码实现核心逻辑 (Python Demo)

这是一个简化的核心类设计，展示如何将上述逻辑代码化：

```python
import pandas as pd
import numpy as np
import itertools

class RX9_System:
    def __init__(self, risk_profile='balanced', budget=200):
        self.risk_profile = risk_profile # conservative, balanced, aggressive
        self.budget = budget
        self.league_stats = df_final_prob.set_index('赛事') # 使用你提供的数据
        
    def calculate_match_score(self, match_data):
        """
        计算每场比赛的稳胆分和博冷分
        match_data 包含: 联赛, 赔率, 让球等
        """
        league = match_data['league']
        # 获取该联赛的最终概率标准差
        try:
            std_win = self.league_stats.loc[league, 'Std_最终概率_胜']
            std_draw = self.league_stats.loc[league, 'Std_最终概率_平']
        except KeyError:
            # 默认值处理
            std_win = 0.15 
            std_draw = 0.03

        # 计算去水后的最终概率 (User defined logic)
        probs = self._normalize_probs(match_data['odds'])
        
        # 评分逻辑
        # 稳胆分：概率高 且 联赛波动小
        safety_score = probs['win'] * (1 - std_win)
        
        # 博冷分：概率适中 但 联赛波动极大 (容易爆冷)
        value_score = (1 / probs['win']) * std_win if probs['win'] > 0 else 0
        
        return safety_score, value_score, probs

    def optimize_ticket(self, matches_14):
        """
        根据策略生成最优的9场组合
        """
        scored_matches = []
        for m in matches_14:
            s_score, v_score, probs = self.calculate_match_score(m)
            m['safety'] = s_score
            m['value'] = v_score
            scored_matches.append(m)
            
        # 排序
        scored_matches.sort(key=lambda x: x['safety'], reverse=True)
        
        if self.risk_profile == 'conservative':
            # 保守策略：选前9个最稳的
            selection = scored_matches[:9]
            strategy_note = "选用了稳胆得分最高的9场，主要集中在五大联赛。"
            
        elif self.risk_profile == 'aggressive':
            # 激进策略：5个稳胆 + 4个高波动博冷
            bankers = scored_matches[:5]
            # 剩下的按 value_score 排序选4个
            remaining = sorted(scored_matches[5:], key=lambda x: x['value'], reverse=True)
            punters = remaining[:4]
            selection = bankers + punters
            strategy_note = "选用了5个稳胆，并在非洲杯/法乙等高波动联赛中配置了4个博冷选项。"
            
        return selection, strategy_note

    def _normalize_probs(self, odds):
        # 简单的去水归一化逻辑
        raw = [1/o for o in odds]
        total = sum(raw)
        return {'win': raw[0]/total, 'draw': raw[1]/total, 'loss': raw[2]/total}

# 示例运行逻辑
# system = RX9_System(risk_profile='aggressive')
# selection, note = system.optimize_ticket(current_week_matches)
# print(note)

```

### 六、 总结与下一步

这个架构完全基于你提供的“任选9”核心规则和“最终概率”数据体系。

1. **量化机会**：通过 `Safety Score` 和 `Value Score` 将感性的“稳”和“冷”变成了可计算的数字。
2. **风格调整**：通过切换 `risk_profile` 参数，系统会自动改变选比赛的权重（侧重 `Std` 还是 `Prob`）。
3. **回测优化**：利用 `df_bonus` 和 `df_outcome_freq` 可以在历史数据中验证，到底“连续3期火锅后加倍倍投”能不能跑赢大盘。

**您希望我下一步为您生成哪一部分的具体代码？**
A. **数据分析模块**：详细计算每个联赛的  (博冷价值)，找出最容易爆冷的联赛。
B. **回测模块**：模拟过去5期（25189-25193），看看如果用“激进策略”，能否抓到 25193 期的 7.9万 大奖。
C. **前端展示**：设计一个简单的界面，输入赔率，直接输出推荐的任九组合。