import pandas as pd
import numpy as np
from typing import List, Dict
from data import DataSource
from engine import ProbabilityEngine, MatchInfo

# ==========================================
# 3. 周期研判系统 (Cycle Detector)
# ==========================================

class CycleAnalyzer:
    """
    分析 df_bonus，判断当前处于什么周期
    """
    def __init__(self, data_source: DataSource):
        self.history = data_source.df_bonus
        
    def predict_current_state(self):
        """
        逻辑：
        1. 获取最近3期的奖金。
        2. 如果连续低奖金（火锅），预测下期为“反弹/比较冷”。
        3. 如果上期是超级大奖，预测下期回归“一般”。
        """
        last_3_prizes = self.history['一等奖'].tail(3).values
        
        # 规则1：火锅不过三 (用户提供的核心规律)
        if all(p < 1000 for p in last_3_prizes):
            return "比较冷", 1.5 # 状态，资金系数(加注)
        
        # 规则2：均值回归
        if last_3_prizes[-1] > 50000:
            return "一般", 0.8 # 上期太冷，下期防热，减注
            
        return "一般", 1.0 # 默认状态

# ==========================================
# 4. 策略生成器 (Strategy Generator)
# ==========================================

class RX9Optimizer:
    def __init__(self, data_source: DataSource, prob_engine: ProbabilityEngine):
        self.ds = data_source
        self.pe = prob_engine

    def generate_ticket(self, matches_14: List[MatchInfo], current_state: str):
        """
        生成最佳任选9方案
        """
        # 1. 计算每场比赛的评分
        match_analysis = []
        for m in matches_14:
            probs = self.pe.calculate_true_probs(m, current_state)
            
            # 安全分 (用于定胆)：胜率 * (1 - 联赛波动性)
            # 注意：engine.py 返回的 probs 中 key 是 '3', '1', '0'
            # volatility_index 也在 probs 中
            safety_score = probs['3'] * (1 - probs['volatility_index'])
            
            # 博冷分 (用于双选)：(平率 + 负率) * 联赛波动性
            value_score = (probs['1'] + probs['0']) * probs['volatility_index']
            
            match_analysis.append({
                'id': m.id,
                'league': m.league,
                'probs': probs,
                'safety_score': safety_score,
                'value_score': value_score,
                'match_obj': m
            })
            
        # 2. 选胆逻辑 (Banker Selection)
        # 按安全分排序，选前N个
        # 约束：如果处于"比较冷"周期，不仅看安全分，还要看该联赛的历史"胜"频次
        match_analysis.sort(key=lambda x: x['safety_score'], reverse=True)
        
        # 动态调整胆码数量：一般周期3胆，冷周期2胆
        num_bankers = 3 if current_state == '一般' else 2
        bankers = match_analysis[:num_bankers]
        
        # 3. 选拖逻辑 (Punter Selection)
        # 剩下的比赛中，按 博冷分 排序，选出最值得防冷的场次
        remaining = match_analysis[num_bankers:]
        remaining.sort(key=lambda x: x['value_score'], reverse=True)
        
        # 选取高价值的场次进行双选/全包覆盖
        # 这里简化演示：选取博冷分最高的6-7场，构建复式
        
        candidates = remaining[:10] # 选取候选池
        
        # 输出结构
        return {
            'strategy_name': f"周期:{current_state} - 智能胆拖",
            'bankers': [b['match_obj'].home_team for b in bankers],
            'focus_matches': [c['match_obj'].home_team for c in candidates],
            'note': "建议对候选场次采用双选(31或10)，重点防范高波动联赛的平局"
        }

# ==========================================
# 5. 主程序入口 (Main Execution)
# ==========================================

def run_system(current_odds_data):
    # 初始化
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    cycle_analyzer = CycleAnalyzer(ds)
    optimizer = RX9Optimizer(ds, pe)
    
    # 1. 研判周期
    state, fund_multiplier = cycle_analyzer.predict_current_state()
    print(f"当前系统研判状态: 【{state}】")
    print(f"建议资金系数: {fund_multiplier}倍")
    
    # 2. 读取目标期望 (基于 df_outcome_freq)
    target_freq = ds.df_outcome_freq.loc[state]
    print(f"本期目标模型分布 -> 胜:{target_freq['胜']:.1f}场, 平:{target_freq['平']:.1f}场, 负:{target_freq['负']:.1f}场")
    
    # 3. 生成策略
    ticket = optimizer.generate_ticket(current_odds_data, state)
    
    print("\n--- 推荐方案 ---")
    print(f"稳胆场次: {ticket['bankers']}")
    print(f"重点博冷场次: {ticket['focus_matches']}")
    print(ticket['note'])

if __name__ == "__main__":
    print("=== Strategy Generator 模块测试 ===")
    
    # 模拟输入数据
    mock_matches = [
        MatchInfo(1, '英超', '曼城', '伯恩利', [1.12, 6.5, 15.0]),
        MatchInfo(2, '非洲杯', '尼日利亚', '赤道几内亚', [1.80, 3.2, 4.5]),
        MatchInfo(3, '法乙', '波尔多', '亚眠', [2.10, 3.0, 3.6]),
        MatchInfo(4, '英超', '阿森纳', '切尔西', [2.5, 3.2, 2.8]),
        MatchInfo(5, '西甲', '皇马', '巴萨', [2.2, 3.4, 3.1]),
        MatchInfo(6, '意甲', '尤文', '米兰', [2.0, 3.1, 3.8]),
        MatchInfo(7, '德甲', '拜仁', '多特', [1.6, 4.0, 5.0]),
        MatchInfo(8, '英超', '利物浦', '曼联', [1.9, 3.5, 3.8]),
        MatchInfo(9, '法甲', '巴黎', '马赛', [1.4, 4.5, 7.0]),
        MatchInfo(10, '英冠', '莱斯特城', '伊普斯维奇', [2.1, 3.3, 3.4]),
        MatchInfo(11, '英冠', '南安普顿', '利兹联', [2.6, 3.2, 2.6]),
        MatchInfo(12, '荷甲', '埃因霍温', '费耶诺德', [1.8, 3.6, 4.0]),
        MatchInfo(13, '葡超', '本菲卡', '波尔图', [2.3, 3.1, 3.0]),
        MatchInfo(14, '亚冠', '利雅得胜利', '利雅得新月', [2.4, 3.4, 2.7])
    ]
    
    run_system(mock_matches)
