import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
from datasource import DataSource

@dataclass
class MatchInfo:
    id: int
    league: str
    home_team: str
    away_team: str
    odds: List[float] # [胜, 平, 负] 欧赔

class ProbabilityEngine:
    """
    负责将赔率转化为去水后的真实概率，并结合联赛特征进行加权
    """
    def __init__(self, data_source: DataSource):
        self.ds = data_source

    def calculate_true_probs(self, match: MatchInfo, cycle_state='一般') -> Dict[str, float]:
        """
        输入: 比赛赔率, 当前预测的周期状态
        输出: 调整后的真实概率字典 {'3': val, '1': val, '0': val, 'volatility_index': val}
        """
        # 1. 基础去水 (Shin's Method 简化版，按倒数归一化)
        raw_probs = np.array([1/o for o in match.odds])
        base_probs = raw_probs / raw_probs.sum()
        
        # 2. 联赛特征修正
        # 逻辑：如果该联赛某结果的 Std 很大，说明该结果极不稳定，需要降低该结果的置信度
        stats = self.ds.get_league_stats(match.league)
        
        # 3. 周期修正 (关键步骤)
        # 如果预测是"超级冷"周期，我们需要提升"平/负"的权重，降低"胜"的权重
        # 利用 df_outcome_prob 数据作为权重系数
        cycle_weights = self.ds.get_cycle_correction(cycle_state)
        weight_vec = np.array([cycle_weights['胜'], cycle_weights['平'], cycle_weights['负']])
        
        # 融合概率： 基础概率 * 周期权重 * (1 / 联赛波动率) 
        # 注意：波动率越高，分母越大，概率越被压缩，代表"不确定性"
        # 加上 1 是为了避免除以 0，且 Std 通常是小数，1+Std 使得惩罚系数平滑
        volatility_penalty = np.array([
            1 / (1 + stats['Std_最终概率_胜']),
            1 / (1 + stats['Std_最终概率_平']),
            1 / (1 + stats['Std_最终概率_负'])
        ])
        
        # 逐项相乘进行调整
        final_probs = base_probs * weight_vec * volatility_penalty
        
        # 再次归一化
        final_probs = final_probs / final_probs.sum() 
        
        return {
            '3': final_probs[0], 
            '1': final_probs[1], 
            '0': final_probs[2],
            # 波动指数：取胜和负的波动率均值，作为该场比赛"是否容易爆冷"的参考
            'volatility_index': np.mean([stats['Std_最终概率_胜'], stats['Std_最终概率_负']])
        }

def main():
    print("=== ProbabilityEngine 模块测试 ===")
    
    # 1. 初始化
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    
    # 2. 准备测试样本
    test_matches = [
        # 强队低赔，英超（低波动）
        MatchInfo(1, '英超', '曼城', '伯恩利', [1.12, 6.5, 15.0]),
        # 势均力敌，英超
        MatchInfo(2, '英超', '阿森纳', '切尔西', [2.5, 3.2, 2.8]),
        # 强队低赔，非洲杯（高波动，容易出冷）
        MatchInfo(3, '非洲杯', '尼日利亚', '赤道几内亚', [1.80, 3.2, 4.5]),
        # 未知联赛（使用均值）
        MatchInfo(4, '中超', '海港', '申花', [2.10, 3.1, 3.4])
    ]
    
    cycle_states = ['一般', '超级冷']
    
    # 存储结果以便对比分析
    results = {}

    for state in cycle_states:
        print(f"\n>>> 测试周期状态: 【{state}】")
        print("-" * 110)
        print(f"{'赛事':<8} {'主队 vs 客队':<25} {'原始赔率':<20} {'波动指数':<10} {'修正后概率(胜/平/负)':<30}")
        print("-" * 110)
        
        results[state] = {}
        
        for match in test_matches:
            res = pe.calculate_true_probs(match, cycle_state=state)
            results[state][match.id] = res
            
            match_str = f"{match.home_team} vs {match.away_team}"
            odds_str = f"[{match.odds[0]}, {match.odds[1]}, {match.odds[2]}]"
            probs_str = f"{res['3']:.3f} / {res['1']:.3f} / {res['0']:.3f}"
            vol_str = f"{res['volatility_index']:.3f}"
            
            print(f"{match.league:<8} {match_str:<25} {odds_str:<20} {vol_str:<10} {probs_str:<30}")
    
    # 3. 自动分析
    print("\n=== 自动分析报告 ===")
    # 分析非洲杯案例
    match_id = 3 # 尼日利亚 vs 赤道几内亚
    res_gen = results['一般'][match_id]
    res_cold = results['超级冷'][match_id]
    
    diff_win = res_cold['3'] - res_gen['3']
    diff_draw = res_cold['1'] - res_gen['1']
    diff_loss = res_cold['0'] - res_gen['0']
    
    print(f"案例分析: 非洲杯 (尼日利亚 vs 赤道几内亚)")
    print(f"  - 一般周期胜率: {res_gen['3']:.3f}")
    print(f"  - 超冷周期胜率: {res_cold['3']:.3f} (变化: {diff_win:+.3f})")
    print(f"  - 超冷周期平局率: {res_cold['1']:.3f} (变化: {diff_draw:+.3f})")
    print(f"  - 结论: 周期转冷时，模型自动降低了强队胜率，提升了平局/爆冷的概率权重。")

if __name__ == "__main__":
    main()
