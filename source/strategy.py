import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from data import DataSource
from engine import ProbabilityEngine, MatchInfo
import itertools

# ==========================================
# 4. 策略生成器 (Strategy Generator)
# ==========================================

class RX9Optimizer:
    def __init__(self, data_source: DataSource, prob_engine: ProbabilityEngine):
        self.ds = data_source
        self.pe = prob_engine

    def generate_ticket(self, matches_14: List[MatchInfo], max_cost: int = 128, risk_tolerance: float = 0.5):
        """
        生成最佳任选9方案
        :param matches_14: 14场比赛信息
        :param max_cost: 仓位成本上限（元），每注2元
        :param risk_tolerance: 风险系数 (0-1)，0为极度保守，1为极度激进
        """
        # 0. 获取目标赛果分布 (Draw Target)
        target_stats = self.ds.get_target_frequency('一般')
        target_draw_count = target_stats['平'] # e.g. 3.14
        
        # 1. 计算每场比赛的评分
        match_analysis = []
        for m in matches_14:
            # 默认使用 '一般' 周期，因为已移除周期研判
            probs = self.pe.calculate_true_probs(m, '一般')
            
            match_analysis.append({
                'id': m.id,
                'league': m.league,
                'home': m.home_team,
                'away': m.away_team,
                'odds': m.odds,
                'probs': probs, # 包含 safety_score 和 value_score
                'match_obj': m,
                'bet': [],
                'bet_type': '未定'
            })
            
        # 2. 策略逻辑：基于 safety_score 和 value_score 决定投注
        # 风险系数影响胆码的选择阈值
        
        # 将比赛按 safety_score 排序
        match_analysis.sort(key=lambda x: x['probs']['safety_score'], reverse=True)
        
        # =========================================================
        # 任选9核心规则：仅选择9场比赛进行投注
        # =========================================================
        # 策略：选取 Safety Score 最高的9场比赛作为基础
        # 未入选的5场比赛将被标记为 "避战" (Skipped)
        
        selected_matches = match_analysis[:9]
        skipped_matches = match_analysis[9:]
        
        # 标记未选场次
        for m in skipped_matches:
            m['bet'] = []
            m['bet_type'] = '避战'
            
        # 仅对入选的9场比赛进行后续的资金分配和优化
        active_analysis = selected_matches
        
        max_notes = max_cost // 2 # 最大注数
        
        # 初始化所有比赛为单选（选概率最高的项）
        current_notes = 1
        
        for match in active_analysis:
            probs = match['probs']
            # 找出概率最高的项
            p_map = {'3': probs['3'], '1': probs['1'], '0': probs['0']}
            best_choice = max(p_map, key=p_map.get)
            
            # 初始状态：每场只选概率最高的
            match['bet'] = [best_choice]
            match['bet_type'] = '单选'

        # ---------------------------------------------------------
        # 优化步骤：平局补全 (Draw Coverage)
        # ---------------------------------------------------------
        
        # 统计当前选了多少个平局
        current_draws = sum(1 for m in active_analysis if '1' in m['bet'])
        min_draws_target = int(target_draw_count) # 向下取整，至少保证这些
        
        # 找出尚未选平局的比赛，按平局概率从高到低排序
        draw_candidates = [m for m in active_analysis if '1' not in m['bet']]
        draw_candidates.sort(key=lambda x: x['probs']['1'], reverse=True)
        
        # 尝试通过升级为双选来补充平局
        draws_needed = max(0, min_draws_target - current_draws)
        
        for i in range(draws_needed):
            if i < len(draw_candidates):
                match = draw_candidates[i]
                
                # 检查资金
                if current_notes * 2 <= max_notes:
                    match['bet'].append('1')
                    match['bet_type'] = '双选(补平)'
                    current_notes *= 2
                    current_draws += 1
                else:
                    break
        
        # ---------------------------------------------------------
        # 常规升级步骤：基于风险和价值 (Existing Logic)
        # ---------------------------------------------------------
        
        # 重新排序，按“不稳程度”即 safety_score 升序（越不安全越需要防）
        candidates_to_upgrade = [m for m in active_analysis if len(m['bet']) < 2]
        candidates_to_upgrade.sort(key=lambda x: x['probs']['safety_score'])
        
        for match in candidates_to_upgrade:
            # 当前已选
            current_choice = match['bet'][0]
            probs = match['probs']
            
            # 寻找第二好的选项
            p_map = {'3': probs['3'], '1': probs['1'], '0': probs['0']}
            del p_map[current_choice]
            second_choice = max(p_map, key=p_map.get)
            
            # 试探性升级为双选
            if current_notes * 2 <= max_notes:
                match['bet'].append(second_choice)
                match['bet_type'] = '双选'
                current_notes *= 2
            else:
                break # 资金耗尽

        
        # 如果还有资金，尝试升级为全包？ (双选 -> 全包 需要 * 1.5 倍注数，即 2->3)
        # 这里简化，只做到双选。若需全包逻辑类似。
        
        # 3. 整理输出
        results_data = []
        
        # 我们希望按原始顺序输出，但只显示选中的9场，或者显示全部但标记状态
        # 用户需求是"输出方案"，通常只关心要买哪几场。
        # 这里只输出选中的9场，并按ID排序方便查找
        
        selected_matches.sort(key=lambda x: x['id'])
        
        for analysis in selected_matches:
            bet_str = "".join(sorted(analysis['bet'], reverse=True)) # 如 "31"
            
            results_data.append({
                '场次': analysis['id'],
                '赛事': analysis['league'],
                '主队': analysis['home'],
                '客队': analysis['away'],
                '欧赔': str(analysis['odds']),
                '胜率': f"{analysis['probs']['3']:.2%}",
                '平率': f"{analysis['probs']['1']:.2%}",
                '负率': f"{analysis['probs']['0']:.2%}",
                '安全分': f"{analysis['probs']['safety_score']:.2f}",
                '博冷分': f"{analysis['probs']['value_score']:.2f}",
                '推荐': bet_str,
                '类型': analysis['bet_type']
            })
            
        df_results = pd.DataFrame(results_data)

        
        return {
            'df': df_results,
            'total_notes': current_notes,
            'total_cost': current_notes * 2
        }

# ==========================================
# 5. 主程序入口 (Main Execution)
# ==========================================

def run_system(current_odds_data, max_cost=128, risk_tolerance=0.5):
    # 初始化
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    # 移除 CycleAnalyzer
    optimizer = RX9Optimizer(ds, pe)
    
    print(f"=== RX9-Alpha 策略生成 ===")
    print(f"参数设置: 成本上限 {max_cost}元, 风险系数 {risk_tolerance}")
    
    # 生成策略
    result = optimizer.generate_ticket(current_odds_data, max_cost, risk_tolerance)
    
    df = result['df']
    print("\n[推荐方案详情]")
    # 设置 pandas 显示参数
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    print(df)
    
    print("\n" + "="*40)
    print(f"总注数: {result['total_notes']} 注")
    print(f"总成本: {result['total_cost']} 元")
    print("="*40)

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
    
    # 测试不同参数
    print("\n>>> 测试场景 1: 低成本保守型 (上限 32元)")
    run_system(mock_matches, max_cost=32, risk_tolerance=0.2)
    
    print("\n>>> 测试场景 2: 中等成本平衡型 (上限 128元)")
    run_system(mock_matches, max_cost=128, risk_tolerance=0.5)
