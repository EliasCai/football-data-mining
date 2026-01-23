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
        # 0. 基础检查：任选9至少需要9场比赛
        if not matches_14 or len(matches_14) < 9:
            return {
                'df': pd.DataFrame(),
                'total_notes': 0,
                'total_cost': 0
            }

        # 0.1 获取目标赛果分布 (Draw Target)
        target_stats = self.ds.get_target_frequency('一般')
        target_draw_count = target_stats['平'] # e.g. 3.14
        
        # 1. 计算每场比赛的评分
        match_analysis = []
        for m in matches_14:
            probs = self.pe.calculate_true_probs(m, '一般')
            
            # 预处理：获取 胜/平/负 的排序
            p_map = {'3': probs['3'], '1': probs['1'], '0': probs['0']}
            sorted_choices = sorted(p_map.items(), key=lambda x: x[1], reverse=True)
            
            match_analysis.append({
                'id': m.id,
                'league': m.league,
                'home': m.home_team,
                'away': m.away_team,
                'odds': m.odds,
                'probs': probs, 
                'choices': sorted_choices, # [(choice, prob), ...]
                'match_obj': m,
                'bet': [],
                'bet_type': '未定'
            })

        # 1.1 增强步骤：利用 df_leagues 优化风险和价值评分
        # 引入联赛的统计特性（卡方检验P值、真实频率偏差）来修正单场评分
        if hasattr(self.ds, 'df_leagues') and not self.ds.df_leagues.empty:
            for item in match_analysis:
                league = item['league']
                # 查找联赛数据
                league_rows = self.ds.df_leagues[self.ds.df_leagues['赛事'] == league]
                
                if not league_rows.empty:
                    stats = league_rows.iloc[0]
                    
                    # --- 优化 Safety Score (基于 P值) ---
                    # P值越高(>0.05)，说明观察到的频数与理论分布无显著差异，即联赛比较"正路"，符合赔率预期。
                    # P值越低(<0.05)，说明联赛赛果分布显著偏离理论值，可能存在某种系统性偏差或冷门多。
                    # 逻辑：P值越高，Safety Score 越高；P值极低，Safety Score 降低。
                    p_val = stats.get('P值', 0.5)
                    
                    # 设定一个调节系数：P值 0.5 为基准
                    # 如果 P=0.8 -> 1 + (0.3)*0.2 = 1.06 (提升6%)
                    # 如果 P=0.03 -> 1 + (-0.47)*0.2 = 0.906 (降低9.4%)
                    safety_factor = 1.0 + (p_val - 0.5) * 0.2
                    
                    # 应用修正，但限制幅度在 0.8 ~ 1.2 之间
                    safety_factor = max(0.8, min(1.2, safety_factor))
                    item['probs']['safety_score'] *= safety_factor
                    
                    # --- 优化 Value Score (基于真实频率偏差) ---
                    # 计算 平/负 的真实频率与理论概率的偏差
                    # Bias > 0 表示该结果真实发生频率高于赔率暗示的概率，是被市场低估的选项，具有博取价值。
                    bias_draw = stats.get('真实频率_平', 0) - stats.get('理论概率_平', 0)
                    bias_loss = stats.get('真实频率_负', 0) - stats.get('理论概率_负', 0)
                    
                    # 基础博冷分
                    current_value = item['probs']['value_score']
                    
                    # 如果平局偏差显著 (>3%)，且当前赔率支持（比如不是极低赔），增加博冷分
                    if bias_draw > 0.03:
                        current_value += bias_draw * 0.5 # 偏差越大加分越多
                        
                    # 如果客胜偏差显著
                    if bias_loss > 0.03:
                        current_value += bias_loss * 0.5
                        
                    item['probs']['value_score'] = current_value
            
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
            match['bet'] = [match['choices'][0][0]] # 选第1优选
            match['bet_type'] = '单选'

        # ---------------------------------------------------------
        # 优化步骤 A：自适应冷门预警与稳胆保护 (Adaptive Trap Detection)
        # ---------------------------------------------------------
        # 基于统计分布动态计算阈值，避免硬编码导致的过拟合
        
        # 计算当前比赛集合的统计特征
        all_safety_scores = [m['probs']['safety_score'] for m in active_analysis]
        all_first_probs = [m['choices'][0][1] for m in active_analysis]
        all_draw_probs = [m['probs']['1'] for m in active_analysis]
        
        if not all_safety_scores:
            # 防御性代码：如果由于某种原因列表为空，使用保守的默认值
            safety_threshold = 0.2
            prob_high_threshold = 0.7
            draw_threshold = 0.3
        else:
            # 动态阈值：基于分位数而非固定值
            # risk_tolerance 越高（激进），阈值越宽松；越低（保守），阈值越严格
            safety_threshold = np.percentile(all_safety_scores, 30)  # 安全分低于30%分位数视为高风险
            prob_high_threshold = np.percentile(all_first_probs, 70)  # 概率高于70%分位数视为高概率
            draw_threshold = np.percentile(all_draw_probs, 60)  # 平局率高于60%分位数需要关注
        
        # 根据 risk_tolerance 动态调整敏感度
        # risk_tolerance=0（保守）：阈值更严格，更容易触发双选
        # risk_tolerance=1（激进）：阈值更宽松，更倾向于单选
        safety_threshold *= (1.2 - risk_tolerance * 0.4)
        prob_high_threshold *= (1.1 - risk_tolerance * 0.2)
        draw_threshold *= (0.9 + risk_tolerance * 0.2)
        
        # 针对高胜率但有"陷阱"嫌疑的比赛进行强制双选
        for match in active_analysis:
            first_choice_prob = match['choices'][0][1]
            safety_score = match['probs']['safety_score']
            value_score = match['probs']['value_score']
            draw_prob = match['probs']['1']
            league = match['league']
            
            # 改进的陷阱检测逻辑：
            # 1. 胜率极高但安全分极低 -> 典型陷阱
            # 2. 胜率中高且博冷分高 -> 存在冷门倾向
            # 3. 赔率分布与真实胜率分布不匹配 (通过 safety_score 体现)
            
            is_trap = (
                (first_choice_prob > prob_high_threshold and safety_score < safety_threshold) or
                (first_choice_prob > 0.65 and safety_score < np.percentile(all_safety_scores, 50)) # 中等胜率但安全性一般
            )
            
            # 自适应平局保护：考虑联赛特性
            league_draw_bias = 0
            if hasattr(self.ds, 'df_leagues') and not self.ds.df_leagues.empty:
                league_rows = self.ds.df_leagues[self.ds.df_leagues['赛事'] == league]
                if not league_rows.empty:
                    stats = league_rows.iloc[0]
                    league_draw_bias = stats.get('真实频率_平', 0) - stats.get('理论概率_平', 0)
            
            # 平局保护阈值根据联赛特性动态调整
            dynamic_draw_threshold = draw_threshold * (0.85 if league_draw_bias > 0.03 else 1.0)
            needs_draw_protection = (
                (safety_score < safety_threshold and draw_prob > dynamic_draw_threshold) or
                (first_choice_prob > prob_high_threshold and draw_prob > dynamic_draw_threshold * 0.8) or
                (league_draw_bias > 0.05 and draw_prob > np.percentile(all_draw_probs, 40)) # 联赛平局多且本场平局率不低
            )
            
            if is_trap:
                if current_notes * 2 <= max_notes:
                    second_choice = match['choices'][1][0]
                    if second_choice not in match['bet']:
                        match['bet'].append(second_choice)
                        match['bet_type'] = '双选(自适应陷阱防御)'
                        current_notes *= 2
            
            if needs_draw_protection and '1' not in match['bet']:
                if current_notes * 2 <= max_notes:
                    match['bet'].append('1')
                    match['bet_type'] = '双选(自适应平局保护)'
                    current_notes *= 2

        # ---------------------------------------------------------
        # 优化步骤 B：中高概率场次强制双选 (Medium-High Probability Protection)
        # ---------------------------------------------------------
        # 新增：对于概率在55%-75%的比赛（最容易翻车的区间），强制双选
        current_draws = sum(1 for m in active_analysis if '1' in m['bet'])
        min_draws_target = int(target_draw_count)
        
        medium_high_candidates = [m for m in active_analysis if len(m['bet']) == 1]
        for match in medium_high_candidates:
            first_choice_prob = match['choices'][0][1]
            if first_choice_prob > 0.55 and first_choice_prob < 0.75:
                if current_notes * 2 <= max_notes:
                    second_choice = match['choices'][1][0]
                    if second_choice not in match['bet']:
                        match['bet'].append(second_choice)
                        match['bet_type'] = '双选(中高概率保护)'
                        current_notes *= 2
                        if second_choice == '1': current_draws += 1

        # ---------------------------------------------------------
        # 优化步骤 C：复合权重升级 (Compound Weight Upgrade)
        # ---------------------------------------------------------
        # 对仍是单选的场次，计算复合升级权重：风险程度 * 价值潜力
        candidates_to_upgrade = [m for m in active_analysis if len(m['bet']) == 1]
        
        def calc_upgrade_weight(m):
            # 自适应权重：基于risk_tolerance动态调整风险与价值的平衡
            # risk_tolerance=0（保守）：风险权重70%，价值权重30%
            # risk_tolerance=1（激进）：风险权重30%，价值权重70%
            risk_factor = (1 - m['probs']['safety_score'])
            value_factor = m['probs']['value_score']
            
            # 动态权重分配
            risk_weight = 0.7 - risk_tolerance * 0.4
            value_weight = 0.3 + risk_tolerance * 0.4
            
            # 如果是平局候选，根据目标平局数给予额外权重
            # 保守策略更重视平局覆盖，激进策略相对弱化
            base_draw_weight = 1.5 + (1 - risk_tolerance) * 0.5  # 保守时最高2.0，激进时最低1.5
            draw_weight = base_draw_weight if (m['choices'][1][0] == '1' and current_draws < min_draws_target) else 1.0
            
            return (risk_factor * risk_weight + value_factor * value_weight) * draw_weight

        candidates_to_upgrade.sort(key=calc_upgrade_weight, reverse=True)
        
        for match in candidates_to_upgrade:
            if current_notes * 2 <= max_notes:
                second_choice = match['choices'][1][0]
                match['bet'].append(second_choice)
                match['bet_type'] = '双选(复合优化)'
                current_notes *= 2
                if second_choice == '1': current_draws += 1
            else:
                break

        # ---------------------------------------------------------
        # 优化步骤 D：全包逻辑 (Triple Bet / All-in)
        # ---------------------------------------------------------
        # 如果还有较多资金，将最不稳的双选升级为全包
        candidates_to_triple = [m for m in active_analysis if len(m['bet']) == 2]
        candidates_to_triple.sort(key=lambda x: x['probs']['safety_score']) # 最不安全的优先
        
        for match in candidates_to_triple:
            # 双选(2注)变全包(3注)，注数增加 1.5倍
            if current_notes * 1.5 <= max_notes:
                third_choice = match['choices'][2][0]
                match['bet'].append(third_choice)
                match['bet_type'] = '全包'
                current_notes = int(current_notes * 1.5)
            else:
                break
        
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

        
        # 准备所有14场比赛的完整数据（包含未选中的5场）
        all_matches_data = []
        for analysis in match_analysis:
            all_matches_data.append({
                'id': analysis['id'],
                'league': analysis['league'],
                'home': analysis['home'],
                'away': analysis['away'],
                'odds': analysis['odds'],
                'win_rate': f"{analysis['probs']['3']:.2%}",
                'draw_rate': f"{analysis['probs']['1']:.2%}",
                'loss_rate': f"{analysis['probs']['0']:.2%}",
                'safety_score': f"{analysis['probs']['safety_score']:.2f}",
                'value_score': f"{analysis['probs']['value_score']:.2f}",
                'is_selected': analysis in selected_matches,
                'bet': "".join(sorted(analysis['bet'], reverse=True)) if analysis['bet'] else "",
                'bet_type': analysis['bet_type']
            })
        
        return {
            'df': df_results,
            'total_notes': current_notes,
            'total_cost': current_notes * 2,
            'all_matches': all_matches_data  # 新增：所有14场比赛的完整数据
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
    
    # 使用 DataSource 中的真实数据
    ds = DataSource()
    if hasattr(ds, 'df_matches') and not ds.df_matches.empty:
        # 选取期数id为 25193 的比赛 (前14场)
        period_id = 25193
        df_period = ds.df_matches[ds.df_matches['期数id'] == period_id].reset_index(drop=True)
        
        if df_period.empty:
            print(f"Error: 未找到期号 {period_id} 的比赛数据")
        else:
            print(f">>> 加载真实历史数据: 期号 {period_id}, 共 {len(df_period)} 场")
            
            real_matches = []
            for idx, row in df_period.iterrows():
                # 构建 MatchInfo 对象
                # 注意：MatchInfo 需要 odds = [胜, 平, 负]
                odds = [float(row['主胜SP值']), float(row['主平SP值']), float(row['主负SP值'])]
                
                real_matches.append(MatchInfo(
                    id=idx + 1, # ID 从 1 开始
                    league=row['赛事'],
                    home_team=row['主队'],
                    away_team=row['客队'],
                    odds=odds
                ))
            
            # 测试不同参数
            print(f"\n>>> 测试场景 1: 低成本保守型 (上限 32元) - 期号 {period_id}")
            run_system(real_matches, max_cost=32, risk_tolerance=0.2)
            
            print(f"\n>>> 测试场景 2: 中等成本平衡型 (上限 128元) - 期号 {period_id}")
            run_system(real_matches, max_cost=128, risk_tolerance=0.5)
            
    else:
        print("Error: DataSource 中未找到 df_matches 数据，请检查 data.py")
