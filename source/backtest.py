import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from data import DataSource
from strategy import RX9Optimizer
from engine import ProbabilityEngine

class Ren9Backtest:
    """
    任选9回测引擎 (重构版)
    基于合并后的 DataFrame 数据和 RX9Optimizer 进行回测
    """
    
    def __init__(self, data_source: DataSource, prob_engine: ProbabilityEngine):
        self.ds = data_source
        self.pe = prob_engine
        self.optimizer = RX9Optimizer()
        
        self.results = []
        self.period_results = []
        self.match_details = {} # 存储每期的详细比赛结果 {period_id: [details]}
        self.current_params = {} # 存储当前运行参数
        
    def _get_match_result_code(self, result: str) -> str:
        """将比赛结果转换为投注代码 ('胜'->'3', '平'->'1', '负'->'0')"""
        result_map = {'胜': '3', '平': '1', '负': '0'}
        return result_map.get(result, '')
    
    def _check_bet_hit(self, bet: str, actual_result: str) -> bool:
        """检查投注是否命中"""
        return actual_result in bet
    
    def _calculate_period_payout(self, period_id: int, df_results: pd.DataFrame, 
                                   actual_results: Dict[int, str]) -> Tuple[float, bool, List[Dict]]:
        """
        计算单期投注的奖金
        :param period_id: 期数ID
        :param df_results: RX9Optimizer 返回的推荐方案 DataFrame
        :param actual_results: 实际结果字典 {match_index: result_code}
        :return: (奖金金额, 是否中奖, 详细对比列表)
        """
        hits = 0
        details = []
        
        for idx, row in df_results.iterrows():
            bet = row['推荐']
            # 注意：这里的 idx 是 df_period 中的索引 (0-13)
            actual_result = actual_results.get(idx, '')
            is_hit = self._check_bet_hit(bet, actual_result)
            
            if is_hit:
                hits += 1
            
            details.append({
                'id': idx + 1,
                'teams': f"{row['主队']} vs {row['客队']}",
                'bet': bet,
                'actual': actual_result,
                'is_hit': is_hit,
                'win_rate': row['胜率'],
                'draw_rate': row['平率'],
                'loss_rate': row['负率'],
                'safety_score': row['安全分'],
                'value_score': row['博冷分']
            })
        
        # 任选9规则：必须9场全部命中
        is_winner = (hits == 9)
        total_payout = 0.0
        
        if is_winner:
            # 从 DataSource 获取该期的真实一等奖奖金
            total_payout = self.ds.get_period_bonus(period_id)
        
        return total_payout, is_winner, details

    def run_backtest(self, period_ids: List[int], i: int = 5, j: int = 2, k: int = 1, l: int = 1, 
                     strategy_name: str = 'strategy_01', betting_scenario: str = 'all') -> pd.DataFrame:
        """
        运行回测
        :param period_ids: 要回测的期数ID列表
        :param i, j, k, l: 策略参数
        :param strategy_name: 策略名称 ('strategy_01', 'strategy_02')
        :param betting_scenario: 投注场景 ('all': 每一期均投注, 'only_cold': 只有预测冷热为1才投注)
        :return: 回测结果DataFrame
        """
        self.period_results = []
        self.match_details = {} 
        self.current_params = {'i': i, 'j': j, 'k': k, 'l': l, 'strategy': strategy_name, 'scenario': betting_scenario}
        
        # 确保预测算法已运行
        df_bonus = self.pe.predict_cold_warm()
        
        # 获取合并后的全量数据
        df_all = self.pe.get_merged_data()
        
        for period_id in period_ids:
            # 检查投注场景过滤
            if betting_scenario == 'only_cold':
                # 在 df_bonus 中查找该期的预测冷热
                bonus_row = df_bonus[df_bonus['期号'].astype(str) == str(period_id)]
                if bonus_row.empty or bonus_row.iloc[0].get('预测冷热', 0) != 1:
                    continue

            # 获取当期数据 (14场)
            df_period = df_all[df_all['期数id'] == period_id].head(14).reset_index(drop=True)
            
            if df_period.empty or len(df_period) < 9:
                continue
                
            # 获取实际结果
            actual_results = {idx: self._get_match_result_code(row['比赛结果']) 
                             for idx, row in df_period.iterrows()}
            
            # 生成投注方案
            try:
                bet_result = self.optimizer.generate_ticket(df_period, i, j, k, l, strategy_name=strategy_name)
            except Exception as e:
                # print(f"Error in generating ticket for {period_id}: {e}")
                continue
                
            df_bet = bet_result['df']
            if df_bet.empty:
                continue
            
            payout, is_winner, _ = self._calculate_period_payout(period_id, df_bet, actual_results)
            
            # 存储详情用于报告
            self.match_details[period_id] = bet_result['all_matches']
            
            cost = bet_result['total_cost']
            actual_bonus = self.ds.get_period_bonus(period_id)
            
            # 获取实际冷热信息用于显示
            bonus_info = df_bonus[df_bonus['期号'].astype(str) == str(period_id)]
            coldness = bonus_info.iloc[0]['赛果冷热'] if not bonus_info.empty else "未知"
            pred_cold = bonus_info.iloc[0]['预测冷热'] if not bonus_info.empty else 0
            
            self.period_results.append({
                '期数id': period_id,
                '投注成本': cost,
                '奖金': payout,
                '净收益': payout - cost,
                '是否中奖': is_winner,
                '当期一等奖': actual_bonus,
                '实际冷热': coldness,
                '预测冷热': pred_cold
            })
        
        self.results = pd.DataFrame(self.period_results)
        return self.results
    
    def generate_report(self) -> Dict:
        """生成回测报告"""
        if len(self.results) == 0:
            return {}
            
        total_payout = self.results['奖金'].sum()
        total_cost = self.results['投注成本'].sum()
        
        report = {
            '命中率': self.results['是否中奖'].mean(),
            'ROI': (total_payout - total_cost) / total_cost if total_cost > 0 else 0,
            '总投注期数': len(self.results),
            '中奖期数': self.results['是否中奖'].sum(),
            '累计投入': total_cost,
            '累计奖金': total_payout,
            '累计净收益': total_payout - total_cost
        }
        return report

    def calculate_win_cycles(self) -> Dict:
        """
        计算当前回测结果的中奖周期统计
        中奖周期 = 本次中奖期数id - 上次中奖期数id
        """
        if self.results is None or self.results.empty:
            return {'说明': '无回测结果'}

        wins_df = self.results[self.results['是否中奖'] == True]
        win_count = len(wins_df)

        if win_count < 2:
            return {
                '中奖次数': win_count,
                '中奖期数': wins_df['期数id'].tolist(),
                '平均周期': None,
                '说明': '中奖次数不足2次，无法计算周期'
            }

        win_periods = wins_df['期数id'].sort_values().tolist()
        cycles = [win_periods[i+1] - win_periods[i] for i in range(len(win_periods)-1)]

        return {
            '策略': self.current_params.get('strategy'),
            '场景': self.current_params.get('scenario'),
            '中奖期数': win_periods,
            '周期列表': cycles,
            '平均周期': np.mean(cycles),
            '最短周期': min(cycles),
            '最长周期': max(cycles),
            '周期标准差': np.std(cycles),
            '中奖次数': win_count,
            '总跨度': win_periods[-1] - win_periods[0]
        }

    def print_report(self):
        """打印回测报告"""
        report = self.generate_report()
        if not report:
            print("没有回测结果可供显示。")
            return

        print("\n" + "=" * 50)
        print("【任选9 总体回测汇总报告】")
        print("=" * 50)
        print(f"策略名称: {self.current_params.get('strategy')}")
        print(f"投注场景: {'每一期均投注' if self.current_params.get('scenario') == 'all' else '仅预测冷时投注'}")
        print(f"参数设置: i={self.current_params.get('i')}, j={self.current_params.get('j')}, "
              f"k={self.current_params.get('k')}, l={self.current_params.get('l')}")
        print(f"总投注期数: {report['总投注期数']}")
        print(f"中奖期数: {report['中奖期数']}")
        print(f"命中率: {report['命中率']:.2%}")
        print(f"累计投入: {report['累计投入']:.2f} 元")
        print(f"累计奖金: {report['累计奖金']:.2f} 元")
        print(f"累计净收益: {report['累计净收益']:.2f} 元")
        print(f"ROI: {report['ROI']:.2%}")
        print("=" * 50)

def calculate_win_cycle_stats(winning_periods_map: dict) -> dict:
    """
    计算各策略的平均中奖周期（基于 winning_periods_map）

    Args:
        winning_periods_map: {(strategy, scenario): set(中奖期号)}

    Returns:
        {key: 统计字典}
    """
    stats = {}

    for key, wins in winning_periods_map.items():
        if len(wins) < 2:
            stats[key] = {
                '中奖期数': sorted(wins),
                '周期列表': [],
                '平均周期': None,
                '最短周期': None,
                '最长周期': None,
                '周期标准差': None,
                '说明': '中奖次数不足2次，无法计算周期'
            }
            continue

        sorted_wins = sorted(wins)
        cycles = [sorted_wins[i+1] - sorted_wins[i] for i in range(len(sorted_wins)-1)]

        stats[key] = {
            '中奖期数': sorted_wins,
            '周期列表': cycles,
            '平均周期': np.mean(cycles),
            '最短周期': min(cycles),
            '最长周期': max(cycles),
            '周期标准差': np.std(cycles),
            '总跨度': sorted_wins[-1] - sorted_wins[0]
        }

    return stats

def print_win_cycle_report(win_cycle_stats: dict):
    """打印中奖周期统计报告"""
    print("\n" + "="*80)
    print("【各策略中奖周期统计】")
    print("="*80)

    for key, stat in win_cycle_stats.items():
        strategy, scenario = key
        scenario_name = '每期投注' if scenario == 'all' else '仅预测冷'

        print(f"\n策略: {strategy} | 场景: {scenario_name}")

        if stat['周期列表']:
            print(f"  中奖期数: {stat['中奖期数']}")
            print(f"  周期列表: {stat['周期列表']}")
            print(f"  平均周期: {stat['平均周期']:.2f} 期")
            print(f"  最短周期: {stat['最短周期']} 期")
            print(f"  最长周期: {stat['最长周期']} 期")
            print(f"  周期标准差: {stat['周期标准差']:.2f}")
            print(f"  总跨度: {stat['总跨度']} 期")
        else:
            print(f"  中奖期数: {stat['中奖期数']}")
            print(f"  说明: {stat['说明']}")

    # 汇总对比
    print("\n" + "-"*80)
    print("【中奖周期对比摘要】")
    print("-"*80)

    valid_stats = {k: v for k, v in win_cycle_stats.items() if v['周期列表']}

    if valid_stats:
        # 找出最优策略
        best_avg = min(valid_stats.items(), key=lambda x: x[1]['平均周期'])
        worst_avg = max(valid_stats.items(), key=lambda x: x[1]['平均周期'])

        print(f"平均周期最短: {best_avg[0]} ({best_avg[1]['平均周期']:.2f}期)")
        print(f"平均周期最长: {worst_avg[0]} ({worst_avg[1]['平均周期']:.2f}期)")
    else:
        print("所有策略中奖次数均不足2次，无法进行对比")

    print("="*80)

if __name__ == "__main__":
    from data import DataSource
    from engine import ProbabilityEngine
    
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    backtester = Ren9Backtest(ds, pe)
    
    # 获取可用的期号 (排除掉没有比赛结果的期号)
    available_periods = ds.df_matches.dropna(subset=['比赛结果'])['期数id'].unique().tolist()
    available_periods = [int(p) for p in available_periods if p < 26018]
    available_periods.sort()
    
    # 取最近的 20 期进行回测 (或者根据实际数据量调整)
    test_periods = available_periods[:] if len(available_periods) > 20 else available_periods
    
    # 为不同策略配置独立的 (i, j, k, l) 参数
    strategy_configs = {
        'strategy_01': {'i': 1, 'j': 3, 'k': 3, 'l': 2},
        'strategy_02': {'i': 1, 'j': 3, 'k': 4, 'l': 1},
        'strategy_03': {'i': 2, 'j': 3, 'k': 3, 'l': 1}
    }
    scenarios = ['all', 'only_cold']
    
    summary_results = []
    # 用于记录每个 (策略, 场景) 的中奖期号集合
    winning_periods_map = {}
    
    print(f"开始全量回测对比 (共 {len(test_periods)} 期)...")
    
    for strategy, params in strategy_configs.items():
        for scenario in scenarios:
            print(f"\n运行: {strategy} | {scenario}")
            # 使用策略各自独立的参数进行对比
            results = backtester.run_backtest(test_periods, 
                                    i=params['i'], j=params['j'], k=params['k'], l=params['l'], 
                                    strategy_name=strategy, betting_scenario=scenario)
            backtester.print_report()
            
            # 记录中奖期号
            wins = set(results[results['是否中奖'] == True]['期数id'].tolist())
            winning_periods_map[(strategy, scenario)] = wins
            
            # 收集摘要数据
            report = backtester.generate_report()
            if report:
                summary_results.append({
                    '策略': strategy,
                    '参数': f"i={params['i']},j={params['j']},k={params['k']},l={params['l']}",
                    '场景': '每期投注' if scenario == 'all' else '仅预测冷',
                    '期数': report['总投注期数'],
                    '中奖': report['中奖期数'],
                    '命中率': f"{report['命中率']:.2%}",
                    'ROI': f"{report['ROI']:.2%}"
                })
    
    print("\n" + "="*80)
    print("【回测结果最终对比摘要】")
    print("="*80)
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))
    
    # 统计策略间中奖期号的交集
    print("\n" + "="*80)    
    print("【策略中奖交集统计】")
    print("="*80)
    for scenario in scenarios:
        scenario_name = '每期投注' if scenario == 'all' else '仅预测冷'
        s1_wins = winning_periods_map.get(('strategy_01', scenario), set())
        s2_wins = winning_periods_map.get(('strategy_02', scenario), set())
        s3_wins = winning_periods_map.get(('strategy_03', scenario), set())

        # 三策略交集
        intersection_123 = s1_wins.intersection(s2_wins).intersection(s3_wins)
        # 两两交集
        intersection_12 = s1_wins.intersection(s2_wins)
        intersection_13 = s1_wins.intersection(s3_wins)
        intersection_23 = s2_wins.intersection(s3_wins)

        print(f"场景: {scenario_name}")
        print(f"  - strategy_01 中奖期数: {len(s1_wins)}")
        print(f"  - strategy_02 中奖期数: {len(s2_wins)}")
        print(f"  - strategy_03 中奖期数: {len(s3_wins)}")
        print(f"  - 三个策略共同中奖期数: {len(intersection_123)}")
        print(f"  - S1&S2共同中奖期数: {len(intersection_12)}")
        print(f"  - S1&S3共同中奖期数: {len(intersection_13)}")
        print(f"  - S2&S3共同中奖期数: {len(intersection_23)}")
        if intersection_123:
            print(f"  - 三策略共同中奖期号: {sorted(list(intersection_123))}")
        print("-" * 40)
    print("="*80)

    # 统计各策略中奖周期
    win_cycle_stats = calculate_win_cycle_stats(winning_periods_map)
    print_win_cycle_report(win_cycle_stats)
