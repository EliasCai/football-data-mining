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

    def run_backtest(self, period_ids: List[int], i: int = 5, j: int = 2, k: int = 1, l: int = 1, strategy_name: str = 'XXX01') -> pd.DataFrame:
        """
        运行回测
        :param period_ids: 要回测的期数ID列表
        :param i, j, k, l: 策略参数
        :param strategy_name: 策略名称
        :return: 回测结果DataFrame
        """
        self.period_results = []
        self.match_details = {} 
        self.current_params = {'i': i, 'j': j, 'k': k, 'l': l, 'strategy': strategy_name}
        
        # 获取合并后的全量数据
        df_all = self.pe.get_merged_data()
        
        for period_id in period_ids:
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
            except Exception:
                continue
                
            df_bet = bet_result['df']
            if df_bet.empty:
                continue
            
            payout, is_winner, _ = self._calculate_period_payout(period_id, df_bet, actual_results)
            
            # 存储详情用于报告
            # 这里简化处理，直接存储推荐结果
            self.match_details[period_id] = bet_result['all_matches']
            
            cost = bet_result['total_cost']
            actual_bonus = self.ds.get_period_bonus(period_id)
            
            # 获取冷热信息
            bonus_info = self.ds.df_bonus[self.ds.df_bonus['期号'].astype(str) == str(period_id)]
            coldness = bonus_info.iloc[0]['赛果冷热'] if not bonus_info.empty else "未知"
            
            self.period_results.append({
                '期数id': period_id,
                '投注成本': cost,
                '奖金': payout,
                '净收益': payout - cost,
                '是否中奖': is_winner,
                '当期一等奖': actual_bonus,
                '赛果冷热': coldness
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

if __name__ == "__main__":
    from data import DataSource
    from engine import ProbabilityEngine
    
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    backtester = Ren9Backtest(ds, pe)
    
    # 获取可用的期号
    available_periods = ds.df_matches['期数id'].unique().tolist()
    test_periods = available_periods[:] # 测试前10期
    
    print(f"开始回测 {len(test_periods)} 期数据 (策略: XXX01)...")
    backtester.run_backtest(test_periods, i=1, j=3, k=3, l=2, strategy_name='XXX01')
    backtester.print_report()

    print(f"\n开始回测 {len(test_periods)} 期数据 (策略: XXX02)...")
    backtester.run_backtest(test_periods, i=1, j=3, k=4, l=1, strategy_name='XXX02')
    backtester.print_report()
