import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from data import DataSource
from strategy import RX9Optimizer
from engine import ProbabilityEngine, MatchInfo

class Ren9Backtest:
    """
    任选9回测引擎
    基于历史比赛数据和投注方案，计算核心评价指标
    """
    
    def __init__(self, data_source: DataSource, prob_engine: ProbabilityEngine):
        self.ds = data_source
        self.pe = prob_engine
        self.optimizer = RX9Optimizer(data_source, prob_engine)
        
        self.results = []
        self.period_results = []
        self.match_details = {} # 新增：存储每期的详细比赛结果 {period_id: [details]}
        
    def _get_match_result_code(self, result: str) -> str:
        """
        将比赛结果转换为投注代码
        '胜' -> '3', '平' -> '1', '负' -> '0'
        """
        result_map = {'胜': '3', '平': '1', '负': '0'}
        return result_map.get(result, '')
    
    def _check_bet_hit(self, bet: List[str], actual_result: str) -> bool:
        """
        检查投注是否命中
        :param bet: 投注选择，如 ['3'] 或 ['3', '1']
        :param actual_result: 实际结果代码，如 '3', '1', '0'
        :return: 是否命中
        """
        return actual_result in bet
    
    def _calculate_period_payout(self, period_id: int, bet_scheme: List[Dict], 
                                   actual_results: Dict[int, str]) -> Tuple[float, bool, List[Dict]]:
        """
        计算单期投注的奖金
        :param period_id: 期数ID
        :param bet_scheme: 投注方案，包含每场比赛的投注选择
        :param actual_results: 实际结果字典 {match_id: result_code}
        :return: (奖金金额, 是否中奖, 详细对比列表)
        """
        hits = 0
        details = []
        for match in bet_scheme:
            match_id = match['id']
            bet = match['bet']
            actual_result = actual_results.get(match_id, '')
            is_hit = self._check_bet_hit(bet, actual_result)
            
            if is_hit:
                hits += 1
            
            match_obj = match['match_obj']
            details.append({
                'id': match_id,
                'teams': f"{match_obj.home_team} vs {match_obj.away_team}",
                'bet': "".join(bet),
                'actual': actual_result,
                'is_hit': is_hit,
                'win_rate': match['win_rate'],
                'draw_rate': match['draw_rate'],
                'loss_rate': match['loss_rate'],
                'safety_score': match['safety_score'],
                'value_score': match['value_score']
            })
        
        # 任选9规则：必须9场全部命中
        is_winner = (hits == 9)
        total_payout = 0.0
        
        if is_winner:
            # 从 DataSource 获取该期的真实一等奖奖金
            total_payout = self.ds.get_period_bonus(period_id)
        
        return total_payout, is_winner, details
    
    def _prepare_matches_for_period(self, period_id: int) -> List[MatchInfo]:
        """
        为指定期数准备比赛数据
        :param period_id: 期数ID
        :return: MatchInfo对象列表
        """
        period_matches = self.ds.df_matches[self.ds.df_matches['期数id'] == period_id].head(14)
        
        matches = []
        for idx, row in enumerate(period_matches.itertuples(), start=1):
            matches.append(MatchInfo(
                id=idx,
                league=row.赛事,
                home_team=row.主队,
                away_team=row.客队,
                odds=[row.主胜SP值, row.主平SP值, row.主负SP值]
            ))
        
        return matches
    
    def _get_actual_results(self, period_id: int) -> Dict[int, str]:
        """
        获取指定期数的实际比赛结果
        :param period_id: 期数ID
        :return: {match_id: result_code}
        """
        period_matches = self.ds.df_matches[self.ds.df_matches['期数id'] == period_id].head(14)
        
        results = {}
        for idx, row in enumerate(period_matches.itertuples(), start=1):
            results[idx] = self._get_match_result_code(row.比赛结果)
        
        return results
    
    def run_backtest(self, period_ids: List[int], max_cost: int = 128, 
                     risk_tolerance: float = 0.5) -> pd.DataFrame:
        """
        运行回测
        :param period_ids: 要回测的期数ID列表
        :param max_cost: 单期最大投注成本
        :param risk_tolerance: 风险系数
        :return: 回测结果DataFrame
        """
        self.period_results = []
        
        for period_id in period_ids:
            matches = self._prepare_matches_for_period(period_id)
            actual_results = self._get_actual_results(period_id)
            
            bet_result = self.optimizer.generate_ticket(matches, max_cost, risk_tolerance)
            
            bet_df = bet_result['df']
            
            selected_matches = []
            for idx, row in bet_df.iterrows():
                bet_str = row['推荐']
                bet_list = list(bet_str)
                
                match_idx = int(row['场次']) - 1
                if match_idx < len(matches):
                    selected_matches.append({
                        'id': int(row['场次']),
                        'bet': bet_list,
                        'bet_type': row['类型'],
                        'match_obj': matches[match_idx],
                        'win_rate': row['胜率'],
                        'draw_rate': row['平率'],
                        'loss_rate': row['负率'],
                        'safety_score': row['安全分'],
                        'value_score': row['博冷分']
                    })
            
            payout, is_winner, details = self._calculate_period_payout(period_id, selected_matches, actual_results)
            self.match_details[period_id] = details
            
            cost = bet_result['total_cost']
            
            period_result = {
                '期数id': period_id,
                '投注成本': cost,
                '奖金': payout,
                '净收益': payout - cost,
                '是否中奖': is_winner,
                '投注场次数': len(selected_matches)
            }
            
            self.period_results.append(period_result)
        
        self.results = pd.DataFrame(self.period_results)
        return self.results
    
    def calculate_hit_rate(self) -> float:
        """
        计算命中率
        :return: 命中率 (0-1)
        """
        if len(self.results) == 0:
            return 0.0
        
        winning_periods = self.results['是否中奖'].sum()
        total_periods = len(self.results)
        
        return winning_periods / total_periods
    
    def calculate_roi(self) -> float:
        """
        计算ROI (Return on Investment)
        :return: ROI值
        """
        if len(self.results) == 0:
            return 0.0
        
        total_payout = self.results['奖金'].sum()
        total_cost = self.results['投注成本'].sum()
        
        if total_cost == 0:
            return 0.0
        
        return (total_payout - total_cost) / total_cost
    
    def calculate_max_drawdown(self) -> float:
        """
        计算最大回撤
        :return: 最大回撤值
        """
        if len(self.results) == 0:
            return 0.0
        
        cumulative_returns = self.results['净收益'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max.abs()
        
        max_drawdown = drawdown.min()
        
        return max_drawdown if not pd.isna(max_drawdown) else 0.0
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.03) -> float:
        """
        计算夏普比率
        :param risk_free_rate: 无风险利率 (默认3%)
        :return: 夏普比率
        """
        if len(self.results) == 0:
            return 0.0
        
        roi = self.calculate_roi()
        roi_std = self.results['净收益'].std()
        
        if roi_std == 0:
            return 0.0
        
        return (roi - risk_free_rate) / roi_std
    
    def calculate_cycle_capture_rate(self, high_bonus_threshold: float = 10000) -> float:
        """
        计算周期捕获率
        :param high_bonus_threshold: 高奖金阈值 (默认10,000元)
        :return: 周期捕获率 (0-1)
        """
        if len(self.results) == 0:
            return 0.0
        
        high_bonus_periods = self.results[self.results['奖金'] > high_bonus_threshold]
        
        if len(high_bonus_periods) == 0:
            return 0.0
        
        captured_high_bonus = high_bonus_periods['是否中奖'].sum()
        total_high_bonus = len(high_bonus_periods)
        
        return captured_high_bonus / total_high_bonus
    
    def generate_report(self) -> Dict:
        """
        生成回测报告
        :return: 包含所有核心指标的字典
        """
        report = {
            '命中率': self.calculate_hit_rate(),
            'ROI': self.calculate_roi(),
            '最大回撤': self.calculate_max_drawdown(),
            '夏普比率': self.calculate_sharpe_ratio(),
            '周期捕获率': self.calculate_cycle_capture_rate(),
            '总投注期数': len(self.results),
            '中奖期数': self.results['是否中奖'].sum() if len(self.results) > 0 else 0,
            '累计投入': self.results['投注成本'].sum() if len(self.results) > 0 else 0,
            '累计奖金': self.results['奖金'].sum() if len(self.results) > 0 else 0,
            '累计净收益': self.results['净收益'].sum() if len(self.results) > 0 else 0
        }
        
        return report
    
    def print_report(self):
        """
        打印回测报告
        """
        self.print_detailed_period_reports()
        
        report = self.generate_report()
        
        print("\n" + "=" * 50)
        print("【任选9 总体回测汇总报告】")
        print("=" * 50)
        print(f"总投注期数: {report['总投注期数']}")
        print(f"中奖期数: {report['中奖期数']}")
        print(f"命中率: {report['命中率']:.2%}")
        print(f"累计投入: {report['累计投入']:.2f} 元")
        print(f"累计奖金: {report['累计奖金']:.2f} 元")
        print(f"累计净收益: {report['累计净收益']:.2f} 元")
        print(f"ROI: {report['ROI']:.2%}")
        print(f"最大回撤: {report['最大回撤']:.2%}")
        print(f"夏普比率: {report['夏普比率']:.4f}")
        print(f"周期捕获率: {report['周期捕获率']:.2%}")
        print("=" * 50)

    def print_detailed_period_reports(self):
        """
        打印每一期的详细报告
        """
        print("\n" + "=" * 100)
        print("【任选9 各期详细投注对账单 & 预测深度分析】")
        print("=" * 100)
        
        for period_id, details in self.match_details.items():
            period_info = self.results[self.results['期数id'] == period_id].iloc[0]
            
            print(f"\n>>> 期数ID: {period_id}")
            print("-" * 100)
            header = f"{'场次':<4} {'对阵信息':<22} {'投注':<6} {'结果':<4} {'状态':<6} {'胜率':<8} {'平率':<8} {'负率':<8} {'安全分':<8} {'博冷分':<8}"
            print(header)
            
            hits = 0
            for d in details:
                status = "✅命中" if d['is_hit'] else "❌错"
                if d['is_hit']: hits += 1
                
                line = f"{d['id']:<4} {d['teams']:<22} {d['bet']:<6} {d['actual']:<4} {status:<6} {d['win_rate']:<8} {d['draw_rate']:<8} {d['loss_rate']:<8} {d['safety_score']:<8} {d['value_score']:<8}"
                print(line)
            
            accuracy = hits / len(details) if details else 0
            print("-" * 100)
            print(f"单期统计: 命中场数 {hits}/9 | 准确率: {accuracy:.2%}")
            print(f"资金情况: 投入 {period_info['投注成本']:.2f} | 奖金 {period_info['奖金']:.2f} | 净收益 {period_info['净收益']:.2f}")
            print(f"最终结果: {'🏆 中奖 (全部命中)' if period_info['是否中奖'] else '💀 未中 (有错失场次)'}")
            print("-" * 100)


if __name__ == "__main__":
    from data import DataSource
    from engine import ProbabilityEngine
    
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    backtest = Ren9Backtest(ds, pe)
    
    # 执行最近两期的回测
    period_ids = [25193, 25192]
    print(f"开始回测期数: {period_ids}...")
    
    results = backtest.run_backtest(period_ids, max_cost=256, risk_tolerance=0.5)
    
    print("\n回测概况:")
    print(results)
    
    backtest.print_detailed_period_reports()
    backtest.print_report()
