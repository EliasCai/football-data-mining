import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from data_sample import DataSource
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
        self.best_params = None # 新增：存储最佳参数组合
        self.current_params = {} # 新增：存储当前运行参数
        
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
    
    def _generate_all_matches_details(self, all_matches: List[MatchInfo], selected_matches: List[Dict], 
                                     actual_results: Dict[int, str], all_matches_data: List[Dict] = None) -> List[Dict]:
        """
        生成完整的14场比赛详情（包含未投注的比赛）
        :param all_matches: 所有14场比赛
        :param selected_matches: 被选中的9场比赛
        :param actual_results: 实际结果
        :return: 完整的14场比赛详情列表
        """
        # 创建选中比赛的映射，方便快速查找
        selected_map = {match['id']: match for match in selected_matches}
        
        # 创建所有比赛数据的映射，方便快速查找
        all_matches_map = {}
        if all_matches_data:
            all_matches_map = {match['id']: match for match in all_matches_data}
        
        all_details = []
        for match in all_matches:
            match_id = match.id
            
            if match_id in selected_map:
                # 这是被选中的比赛，使用原有的详细信息
                selected_match = selected_map[match_id]
                actual_result = actual_results.get(match_id, '')
                is_hit = self._check_bet_hit(selected_match['bet'], actual_result)
                
                all_details.append({
                    'id': match_id,
                    'teams': f"{match.home_team} vs {match.away_team}",
                    'bet': "".join(selected_match['bet']),
                    'actual': actual_result,
                    'is_hit': is_hit,
                    'win_rate': selected_match['win_rate'],
                    'draw_rate': selected_match['draw_rate'],
                    'loss_rate': selected_match['loss_rate'],
                    'safety_score': selected_match['safety_score'],
                    'value_score': selected_match['value_score'],
                    'is_selected': True,
                    'bet_type': selected_match['bet_type']
                })
            else:
                # 这是未选中的比赛，标记为"未投注"
                actual_result = actual_results.get(match_id, '')
                
                # 从all_matches_data获取统计信息
                match_data = all_matches_map.get(match_id, {})
                
                all_details.append({
                    'id': match_id,
                    'teams': f"{match.home_team} vs {match.away_team}",
                    'bet': "未投注",
                    'actual': actual_result,
                    'is_hit': False,  # 未投注的比赛不算命中
                    'win_rate': match_data.get('win_rate', 'N/A'),
                    'draw_rate': match_data.get('draw_rate', 'N/A'),
                    'loss_rate': match_data.get('loss_rate', 'N/A'),
                    'safety_score': match_data.get('safety_score', 'N/A'),
                    'value_score': match_data.get('value_score', 'N/A'),
                    'is_selected': False,
                    'bet_type': '未投注'
                })
        
        # 按场次ID排序
        all_details.sort(key=lambda x: x['id'])
        return all_details
    
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
        self.match_details = {} # 重置每期的详细比赛结果
        self.current_params = {'max_cost': max_cost, 'risk_tolerance': risk_tolerance}
        
        for period_id in period_ids:
            matches = self._prepare_matches_for_period(period_id)
            if not matches or len(matches) < 9:
                # print(f"警告: 期数 {period_id} 比赛数据不足 ({len(matches)}场)，跳过该期。")
                continue
                
            actual_results = self._get_actual_results(period_id)
            
            bet_result = self.optimizer.generate_ticket(matches, max_cost, risk_tolerance)
            
            bet_df = bet_result['df']
            if bet_df.empty:
                continue
            
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
            
            # 生成完整的14场比赛详情（包含未投注的比赛）
            all_matches_data = bet_result.get('all_matches', [])
            all_matches_details = self._generate_all_matches_details(matches, selected_matches, actual_results, all_matches_data)
            self.match_details[period_id] = all_matches_details
            
            cost = bet_result['total_cost']
            
            # 获取当期的一等奖金（作为预期奖金参考）
            actual_bonus = self.ds.get_period_bonus(period_id)
            
            # 获取当期的冷热统计信息
            bonus_info = self.ds.df_bonus[self.ds.df_bonus['期号'].astype(str) == str(period_id)]
            coldness = bonus_info.iloc[0]['赛果冷热'] if not bonus_info.empty else "未知"
            
            period_result = {
                '期数id': period_id,
                '投注成本': cost,
                '奖金': payout,
                '净收益': payout - cost,
                '是否中奖': is_winner,
                '投注场次数': len(selected_matches),
                '当期一等奖': actual_bonus,
                '赛果冷热': coldness
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
        
        if self.best_params:
            print("-" * 50)
            print("【历史最佳参数组合 (基于本次搜索)】")
            print(f"-> 最佳成本上限: {self.best_params['max_cost']}")
            print(f"-> 最佳风险系数: {self.best_params['risk_tolerance']}")
            print(f"-> 最高场次准确率: {self.best_params['match_accuracy']:.2%}")
            print(f"-> 预计ROI: {self.best_params['roi']:.2%}")
            
        print("=" * 50)

    def save_report_to_file(self, filename: str = "backtest_report.md"):
        """
        将回测报告保存到Markdown文件
        """
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# 任选9 回测深度分析报告\n\n")
            
            f.write("## 一、 核心指标汇总\n")
            f.write("| 指标名称 | 数值 |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| 总投注期数 | {report['总投注期数']} |\n")
            f.write(f"| 中奖期数 | {report['中奖期数']} |\n")
            f.write(f"| 命中率 | {report['命中率']:.2%} |\n")
            f.write(f"| 累计投入 | {report['累计投入']:.2f} 元 |\n")
            f.write(f"| 累计奖金 | {report['累计奖金']:.2f} 元 |\n")
            f.write(f"| 累计净收益 | {report['累计净收益']:.2f} 元 |\n")
            f.write(f"| ROI | {report['ROI']:.2%} |\n")
            f.write(f"| 最大回撤 | {report['最大回撤']:.2%} |\n")
            f.write(f"| 夏普比率 | {report['夏普比率']:.4f} |\n")
            f.write(f"| 周期捕获率 | {report['周期捕获率']:.2%} |\n\n")
            
            if self.best_params:
                f.write("## 二、 最优参数推荐\n")
                f.write("> 基于本次网格搜索生成的最佳配置\n\n")
                f.write(f"- **最佳成本上限**: {self.best_params['max_cost']} 元\n")
                f.write(f"- **最佳风险系数**: {self.best_params['risk_tolerance']}\n")
                f.write(f"- **最高场次准确率**: {self.best_params['match_accuracy']:.2%}\n")
                f.write(f"- **预计ROI**: {self.best_params['roi']:.2%}\n\n")

            if self.current_params:
                f.write("## 三、 当前运行参数\n")
                f.write(f"- **成本上限**: {self.current_params.get('max_cost')} 元\n")
                f.write(f"- **风险系数**: {self.current_params.get('risk_tolerance')}\n\n")

            f.write("## 四、 各期详细对账单\n")
            for period_id, details in self.match_details.items():
                period_matches = self.results[self.results['期数id'] == period_id]
                if period_matches.empty: continue
                info = period_matches.iloc[0]
                
                # 仅保留“一般冷”和“超级冷”的期数
                if info['赛果冷热'] not in ['一般冷', '超级冷']:
                    continue
                
                f.write(f"### 期数: {period_id} ({info['赛果冷热']})\n")
                f.write(f"- **结果**: {'🏆 中奖' if info['是否中奖'] else '💀 未中'}\n")
                f.write(f"- **盈亏**: 投入 {info['投注成本']:.2f} | 奖金 {info['奖金']:.2f} | 净收益 {info['净收益']:.2f}\n")
                f.write(f"- **预期奖金**: {info['当期一等奖']:.2f} (当期实际一等奖金)\n\n")
                
                f.write("| 场次 | 对阵 | 投注 | 结果 | 状态 | 胜率 | 平率 | 负率 | 安全分 | 博冷分 |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                
                selected_hits = 0
                selected_count = 0
                for d in details:
                    if d['is_selected']:
                        selected_count += 1
                        status = "✅命中" if d['is_hit'] else "❌错"
                        if d['is_hit']: selected_hits += 1
                    else:
                        status = "未投注"
                    
                    # 格式化输出，处理N/A值
                    win_rate = d['win_rate'] if d['win_rate'] != 'N/A' else 'N/A'
                    draw_rate = d['draw_rate'] if d['draw_rate'] != 'N/A' else 'N/A'
                    loss_rate = d['loss_rate'] if d['loss_rate'] != 'N/A' else 'N/A'
                    safety_score = d['safety_score'] if d['safety_score'] != 'N/A' else 'N/A'
                    value_score = d['value_score'] if d['value_score'] != 'N/A' else 'N/A'
                    
                    f.write(f"| {d['id']} | {d['teams']} | {d['bet']} | {d['actual']} | {status} | {win_rate} | {draw_rate} | {loss_rate} | {safety_score} | {value_score} |\n")
                
                accuracy = selected_hits / selected_count if selected_count > 0 else 0
                f.write(f"\n**单期统计**: 选中场数 {selected_count}/9 | 命中场数 {selected_hits}/9 | 选中准确率: {accuracy:.2%}\n\n")
                f.write("---\n\n")
                
        print(f"\n[系统] 报告已保存至: {filename}")

    def save_results_to_csv(self, filename: str = "backtest_results.csv"):
        """
        将期数结果保存到CSV文件
        """
        if not self.results.empty:
            self.results.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"[系统] 原始数据已导出至: {filename}")


    def print_detailed_period_reports(self):
        """
        打印每一期的详细报告（显示全部14场比赛）
        """
        print("\n" + "=" * 100)
        print("【任选9 各期详细投注对账单 & 预测深度分析】")
        if self.current_params:
            print(f"当前回测参数: 成本上限={self.current_params.get('max_cost')} | 风险系数={self.current_params.get('risk_tolerance')}")
        print("=" * 100)
        
        for period_id, details in self.match_details.items():
            # 检查是否有该期结果
            period_matches = self.results[self.results['期数id'] == period_id]
            if period_matches.empty:
                continue
            period_info = period_matches.iloc[0]
            
            # 仅保留“一般冷”和“超级冷”的期数
            if period_info['赛果冷热'] not in ['一般冷', '超级冷']:
                continue
            
            print(f"\n>>> 期数ID: {period_id} ({period_info['赛果冷热']})")
            print("-" * 100)
            header = f"{'场次':<4} {'对阵信息':<22} {'投注':<8} {'结果':<4} {'状态':<8} {'胜率':<8} {'平率':<8} {'负率':<8} {'安全分':<8} {'博冷分':<8}"
            print(header)
            
            selected_hits = 0
            selected_count = 0
            for d in details:
                if d['is_selected']:
                    selected_count += 1
                    status = "✅命中" if d['is_hit'] else "❌错"
                    if d['is_hit']: selected_hits += 1
                else:
                    status = "未投注"
                
                # 格式化输出，处理N/A值
                win_rate = d['win_rate'] if d['win_rate'] != 'N/A' else 'N/A'
                draw_rate = d['draw_rate'] if d['draw_rate'] != 'N/A' else 'N/A'
                loss_rate = d['loss_rate'] if d['loss_rate'] != 'N/A' else 'N/A'
                safety_score = d['safety_score'] if d['safety_score'] != 'N/A' else 'N/A'
                value_score = d['value_score'] if d['value_score'] != 'N/A' else 'N/A'
                
                line = f"{d['id']:<4} {d['teams']:<22} {d['bet']:<8} {d['actual']:<4} {status:<8} {win_rate:<8} {draw_rate:<8} {loss_rate:<8} {safety_score:<8} {value_score:<8}"
                print(line)
            
            accuracy = selected_hits / selected_count if selected_count > 0 else 0
            print("-" * 100)
            print(f"单期统计: 选中场数 {selected_count}/9 | 命中场数 {selected_hits}/9 | 选中准确率: {accuracy:.2%}")
            print(f"资金情况: 投入 {period_info['投注成本']:.2f} | 奖金 {period_info['奖金']:.2f} | 净收益 {period_info['净收益']:.2f}")
            print(f"预期奖金: {period_info['当期一等奖']:.2f} (当期实际一等奖金)")
            print(f"最终结果: {'🏆 中奖' if period_info['是否中奖'] else '💀 未中 (有错失场次)'}")
            print("-" * 100)

    def parameter_search(self, period_ids: List[int], cost_range: List[int], risk_range: List[float]):
        """
        参数搜索：寻找最佳参数组合
        :param period_ids: 回测期数
        :param cost_range: 成本上限范围
        :param risk_range: 风险系数范围
        :return: (最佳参数字典, 所有结果DataFrame)
        """
        print("\n" + "=" * 50)
        print("【自适应参数网格搜索中...】")
        print("=" * 50)
        
        best_roi = -float('inf')
        best_params = {}
        all_search_results = []

        for cost in cost_range:
            for risk in risk_range:
                # 运行回测（静默模式，不打印详细报告）
                self.run_backtest(period_ids, max_cost=cost, risk_tolerance=risk)
                
                # 计算综合指标
                total_hits = 0
                total_matches = 0
                for pid in period_ids:
                    details = self.match_details.get(pid, [])
                    total_hits += sum(1 for d in details if d['is_hit'])
                    total_matches += len(details)
                
                avg_accuracy = total_hits / total_matches if total_matches > 0 else 0
                winning_periods = int(self.results['是否中奖'].sum())
                total_periods = len(self.results)
                total_payout = self.results['奖金'].sum()
                roi = self.calculate_roi()
                total_cost = self.results['投注成本'].sum()
                total_notes = int(total_cost // 2)
                
                res = {
                    'max_cost': cost,
                    'risk_tolerance': round(risk, 2),
                    'match_accuracy': avg_accuracy,
                    'winning_periods': winning_periods,
                    'total_payout': total_payout,
                    'total_notes': total_notes,
                    'roi': roi,
                    'total_cost': total_cost
                }
                all_search_results.append(res)
                
                # 打印进度
                print(f"测试: cost={cost:<4} risk={risk:.2f} | 准确率: {avg_accuracy:.2%} | 投注: {total_notes:>4}注 | 中奖: {winning_periods}/{total_periods} | 奖金: {total_payout:>8.2f} | ROI: {roi:.2%}")
                
                # 寻找最优：优先ROI，ROI相同时优先准确率，再相同时优先低成本
                is_better = False
                if roi > best_roi:
                    is_better = True
                elif abs(roi - best_roi) < 1e-6:
                    if avg_accuracy > best_params.get('match_accuracy', -1.0):
                        is_better = True
                    elif abs(avg_accuracy - best_params.get('match_accuracy', -1.0)) < 1e-6:
                        if cost < best_params.get('max_cost', float('inf')):
                            is_better = True
                
                if is_better:
                    best_roi = roi
                    best_params = res
                    self.best_params = best_params # 存储到实例中以便在汇总报告中使用
        
        df_search = pd.DataFrame(all_search_results)
        
        print("\n" + "*" * 50)
        print("【搜索完成！最优参数推荐 (基于最高ROI)】")
        print(f"-> 最佳成本上限: {best_params['max_cost']}")
        print(f"-> 最佳风险系数: {best_params['risk_tolerance']}")
        print(f"-> 最高场次准确率: {best_params['match_accuracy']:.2%}")
        print(f"-> 累计中奖次数: {best_params['winning_periods']}")
        print(f"-> 累计中奖金额: {best_params['total_payout']:.2f}")
        print(f"-> 累计投注注数: {best_params['total_notes']}")
        print(f"-> 预计ROI: {best_params['roi']:.2%}")
        print("*" * 50)
        
        return best_params, df_search


if __name__ == "__main__":
    from data import DataSource
    from engine import ProbabilityEngine
    
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    backtest = Ren9Backtest(ds, pe)
    
    # 待搜索的期数
    period_ids = list(range(25001, 25080)) # 25194)) # 示例期数
    
    # 定义搜索范围
    cost_range = [128, 256] # 32, 64, 
    risk_range = np.linspace(0.1, 0.9, 9)
    
    # 执行参数搜索
    best_params, search_results = backtest.parameter_search(period_ids, cost_range, risk_range)
    
    # 使用最优参数运行最后一次回测并打印详细报告
    print("\n\n>>> 正在使用最优参数运行最终回测验证...")
    backtest.run_backtest(period_ids, 
                         max_cost=best_params['max_cost'], 
                         risk_tolerance=best_params['risk_tolerance'])
    
    # 显式更新 best_params，确保它包含最终运行的 ROI 和其它统计数据
    # 注意：parameter_search 内部已经更新了 self.best_params，
    # 但再次运行 run_backtest 后，我们可以确保 self.results 等状态与 best_params 一致
    backtest.print_report()
    
    # 保存报告到文件
    print("\n>>> 正在保存回测报告...")
    backtest.save_report_to_file("RX9_Backtest_Report.md")
    backtest.save_results_to_csv("RX9_Backtest_Data.csv")

