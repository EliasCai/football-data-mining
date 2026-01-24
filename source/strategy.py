import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class RX9Optimizer:
    """
    任选9 策略优化器 (重构版)
    
    支持基于 (i, j, k, l) 参数的组合策略：
    - i: 单选场数
    - j: 双选 (主平/客平) 场数
    - k: 双选 (主客) 场数
    - l: 全选 (310) 场数
    约束条件: i + j + k + l = 9
    """
    
    def __init__(self):
        self.strategies = {
            'XXX01': self._strategy_XXX01,
            'XXX02': self._strategy_XXX02
        }

    def generate_ticket(self, df_period: pd.DataFrame, i: int, j: int, k: int, l: int, strategy_name: str = 'XXX01') -> Dict[str, Any]:
        """
        核心策略生成接口
        
        Args:
            df_period: 当前期次的比赛数据 (包含概率和P值)
            i, j, k, l: 各类投注类型的场数
            strategy_name: 使用的策略名称
            
        Returns:
            Dict 包含方案详情、注数、成本及完整比赛数据
        """
        if i + j + k + l != 9:
            raise ValueError(f"投注场次总和必须为 9 (当前: {i}+{j}+{k}+{l}={i+j+k+l})")
            
        if strategy_name not in self.strategies:
            raise ValueError(f"未知的策略名称: {strategy_name}")
            
        # 预处理数据
        df = self._preprocess_data(df_period)
        
        # 调用具体策略逻辑
        return self.strategies[strategy_name](df, i, j, k, l)

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理与清洗"""
        df = df.copy()
        # 核心概率列转换与填充
        prob_cols = ['主胜概率', '主平概率', '主负概率']
        for col in prob_cols:
            df[col] = pd.to_numeric(df.get(col, 0.33), errors='coerce').fillna(0.33)
        
        # P值处理
        df['P值'] = pd.to_numeric(df.get('P值', 0.5), errors='coerce').fillna(0.5)
        
        # 初始化结果列
        df['推荐'] = ""
        df['类型'] = "未选"
        return df

    def _strategy_XXX01(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
        """
        策略 XXX01 核心逻辑：
        1. 全选 (l 场)：选取不确定性（熵）最高的场次，投注 310
        2. 双选平 (j 场)：按主平概率降序选取，投注“胜/负（取大者）+ 平”
        3. 双选主客 (k 场)：按胜负概率差 (abs) 升序选取，投注 30
        4. 单选稳胆 (i 场)：按概率最大值与 P值加权选取，投注 3 或 0
        """
        selected_indices = []

        # 1. 全选 (l 场): 计算不确定性最大的 l 场
        def _calc_entropy(row):
            probs = [row['主胜概率'], row['主平概率'], row['主负概率']]
            return -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            
        df['entropy'] = df.apply(_calc_entropy, axis=1)
        l_selected = df.sort_values('entropy', ascending=False).head(l).index.tolist()
        for idx in l_selected:
            df.at[idx, '推荐'] = "310"
            df.at[idx, '类型'] = "全选"
            selected_indices.append(idx)

        # 2. 双选 (主平/客平) (j 场): 按照主平概率倒序选择
        remaining = df.drop(selected_indices)
        j_selected = remaining.sort_values('主平概率', ascending=False).head(j).index.tolist()
        for idx in j_selected:
            row = df.loc[idx]
            main_choice = '3' if row['主胜概率'] >= row['主负概率'] else '0'
            df.at[idx, '推荐'] = "".join(sorted([main_choice, '1'], reverse=True))
            df.at[idx, '类型'] = "双选(主平/客平)"
            selected_indices.append(idx)

        # 3. 双选 (主客) (k 场): 按照 abs(主胜 - 主负) 升序选择
        remaining = df.drop(selected_indices)
        remaining['diff_wl'] = (remaining['主胜概率'] - remaining['主负概率']).abs()
        k_selected = remaining.sort_values('diff_wl', ascending=True).head(k).index.tolist()
        for idx in k_selected:
            df.at[idx, '推荐'] = "30"
            df.at[idx, '类型'] = "双选(主客)"
            selected_indices.append(idx)

        # 4. 单选 (i 场): 按照 max(主胜, 主负) 与 P值加权选择
        remaining = df.drop(selected_indices)
        remaining['single_score'] = remaining[['主胜概率', '主负概率']].max(axis=1) * (1 + remaining['P值'])
        i_selected = remaining.sort_values('single_score', ascending=False).head(i).index.tolist()
        for idx in i_selected:
            row = df.loc[idx]
            df.at[idx, '推荐'] = '3' if row['主胜概率'] >= row['主负概率'] else '0'
            df.at[idx, '类型'] = "单选"
            selected_indices.append(idx)

        return self._format_results(df, selected_indices)

    def _strategy_XXX02(self, df: pd.DataFrame, i: int, j: int, k: int, l: int) -> Dict[str, Any]:
        """
        策略 XXX02 核心逻辑：
        1. 全选 (l 场)：选取不确定性（熵）最高的场次，投注 310
        2. 双选平 (j 场)：按主平概率降序选取，投注“胜/负（取大者）+ 平”
        3. 双选主客 (k 场)：按胜负概率差 (abs) 升序选取，投注 30
        4. 单选博冷 (i 场)：按主负概率降序且 P值 > 0.5 选取，投注 0
        """
        selected_indices = []

        # 1. 全选 (l 场): 计算不确定性最大的 l 场
        def _calc_entropy(row):
            probs = [row['主胜概率'], row['主平概率'], row['主负概率']]
            return -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            
        df['entropy'] = df.apply(_calc_entropy, axis=1)
        l_selected = df.sort_values('entropy', ascending=False).head(l).index.tolist()
        for idx in l_selected:
            df.at[idx, '推荐'] = "310"
            df.at[idx, '类型'] = "全选"
            selected_indices.append(idx)

        # 2. 双选 (主平/客平) (j 场): 按照主平概率倒序选择最大的 j 场
        remaining = df.drop(selected_indices)
        j_selected = remaining.sort_values('主平概率', ascending=False).head(j).index.tolist()
        for idx in j_selected:
            row = df.loc[idx]
            main_choice = '3' if row['主胜概率'] >= row['主负概率'] else '0'
            df.at[idx, '推荐'] = "".join(sorted([main_choice, '1'], reverse=True))
            df.at[idx, '类型'] = "双选(主平/客平)"
            selected_indices.append(idx)

        # 3. 双选 (主客) (k 场): 按照 abs(主胜 - 主负) 的顺序选择 k 场
        remaining = df.drop(selected_indices)
        remaining['diff_wl'] = (remaining['主胜概率'] - remaining['主负概率']).abs()
        k_selected = remaining.sort_values('diff_wl', ascending=True).head(k).index.tolist()
        for idx in k_selected:
            df.at[idx, '推荐'] = "30"
            df.at[idx, '类型'] = "双选(主客)"
            selected_indices.append(idx)

        # 4. 单选 (i 场): 按照主负倒序选择最大的 i 场，并且 P值 > 0.5
        remaining = df.drop(selected_indices)
        # 过滤 P值 > 0.5
        remaining_filtered = remaining[remaining['P值'] > 0.5]
        if len(remaining_filtered) < i:
            # 如果符合条件的不足 i 场，降级处理或给出警告，这里为了鲁棒性，取剩下的主负最大的
            i_selected = remaining.sort_values('主负概率', ascending=False).head(i).index.tolist()
        else:
            i_selected = remaining_filtered.sort_values('主负概率', ascending=False).head(i).index.tolist()
            
        for idx in i_selected:
            df.at[idx, '推荐'] = "0"
            df.at[idx, '类型'] = "单选(博冷客胜)"
            selected_indices.append(idx)

        return self._format_results(df, selected_indices)

    def _format_results(self, df: pd.DataFrame, selected_indices: List[Any]) -> Dict[str, Any]:
        """整理并格式化输出结果"""
        df_results = df.loc[selected_indices].sort_index().copy()
        
        # 格式化展示列 (用于回测报告)
        df_results['胜率'] = df_results['主胜概率'].apply(lambda x: f"{x:.2%}")
        df_results['平率'] = df_results['主平概率'].apply(lambda x: f"{x:.2%}")
        df_results['负率'] = df_results['主负概率'].apply(lambda x: f"{x:.2%}")
        df_results['安全分'] = df_results['P值'].apply(lambda x: f"{x:.2f}")
        df_results['博冷分'] = df_results.get('entropy', 0).apply(lambda x: f"{x:.2f}")

        # 计算总注数
        notes = 1
        for bet in df_results['推荐']:
            notes *= len(bet)
            
        return {
            'df': df_results,
            'total_notes': notes,
            'total_cost': notes * 2,
            'all_matches': df.to_dict('records')
        }

def main():
    print("=== RX9Optimizer 重构版测试 ===")
    from engine import ProbabilityEngine
    from data import DataSource
    
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    df_merged = pe.get_merged_data()
    
    if df_merged.empty:
        print("错误: 无法获取合并后的数据")
        return
        
    # 选取第一个期号进行测试
    period_id = df_merged['期数id'].iloc[0]
    df_period = df_merged[df_merged['期数id'] == period_id].head(14).reset_index(drop=True)
    
    print(f"\n[测试] 模拟生成方案 - 期号: {period_id}")
    optimizer = RX9Optimizer()
    
    # 设定参数测试 1: i=5, j=2, k=1, l=1
    try:
        print("\n--- 测试方案 XXX01 (5+2+1+1) ---")
        result = optimizer.generate_ticket(df_period, i=5, j=2, k=1, l=1, strategy_name='XXX01')
        print(f"总注数: {result['total_notes']}, 成本: {result['total_cost']} 元")
        print(result['df'][['赛事', '主队', '客队', '推荐', '类型']])
        
        print("\n--- 测试方案 XXX02 (5+2+1+1) ---")
        result2 = optimizer.generate_ticket(df_period, i=5, j=2, k=1, l=1, strategy_name='XXX02')
        print(f"总注数: {result2['total_notes']}, 成本: {result2['total_cost']} 元")
        print(result2['df'][['赛事', '主队', '客队', '推荐', '类型']])
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    main()
