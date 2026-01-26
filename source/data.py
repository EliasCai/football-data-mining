import pandas as pd
import numpy as np
import os
from overview import MatchOverview

class DataSource:
    """
    存储并提供所有静态历史数据，作为 RX9-Alpha 系统的核心参数源
    """
    def __init__(self):
        # 获取项目根目录 (source 的上一级)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 1. 联赛特征数据 (基于历史样本统计的均值和标准差)
        self.df_final_prob = pd.DataFrame(
            {"赛事": ["J2联赛", "J联赛", "世亚预", "世南美预", "世欧预", "亚冠", "亚洲杯", "友谊赛", "国王杯", "天皇杯", "奥运女足", "奥运男足", "德乙", "德国杯", "德甲", "意大利杯", "意甲", "挪超", "欧冠", "欧协联", "欧国联", "欧洲杯", "欧联", "法乙", "法国杯", "法甲", "瑞典超", "美洲杯", "美职", "芬超", "英冠", "英甲", "英联杯", "英超", "荷乙", "荷甲", "葡超", "西甲", "解放者杯", "足总杯", "非洲杯"], "Avg_最终概率_平": [0.2912, 0.279, 0.2494, 0.2897, 0.2346, 0.2522, 0.264, 0.2487, 0.2601, 0.2466, 0.297, 0.2895, 0.2551, 0.2501, 0.2426, 0.2505, 0.2707, 0.2448, 0.2417, 0.249, 0.2694, 0.2681, 0.2495, 0.2793, 0.2618, 0.2496, 0.2546, 0.273, 0.2378, 0.235, 0.2687, 0.2736, 0.2238, 0.2484, 0.2382, 0.2389, 0.2683, 0.2688, 0.283, 0.2295, 0.2868], "Avg_最终概率_胜": [0.391, 0.4211, 0.6397, 0.5561, 0.6735, 0.5596, 0.6008, 0.638, 0.4693, 0.4691, 0.6489, 0.5376, 0.4529, 0.5187, 0.539, 0.5961, 0.503, 0.5369, 0.5542, 0.5389, 0.5902, 0.5226, 0.5297, 0.4556, 0.5812, 0.5081, 0.4829, 0.622, 0.5217, 0.567, 0.459, 0.4, 0.5033, 0.5402, 0.4854, 0.556, 0.5605, 0.5334, 0.5978, 0.6275, 0.5037], "Avg_最终概率_负": [0.4104, 0.4076, 0.586, 0.317, 0.612, 0.486, 0.5766, 0.2996, 0.3147, 0.4631, 0.4916, 0.4522, 0.3128, 0.4382, 0.414, 0.3113, 0.4452, 0.4118, 0.4214, 0.3192, 0.4277, 0.453, 0.383, 0.3364, 0.5429, 0.3835, 0.3831, 0.4309, 0.329, 0.4249, 0.358, -1.0, 0.4541, 0.4355, 0.363, 0.4625, 0.4592, 0.4077, 0.3007, 0.4636, 0.3535], "Std_最终概率_平": [0.0202, 0.0125, 0.0632, 0.0389, 0.0646, 0.0393, 0.0621, 0.0361, 0.0233, 0.0572, -1.0, 0.0458, 0.0159, 0.0109, 0.0429, 0.0679, 0.0472, 0.0379, 0.0465, 0.0513, 0.0423, 0.0642, 0.039, 0.0223, 0.0128, 0.0441, 0.0306, 0.0756, 0.0281, 0.032, 0.0356, 0.0243, 0.0064, 0.0356, 0.0336, 0.0499, 0.0486, 0.0459, 0.0367, 0.0397, 0.0416], "Std_最终概率_胜": [0.19, 0.0997, 0.1916, 0.2563, 0.2522, 0.1867, 0.2125, 0.2254, 0.1644, 0.2022, 0.2199, 0.1753, 0.1161, 0.1973, 0.1747, 0.1538, 0.1659, 0.1636, 0.1921, 0.1913, 0.1823, 0.2011, 0.1634, 0.1037, 0.1577, 0.1718, 0.1467, 0.1969, 0.1071, 0.0771, 0.1339, 0.1486, 0.1337, 0.1772, 0.1396, 0.1754, 0.1876, 0.1683, 0.1464, 0.2134, 0.1699], "Std_最终概率_负": [0.1295, 0.0941, 0.2385, 0.0262, 0.2372, 0.1435, 0.2212, 0.0996, 0.1307, 0.1968, 0.1907, 0.2266, 0.097, 0.2561, 0.1763, 0.1514, 0.1467, 0.1511, 0.1835, 0.1357, 0.172, 0.1808, 0.1535, 0.0863, 0.2862, 0.1569, 0.145, 0.2054, 0.0776, 0.1442, 0.1298, -1.0, 0.1724, 0.1725, 0.1391, 0.1858, 0.2106, 0.1661, 0.1428, 0.171, 0.2506]}
        ).set_index('赛事')

        # 2. 历史奖金周期 (用于识别火锅奖规律)
        csv_path = os.path.join(self.project_root, 'data', 'lottery', 'football_lottery_results.csv')
        try:
            self.df_bonus = pd.read_csv(csv_path)
            # 验证字段数量和名称
            expected_columns = ['期号', '赛果冷热', '一等奖']
            if not all(col in self.df_bonus.columns for col in expected_columns):
                print(f"警告: CSV 文件列名不匹配。预期: {expected_columns}, 实际: {list(self.df_bonus.columns)}")
                # 尝试按顺序映射列名
                self.df_bonus.columns = expected_columns[:len(self.df_bonus.columns)]
            
            # 验证并强制转换类型，确保与原逻辑一致
            self.df_bonus['期号'] = self.df_bonus['期号'].astype(int)
            self.df_bonus['赛果冷热'] = self.df_bonus['赛果冷热'].astype(str)
            self.df_bonus['一等奖'] = pd.to_numeric(self.df_bonus['一等奖'], errors='coerce').fillna(0).astype(int)
            
            # 仅保留需要的列
            self.df_bonus = self.df_bonus[expected_columns]
            
        except Exception as e:
            print(f"读取奖金数据失败: {e}。将使用空 DataFrame。")
            self.df_bonus = pd.DataFrame(columns=['期号', '赛果冷热', '一等奖'])

        # 3. 赛果频率分布 (用于构建组合时的约束，定义每种状态下胜平负的平均场次)
        self.df_outcome_freq = pd.DataFrame({
            '赛果冷热统计': ['一般', '比较冷', '超级冷'],
            '胜': [6.84, 5.66, 5.42],
            '平': [3.14, 3.9, 4.29],
            '负': [4.2, 4.6, 4.34]
        }).set_index('赛果冷热统计')

        # 5. 联赛详细统计 (从 CSV 加载)
        league_stats_path = os.path.join(self.project_root, 'data', 'league', 'rolling_chi2_analysis.csv')
        try:
            self.df_leagues = pd.read_csv(league_stats_path)
            # 筛选样本量大于 10 的数据
            if '样本量' in self.df_leagues.columns:
                self.df_leagues = self.df_leagues[self.df_leagues['样本量'] > 10]
            
            # 将 '当前期数id' 重命名为 '期数id' 以便后续合并
            if '当前期数id' in self.df_leagues.columns:
                self.df_leagues = self.df_leagues.rename(columns={'当前期数id': '期数id'})
                
        except Exception as e:
            print(f"读取联赛统计数据失败: {e}。将使用空 DataFrame。")
            self.df_leagues = pd.DataFrame()


        # 初始化处理器（可自定义路径、阈值、输出列）
        processor = MatchOverview(
            data_dir=os.path.join(self.project_root, 'data', 'overview'),
            issue_threshold=25000
        )
    
        self.df_matches = processor.load_data().process().get_output()

        
        # 4. 概率修正系数 (用于根据周期状态调整原始赔率隐含概率)
        self.df_outcome_prob = pd.DataFrame({
            '赛果冷热统计': ['一般', '比较冷', '超级冷'],
            '平': [0.26, 0.254, 0.252],
            '胜': [0.56, 0.513, 0.465],
            '负': [0.452, 0.393, 0.351]
        }).set_index('赛果冷热统计')

    def get_league_stats(self, league_name: str) -> pd.Series:
        """获取特定联赛的均值和方差，若无匹配则返回全局均值"""
        if league_name in self.df_final_prob.index:
            return self.df_final_prob.loc[league_name]
        return self.df_final_prob.mean()

    def get_cycle_correction(self, state: str) -> pd.Series:
        """根据周期状态（一般/比较冷/超级冷）获取概率修正系数"""
        if state in self.df_outcome_prob.index:
            return self.df_outcome_prob.loc[state]
        return self.df_outcome_prob.loc['一般']

    def get_target_frequency(self, state: str) -> pd.Series:
        """根据周期状态获取目标赛果分布频率"""
        if state in self.df_outcome_freq.index:
            return self.df_outcome_freq.loc[state]
        return self.df_outcome_freq.loc['一般']

    def get_period_bonus(self, period_id: int) -> float:
        """根据期数ID获取任选9一等奖奖金"""
        match = self.df_bonus[self.df_bonus['期号'] == period_id]
        if not match.empty:
            return float(match.iloc[0]['一等奖'])
        return 0.0

    def get_recent_bonus(self, n: int = 3) -> np.ndarray:
        """获取最近 N 期的奖金数据"""
        return self.df_bonus['一等奖'].tail(n).values

def main():
    print("=== DataSource 模块测试 ===")
    ds = DataSource()
    
    # 1. 测试联赛数据获取
    print("\n[测试1] 获取英超联赛特征:")
    print(ds.get_league_stats('英超'))
    
    print("\n[测试2] 获取未知联赛特征 (应返回均值):")
    print(ds.get_league_stats('中超'))
    
    # 2. 测试周期修正系数
    print("\n[测试3] 获取'超级冷'状态下的修正系数:")
    print(ds.get_cycle_correction('超级冷'))
    
    # 3. 测试目标频率
    print("\n[测试4] 获取'比较冷'状态下的目标频率:")
    print(ds.get_target_frequency('比较冷'))
    
    # 4. 数据完整性检查
    print("\n[测试5] 数据完整性检查:")
    print(f"联赛特征数据量: {len(ds.df_final_prob)}")
    print(f"历史奖金数据量: {len(ds.df_bonus)}")

    # 5. 测试新增数据结构 (df_leagues, df_matches)
    print("\n[测试6] 联赛详细统计 (df_leagues):")
    if hasattr(ds, 'df_leagues'):
        print(ds.df_leagues[['赛事', '样本量', '理论概率_胜']].head(3))
        print(f"Shape: {ds.df_leagues.shape}")
    else:
        print("未找到 df_leagues")

    print("\n[测试7] 历史比赛数据 (df_matches):")
    if hasattr(ds, 'df_matches'):
        print(ds.df_matches[['期数id', '赛事', '主队', '客队', '比赛结果']].head(3))
        print(f"Shape: {ds.df_matches.shape}")
        # print(ds.df_matches['期数id'].sort_values().unique())
    else:
        print("未找到 df_matches")

    print("测试完成！")

if __name__ == "__main__":
    main()