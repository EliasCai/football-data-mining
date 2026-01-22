import pandas as pd
import numpy as np

class DataSource:
    """
    存储并提供所有静态历史数据，作为 RX9-Alpha 系统的核心参数源
    """
    def __init__(self):
        # 1. 联赛特征数据 (基于历史样本统计的均值和标准差)
        self.df_final_prob = pd.DataFrame({
            '赛事': ['英超', '英冠', '非洲杯', '亚冠', '英联杯', '西甲'],
            'Avg_最终概率_平': [0.2484, 0.2687, 0.2868, 0.2522, 0.2238, 0.2688],
            'Avg_最终概率_胜': [0.5402, 0.459, 0.5037, 0.5596, 0.5033, 0.5334],
            'Avg_最终概率_负': [0.4355, 0.358, 0.3535, 0.486, 0.4541, 0.4077],
            'Std_最终概率_平': [0.0356, 0.0356, 0.0416, 0.0393, 0.0064, 0.0459],
            'Std_最终概率_胜': [0.1772, 0.1339, 0.1699, 0.1867, 0.1337, 0.1683],
            'Std_最终概率_负': [0.1725, 0.1298, 0.2506, 0.1435, 0.1724, 0.1661]
        }).set_index('赛事')

        # 2. 历史奖金周期 (用于识别火锅奖规律)
        self.df_bonus = pd.DataFrame({
            '期号': ['25189', '25190', '25191', '25192', '25193'],
            '赛果冷热': ['比较冷', '一般', '一般', '一般', '超级冷'],
            '一等奖': [11210, 1906, 466, 195, 79934]
        })

        # 3. 赛果频率分布 (用于构建组合时的约束，定义每种状态下胜平负的平均场次)
        self.df_outcome_freq = pd.DataFrame({
            '赛果冷热统计': ['一般', '比较冷', '超级冷'],
            '胜': [6.84, 5.66, 5.42],
            '平': [3.14, 3.9, 4.29],
            '负': [4.2, 4.6, 4.34]
        }).set_index('赛果冷热统计')
        
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
    print("测试完成！")

if __name__ == "__main__":
    main()