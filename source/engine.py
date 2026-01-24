import pandas as pd
from data import DataSource

class ProbabilityEngine:
    """
    数据合并引擎：负责获取 df_matches 并与 df_leagues 进行合并，提供统一的数据视图
    """
    def __init__(self, data_source: DataSource):
        self.ds = data_source
        self.df_data = self._prepare_data()

    def _prepare_data(self) -> pd.DataFrame:
        """
        合并 df_matches 和 df_leagues 数据
        """
        if not hasattr(self.ds, 'df_matches') or self.ds.df_matches.empty:
            return pd.DataFrame()
            
        # 执行合并：基于'赛事'列进行左连接
        # df_matches 包含比赛细节，df_leagues 包含联赛统计特征
        df_merged = pd.merge(
            self.ds.df_matches, 
            self.ds.df_leagues, 
            on='赛事', 
            how='left'
        )
        return df_merged

    def get_merged_data(self) -> pd.DataFrame:
        """获取合并后的完整数据"""
        return self.df_data

def main():
    print("=== ProbabilityEngine 重构测试 ===")
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    
    df_merged = pe.get_merged_data()
    
    print(f"\n[测试1] 合并后数据形状: {df_merged.shape}")
    
    if not df_merged.empty:
        print("\n[测试2] 前5行数据预览 (部分列):")
        # 选择一些代表性的列进行展示
        display_cols = ['期数id', '赛事', '主队', '客队', '比赛结果', '样本量', '理论概率_胜', '真实频率_胜', 'P值']
        available_cols = [c for c in display_cols if c in df_merged.columns]
        print(df_merged[available_cols].head())
        
        print("\n[测试3] 检查合并质量 (是否有缺失的联赛信息):")
        # 检查 '样本量' 是否为空，如果为空说明在 df_leagues 中没找到该赛事
        missing_leagues = df_merged[df_merged['样本量'].isna()]['赛事'].unique()
        if len(missing_leagues) > 0:
            print(f"以下赛事在 df_leagues 中未找到匹配 (正常现象，部分小联赛可能缺失统计):")
            print(missing_leagues)
        else:
            print("所有赛事均成功匹配联赛统计信息。")
            
        print(f"\n[测试4] 验证数据完整性:")
        print(f"原始比赛数: {len(ds.df_matches)}")
        print(f"合并后比赛数: {len(df_merged)}")
        
    else:
        print("\n警告: 合并后的数据集为空，请检查 DataSource 是否正确加载数据。")

if __name__ == "__main__":
    main()
