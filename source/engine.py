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

    def predict_cold_warm(self) -> pd.DataFrame:
        """
        预测赛果冷热并将结果更新至 DataSource.df_bonus
        1.1 如果 N-1 期为比较冷/超级冷，则 N 期预测为 1
        1.2 如果 N-1 至 N-3 期均为一般，则 N 期预测为 1
        1.3 其他情况均为 0
        """
        if not hasattr(self.ds, 'df_bonus') or self.ds.df_bonus.empty:
            return pd.DataFrame()

        # 确保按期号排序
        df = self.ds.df_bonus.sort_values('期号').reset_index(drop=True)
        
        # 定义实际冷热：比较冷/超级冷 为 1，一般 为 0
        df['实际冷热'] = df['赛果冷热'].apply(lambda x: 1 if x in ['比较冷', '超级冷'] else 0)
        
        # 初始化预测列
        df['预测冷热'] = 0
        df['预测类型'] = '其他'
        
        for i in range(len(df)):
            if i < 1:
                continue
            
            # 规则 1.1: N-1 期为比较冷/超级冷
            prev_1 = df.loc[i-1, '赛果冷热']
            if prev_1 in ['比较冷', '超级冷']:
                df.at[i, '预测冷热'] = 1
                df.at[i, '预测类型'] = '规则1.1(N-1冷)'
                continue
            
            # 规则 1.2: N-1 至 N-3 均为一般
            if i >= 3:
                prev_3 = df.loc[i-3:i-1, '赛果冷热'].tolist()
                if all(x == '一般' for x in prev_3):
                    df.at[i, '预测冷热'] = 1
                    df.at[i, '预测类型'] = '规则1.2(N-1~3一般)'
                    continue
        
        # 将结果更新回 DataSource
        self.ds.df_bonus = df
        return df

def main():
    print("=== ProbabilityEngine 重构与预测算法测试 ===")
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    
    # 1. 测试数据合并
    df_merged = pe.get_merged_data()
    print(f"\n[测试1] 合并后数据形状: {df_merged.shape}")
    
    # 2. 测试冷热预测算法
    print("\n[测试2] 赛果冷热预测混淆矩阵:")
    df_pred = pe.predict_cold_warm()
    
    if not df_pred.empty:
        # 使用 pd.crosstab 生成混淆矩阵
        confusion_matrix = pd.crosstab(
            df_pred['实际冷热'], 
            df_pred['预测冷热'], 
            rownames=['实际 (Actual)'], 
            colnames=['预测 (Predicted)'],
            margins=True
        )
        print(confusion_matrix)
        
        # 计算具体指标
        tp = confusion_matrix.loc[1, 1] if 1 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
        fp = confusion_matrix.loc[0, 1] if 0 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
        fn = confusion_matrix.loc[1, 0] if 1 in confusion_matrix.index and 0 in confusion_matrix.columns else 0
        tn = confusion_matrix.loc[0, 0] if 0 in confusion_matrix.index and 0 in confusion_matrix.columns else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n预测性能指标:")
        print(f"- 准确率 (Accuracy): {(tp + tn) / len(df_pred):.2%}")
        print(f"- 精准率 (Precision, 预测为冷且实际冷的比例): {precision:.2%}")
        print(f"- 召回率 (Recall, 实际冷且被预测出的比例): {recall:.2%}")
        
        print("\n预测类型分布:")
        print(df_pred['预测类型'].value_counts())
    
    # 3. 原始合并数据检查
    if not df_merged.empty:
        print("\n[测试3] 原始数据合并质量检查...")

if __name__ == "__main__":
    main()
