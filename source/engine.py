import pandas as pd
from data import DataSource
from bet import ColdnessPredictor

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
            
        # 执行合并：基于'赛事'和'期数id'列进行左连接
        # df_matches 包含比赛细节，df_leagues 包含从 CSV 加载的联赛统计特征
        df_merged = pd.merge(
            self.ds.df_matches, 
            self.ds.df_leagues, 
            on=['赛事', '期数id'], 
            how='left'
        )
        
        # 针对 df_leagues 为空的字段通过填充 0 解决
        if not self.ds.df_leagues.empty:
            league_cols = [col for col in self.ds.df_leagues.columns if col not in ['赛事', '期数id']]
            df_merged[league_cols] = df_merged[league_cols].fillna(0)
            
        return df_merged

    def get_merged_data(self) -> pd.DataFrame:
        """获取合并后的完整数据"""
        return self.df_data

    def predict_cold_warm(self) -> pd.DataFrame:
        """
        使用机器学习模型预测赛果冷热
        1. 使用 ColdnessPredictor 进行滚动窗口预测
        2. 将预测结果(0/1)更新至 DataSource.df_bonus
        """
        if not hasattr(self.ds, 'df_bonus') or self.ds.df_bonus.empty:
            return pd.DataFrame()

        # 确保按期号排序
        df = self.ds.df_bonus.sort_values('期号').reset_index(drop=True)
        
        # 定义实际冷热：比较冷/超级冷 为 1，一般 为 0
        df['实际冷热'] = df['赛果冷热'].apply(lambda x: 1 if x in ['比较冷', '超级冷'] else 0)
        
        # 初始化预测列
        df['预测冷热'] = 0
        df['预测类型'] = '机器学习(Rolling 5CV)'
        df['预测概率'] = 0.0
        
        # 初始化预测器
        predictor = ColdnessPredictor(threshold=0.5)
        predictor.prepare_data()
        
        # 批量预测所有期号
        period_ids = df['期号'].tolist()
        
        # 逐个调用预测 (因为模型需要依赖历史窗口，必须确保 predictor 内部数据包含历史)
        # 注意：df['期号'] 对应的是 csv 中的 '期数id'
        
        print("正在执行机器学习冷热预测...")
        for i, row in df.iterrows():
            pid = row['期号']
            pred, prob = predictor.predict_single(pid)
            
            df.at[i, '预测冷热'] = pred
            df.at[i, '预测概率'] = prob
            
            # 记录一下无法预测的情况 (比如历史数据不足)
            if prob == 0.0 and pred == 0:
                 df.at[i, '预测类型'] = '数据不足/一般'

        # 将结果更新回 DataSource
        # 注意：这里需要确保返回值的逻辑与原有保持一致
        # 原逻辑：df['预测冷热'] = df['预测冷热'].map(lambda x: abs(x-1))  <-- 这是一个巨大的坑！
        # 原逻辑中：
        #   预测为 1 (冷门) -> abs(1-1) = 0
        #   预测为 0 (一般) -> abs(0-1) = 1
        #   也就是说，原代码最终返回的 '预测冷热'：0 代表预测冷门，1 代表预测一般
        #   这与直觉完全相反，但必须排查下游代码是否依赖这个反转逻辑。
        
        # 检查 user 提供的 snippet:
        # 规则 1.1: N-1 期为比较冷/超级冷 -> 预测为 1
        # ...
        # df['预测冷热'] = df['预测冷热'].map(lambda x: abs(x-1))
        # 也就是最终存入的是反转值。
        
        # 为了保持兼容性，我先保留这个反转逻辑，并在文档字符串中注明。
        # TODO: 建议后续重构掉这个反人类的逻辑，但本次任务仅替换预测算法。
        
        df['原始预测值'] = df['预测冷热'] # 保留一份直观的 (1=冷, 0=热)
        df['预测冷热'] = df['预测冷热'].map(lambda x: abs(x-1))
        
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
