import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

class ColdnessPredictor:
    """
    冷热预测器：基于滚动窗口和 5CV 集成学习的逻辑回归模型
    """
    def __init__(self, window_size=200, n_splits=5, threshold=0.5):
        self.window_size = window_size
        self.n_splits = n_splits
        self.threshold = threshold
        self.feature_cols = None
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data = None

    def prepare_data(self, file_path=None, latest_df=None):
        """
        加载并准备数据特征。
        如果提供了 latest_df，则将其整合到历史数据中。
        """
        if file_path is None:
            file_path = os.path.join(self.project_root, 'data', 'lottery', 'predict_lottery_cold.csv')
        
        if not os.path.exists(file_path):
            print(f"警告: 找不到预测数据文件 {file_path}")
            return None

        df = pd.read_csv(file_path)
        
        # 加载奖金数据并合并
        results_path = os.path.join(self.project_root, 'data', 'lottery', 'football_lottery_results.csv')
        if os.path.exists(results_path):
            df_results = pd.read_csv(results_path)
            # 确保期号列一致
            df_results = df_results.rename(columns={'期号': '期数id'})
            df = df.merge(df_results[['期数id', '一等奖']], on='期数id', how='left')
        
        # 如果有最新一期的数据，进行整合
        if latest_df is not None:
            # 确保列顺序一致，且包含 target、赛果冷热、一等奖（填充为 NaN）
            for col in ['赛果冷热', 'target', '一等奖']:
                if col not in latest_df.columns:
                    latest_df[col] = np.nan
            
            # 保证列顺序与历史数据一致
            latest_df = latest_df[df.columns]
            df = pd.concat([df, latest_df], ignore_index=True)

        df = df.set_index("期数id")
        
        # 特征工程：增加滞后项 (N-1, N-2 期是否冷门)
        df["N-1为1"] = df["target"].shift(1)
        df["N-2为1"] = df["target"].shift(2)
        
        # 记录特征列名（不包含目标列和描述列）
        self.feature_cols = [c for c in df.columns if c not in ['赛果冷热', 'target', '一等奖']]
        
        # 删除特征中包含 NaN 的行（主要是前两期的滞后项）
        df = df.dropna(subset=self.feature_cols)
        
        # 对于训练和回测，我们需要删除 target 为空的行
        # 但如果是为了预测最后一期，我们需要保留最后一行
        self.data = df
        return df

    def process_overview_data(self, period_id: str) -> pd.DataFrame:
        """
        处理指定期数的概览数据，计算赛事类别分布。
        """
        # 1. 读取指定期数的概览数据
        file_path = os.path.join(self.project_root, 'data', 'overview', f"{period_id}.csv")
        if not os.path.exists(file_path):
            print(f"警告: 找不到期数 {period_id} 的概览文件 {file_path}")
            return pd.DataFrame()

        df_overview_period = pd.read_csv(file_path)

        # 2. 定义赛事与聚类标签的映射
        df_league_cluster_map = pd.DataFrame({
            "赛事": ["世亚预", "世俱杯", "世欧预", "亚冠", "德乙", "德甲", "意甲", "挪超", "欧冠", "欧协联", "欧国联", "欧洲杯", "欧联", "法乙", "法甲", "瑞典超", "美职", "英冠", "英超", "荷乙", "荷甲", "葡超", "西甲", "足总杯", "非洲杯"],
            "聚类标签": [2, 7, 3, 2, 4, 1, 1, 6, 7, 4, 5, 1, 1, 0, 5, 4, 0, 4, 1, 4, 6, 6, 5, 1, 6]
        }).set_index("赛事")

        # 3. 将赛事类别映射到概览数据中
        df_overview_period["赛事类别"] = df_overview_period["赛事"].map(
            lambda x: df_league_cluster_map.loc[x, "聚类标签"] if x in df_league_cluster_map.index else -1
        )

        # 4. 透视表格以获取每期各种赛事类别的比赛数量
        df_category_counts = df_overview_period.pivot_table(
            index="期数id",
            columns="赛事类别",
            aggfunc="count",
            values="比赛id",
            fill_value=0
        )

        # 5. 确保所有赛事类别列（-1到7）都存在，如果不存在则填充0
        for category_id in range(-1, 8):
            if category_id not in df_category_counts.columns:
                df_category_counts[category_id] = 0
        
        # 确保列顺序一致，便于后续模型使用
        df_category_counts = df_category_counts[sorted(df_category_counts.columns)]
        
        # 将 index 转换回列，以便后续合并
        df_category_counts = df_category_counts.reset_index()
        
        # 确保列名是字符串，与历史数据一致
        df_category_counts.columns = [str(c) if isinstance(c, int) else c for c in df_category_counts.columns]

        return df_category_counts

    def predict_latest(self, period_id: str):
        """
        整合最新数据并预测
        """
        # 1. 处理最新数据
        latest_df = self.process_overview_data(period_id)
        if latest_df.empty:
            return None, 0.0

        # 2. 整合历史数据与最新数据
        self.prepare_data(latest_df=latest_df)
        
        # 3. 进行预测
        try:
            pid = int(period_id)
        except ValueError:
            pid = period_id

        # 打印整合后的特征数据，确认无误
        print(f"\n期数 {period_id} 的整合特征:")
        print(self.data.loc[[pid], self.feature_cols].to_markdown())

        pred, prob = self.predict_single(pid)
        
        print(f"\n>>> 期数 {period_id} 预测结果:")
        print(f"    - 预测类别: {'冷门' if pred == 1 else '一般'} ({pred})")
        print(f"    - 冷门概率: {prob:.4f}")
        print(f"    - 判定阈值: {self.threshold}")
        
        return pred, prob

    def predict_single(self, target_period_id):
        """
        对单期进行预测：使用滚动窗口训练 + 5CV 平均概率
        """
        if self.data is None:
            self.prepare_data()
        
        if self.data is None or target_period_id not in self.data.index:
            return 0, 0.0

        target_idx = self.data.index.get_loc(target_period_id)
        
        # 如果数据量不足以支撑窗口大小，返回默认值
        if target_idx < self.window_size:
            return 0, 0.0

        # 获取训练窗口 [i - WINDOW_SIZE, i)
        train_start = target_idx - self.window_size
        train_end = target_idx
        
        X_all = self.data[self.feature_cols]
        y_all = self.data['target']
        
        X_train_window = X_all.iloc[train_start:train_end]
        y_train_window = y_all.iloc[train_start:train_end]
        
        # 确保训练数据中没有缺失值 (针对 target 列)
        valid_idx = y_train_window.notna()
        X_train_window = X_train_window[valid_idx]
        y_train_window = y_train_window[valid_idx]

        X_test_sample = X_all.iloc[target_idx:target_idx+1]
        
        # 5折时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        preds_prob = []
        
        for train_idx, _ in tscv.split(X_train_window):
            X_cv_train = X_train_window.iloc[train_idx]
            y_cv_train = y_train_window.iloc[train_idx]
            
            # 使用逻辑回归 (保持原算法要求)
            model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
            model.fit(X_cv_train, y_cv_train)
            
            # 预测类别 1 (冷门) 的概率
            prob = model.predict_proba(X_test_sample)[0][1]
            preds_prob.append(prob)
            
        # 取 5 次预测概率的平均值
        avg_prob = np.mean(preds_prob)
        final_pred = 1 if avg_prob < self.threshold else 0
        return final_pred, avg_prob

    def batch_predict(self, period_ids):
        """
        批量预测一组期号
        """
        if self.data is None:
            self.prepare_data()
            
        results = {}
        for pid in period_ids:
            pred, prob = self.predict_single(pid)
            results[pid] = {'pred': pred, 'prob': prob}
        return results

    def run_evaluation(self, test_size=100):
        """
        运行回测评估并输出指标
        """
        if self.data is None:
            self.prepare_data()
            
        test_indices = range(len(self.data) - test_size, len(self.data))
        y_true = []
        y_pred = []
        period_ids = []
        bonuses = []
        
        print(f"开始滚动窗口回测评估 (测试期数: {test_size})...")
        
        for i in test_indices:
            pid = self.data.index[i]
            pred, prob = self.predict_single(pid)
            y_true.append(self.data['target'].iloc[i])
            y_pred.append(pred)
            period_ids.append(pid)
            bonuses.append(self.data['一等奖'].iloc[i] if '一等奖' in self.data.columns else 0)
            
        # 输出分类报告
        print("\n" + "="*50)
        print(f"机器学习模型 - 最后 {test_size} 期预测评估")
        print("="*50)
        print(classification_report(y_true, y_pred))
        
        # 混淆矩阵
        print("\n混淆矩阵:")
        print(confusion_matrix(y_true, y_pred))
        
        # 综合指标
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 计算平均奖金
        df_eval = pd.DataFrame({
            'target': y_true,
            'pred': y_pred,
            'bonus': bonuses
        })
        
        avg_bonus_actual_cold = df_eval[df_eval['target'] == 1]['bonus'].mean()
        avg_bonus_pred_cold = df_eval[df_eval['pred'] == 1]['bonus'].mean()
        avg_bonus_tp = df_eval[(df_eval['target'] == 1) & (df_eval['pred'] == 1)]['bonus'].mean()

        print(f"\n指标汇总:")
        print(f"- 准确率 (Accuracy): {acc:.4f}")
        print(f"- 精确率 (Precision): {prec:.4f}")
        print(f"- 召回率 (Recall):    {rec:.4f}")
        print(f"- F1分数 (F1 Score):  {f1:.4f}")
        
        print(f"\n奖金统计 (冷门期):")
        print(f"- 实际冷门平均奖金: {avg_bonus_actual_cold:.2f} 元")
        print(f"- 预测冷门平均奖金: {avg_bonus_pred_cold:.2f} 元")
        print(f"- 命中冷门平均奖金: {avg_bonus_tp:.2f} 元")
        
        return pd.DataFrame({
            '期号': period_ids,
            '真实': y_true,
            '预测': y_pred,
            '奖金': bonuses
        })

if __name__ == "__main__":
    # 脚本模式下的自测逻辑
    predictor = ColdnessPredictor(threshold=0.5)
    
    # 1. 运行回测评估（可选）
    predictor.run_evaluation(test_size=100)
    
    # 2. 测试最新一期的预测
    id_to_process = "26020"
    predictor.predict_latest(id_to_process)

