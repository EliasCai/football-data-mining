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
    def __init__(self, window_size=200, n_splits=5, threshold=0.4):
        self.window_size = window_size
        self.n_splits = n_splits
        self.threshold = threshold
        self.feature_cols = None
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data = None

    def prepare_data(self, file_path=None):
        """
        加载并准备数据特征
        """
        if file_path is None:
            file_path = os.path.join(self.project_root, 'data', 'lottery', 'predict_lottery_cold.csv')
        
        if not os.path.exists(file_path):
            print(f"警告: 找不到预测数据文件 {file_path}")
            return None

        df = pd.read_csv(file_path)
        df = df.set_index("期数id")
        
        # 特征工程：增加滞后项 (N-1, N-2 期是否冷门)
        df["N-1为1"] = df["target"].shift(1)
        df["N-2为1"] = df["target"].shift(2)
        df = df.dropna(axis=0)

        # 选定特征列
        # 排除非特征列 '赛果冷热' 和 'target'
        self.feature_cols = [c for c in df.columns if c not in ['赛果冷热', 'target']]
        self.data = df
        return df

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
        final_pred = 1 if avg_prob >= self.threshold else 0
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
        
        print(f"开始滚动窗口回测评估 (测试期数: {test_size})...")
        
        for i in test_indices:
            pid = self.data.index[i]
            pred, prob = self.predict_single(pid)
            y_true.append(self.data['target'].iloc[i])
            y_pred.append(pred)
            period_ids.append(pid)
            
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

        print(f"\n指标汇总:")
        print(f"- 准确率 (Accuracy): {acc:.4f}")
        print(f"- 精确率 (Precision): {prec:.4f}")
        print(f"- 召回率 (Recall):    {rec:.4f}")
        print(f"- F1分数 (F1 Score):  {f1:.4f}")
        
        return pd.DataFrame({
            '期号': period_ids,
            '真实': y_true,
            '预测': y_pred
        })

if __name__ == "__main__":
    # 脚本模式下的自测逻辑
    predictor = ColdnessPredictor(threshold=0.4)
    predictor.run_evaluation(test_size=100)

