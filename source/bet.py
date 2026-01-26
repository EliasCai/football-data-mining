from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import os

# 动态获取项目根目录并构建路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, 'data', 'lottery', 'predict_lottery_cold.csv')

df3 = pd.read_csv(csv_path)

df3 = df3.set_index("期数id")
df3["N-1为1"] = df3["target"].shift(1)
df3["N-2为1"] = df3["target"].shift(2)

df3 = df3.dropna(axis=0)

# Prepare the data
# Features are the counts of matches in each category
feature_cols = df3.columns[[0,1,2,3,4,5,6,7,8,11,12]]
X_all = df3.iloc[:,[0,1,2,3,4,5,6,7,8,11,12]]
y_all = df3['target']

# 需求配置
WINDOW_SIZE = 200    # N-200 窗口大小
TEST_SIZE = 100       # 预测最后 20 期
N_SPLITS = 5         # 5CV (5折交叉验证)

# 确保有足够的数据
if len(df3) < WINDOW_SIZE + TEST_SIZE:
    raise ValueError(f"数据量不足，需要至少 {WINDOW_SIZE + TEST_SIZE} 条数据，当前只有 {len(df3)} 条")

# 待预测的最后 20 期索引
test_indices = range(len(df3) - TEST_SIZE, len(df3))

y_true_all = []
y_pred_all = []
y_prob_all = []

print(f"开始滚动窗口预测评估...")
print(f"窗口大小: {WINDOW_SIZE}, 测试期数: {TEST_SIZE}, 交叉验证: {N_SPLITS}折 (TimeSeriesSplit)")

for i in test_indices:
    # 当前预测的期号索引为 i
    # 训练数据窗口：[i - WINDOW_SIZE, i)
    train_start = i - WINDOW_SIZE
    train_end = i
    
    X_train_window = X_all.iloc[train_start:train_end]
    y_train_window = y_all.iloc[train_start:train_end]
    
    # 当前测试样本
    X_test_sample = X_all.iloc[i:i+1]
    y_test_sample = y_all.iloc[i]
    
    # 5折时间序列交叉验证 (用于模型训练集成)
    # 注意：这里的 CV 是在 WINDOW_SIZE 的训练集内部进行的
    # 但用户的需求是 "5次建模的结果求平均"，通常指 Bagging 或者 CV 后的平均
    # 结合 "严格按照顺序拆分"，使用 TimeSeriesSplit 切分训练窗口
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    preds_prob = []
    
    # 在当前训练窗口内进行 5 次切分训练
    # 每次切分会使用窗口内不同长度的历史数据来训练模型
    # 然后预测同一个未来的样本 (即当前的 X_test_sample)
    
    for train_idx, _ in tscv.split(X_train_window):
        # 获取 CV 的训练子集
        X_cv_train = X_train_window.iloc[train_idx]
        y_cv_train = y_train_window.iloc[train_idx]
        
        # 训练模型
        model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
        model.fit(X_cv_train, y_cv_train)
        
        # 预测当前测试样本的概率 (关注类别 1 '冷门')
        prob = model.predict_proba(X_test_sample)[0][1]
        preds_prob.append(prob)
    
    # 取 5 次预测概率的平均值
    avg_prob = np.mean(preds_prob)
    final_pred = 1 if avg_prob >= 0.4 else 0
    
    y_true_all.append(y_test_sample)
    y_pred_all.append(final_pred)
    y_prob_all.append(avg_prob)
    
    current_period = df3.index[i]
    print(f"期号: {current_period} | 真实: {y_test_sample} | 预测: {final_pred} (概率: {avg_prob:.4f})")

# 评估指标
print("\n" + "="*50)
print(f"最后 {TEST_SIZE} 期预测评估结果")
print("="*50)

print("\n分类报告:")
print(classification_report(y_true_all, y_pred_all))

print("\n混淆矩阵:")
cm = confusion_matrix(y_true_all, y_pred_all)
print(cm)

# 额外指标
acc = accuracy_score(y_true_all, y_pred_all)
prec = precision_score(y_true_all, y_pred_all, zero_division=0)
rec = recall_score(y_true_all, y_pred_all, zero_division=0)
f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

print(f"\n准确率 (Accuracy): {acc:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"F1分数 (F1 Score):  {f1:.4f}")
