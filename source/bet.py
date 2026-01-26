from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix # Import new metrics
import pandas as pd

df3 = pd.read_csv("D:\\05-CodeProject\\football-data-mining\\data\\lottery\\predict_lottery_cold.csv")

df3 = df3.set_index("期数id")
df3["N-1为1"] = df3["target"].shift(1)
df3["N-2为1"] = df3["target"].shift(2)

df3 = df3.dropna(axis=0)


# Prepare the data
X = df3.iloc[:,[0,1,2,3,4,5,6,7,8,11,12]]  # Features are the counts of matches in each category
y = df3['target']          # Target is whether it's '冷门' (1) or '一般' (0)

# --- Step 1: Check Class Distribution ---
print("\nTarget variable class distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))

# Calculate the split index for sequential splitting
split_index = int(len(X) * (1 - 0.3)) # 0.3 for test size

# Split data into training and testing sets sequentially
X_train = X.iloc[:split_index].to_numpy()
X_test = X.iloc[split_index:].to_numpy()
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# --- Step 2 & 3: Initialize and train the Logistic Regression model with class_weight ---
# Using 'liblinear' solver which is good for small datasets and handles L1/L2 penalties
# Added class_weight='balanced' to handle class imbalance
model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Get feature importances (coefficients)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

# Sort by absolute coefficient value for better visualization
feature_importance['Absolute_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values(by='Absolute_Coefficient', ascending=False)

print("\n特征重要性（逻辑回归系数）:")
print(feature_importance[['Feature', 'Coefficient']].to_markdown(index=False))


# Calculate and print the classification report
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# Calculate and print the confusion matrix
print("\n混淆矩阵:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
