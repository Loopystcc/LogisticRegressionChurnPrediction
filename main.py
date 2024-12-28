# 导入必要的库
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# 设置随机种子以保证结果可复现
np.random.seed(38)

# 读取数据
data_path = 'data_cleaned.csv'

data = pd.read_csv(data_path)

# 分离特征和目标
X = data.drop('Exited', axis=1)
y = data['Exited']

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割数据集为训练集(75%)和临时集(25%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=44, stratify=y)

# 将临时集进一步分割为验证集(12.5%)和测试集(12.5%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 自定义逻辑回归类
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=0.0001):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X_train, y_train, X_val, y_val):
        n_samples, n_features = X_train.shape
        # 初始化权重，包含偏置项
        self.w = np.random.normal(0, 0.01, n_features + 1)
        
        # 添加偏置项到训练集和验证集
        X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_val_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
        
        prev_loss = np.inf
        for iteration in range(self.max_iter):
            # 计算z和预测概率
            z = np.dot(X_train_bias, self.w)
            p = self.sigmoid(z)
            
            # 计算梯度
            grad = np.dot(X_train_bias.T, (p - y_train)) / n_samples
            
            # 生成扰动向量
            epsilon = np.random.normal(0, 0.000001, self.w.shape)
            
            # 更新权重
            self.w -= self.learning_rate * grad + epsilon
            
            # 计算验证集上的损失（交叉熵损失）
            z_val = np.dot(X_val_bias, self.w)
            p_val = self.sigmoid(z_val)
            loss = -np.mean(y_val * np.log(p_val + 1e-15) + (1 - y_val) * np.log(1 - p_val + 1e-15))
            
            # 检查收敛条件
            if iteration > 0 and abs(loss - prev_loss) < self.tol:
                print(f"自定义逻辑回归在第 {iteration} 次迭代时收敛。验证集损失: {loss:.4f}")
                break
            prev_loss = loss
        
        print("自定义逻辑回归训练完成。")
    
    def predict_proba(self, X):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        z = np.dot(X_bias, self.w)
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

# 初始化自定义逻辑回归模型
custom_lr = CustomLogisticRegression(learning_rate=0.1, max_iter=2000, tol=0.0000001)

# 训练自定义逻辑回归模型
custom_lr.fit(X_train, y_train, X_val, y_val)

# 使用自定义逻辑回归模型在验证集上进行简单参数调整（例如学习率）
# 这里简化处理，因为自定义模型参数有限
# 假设不进行额外的参数调整

# 初始化库函数逻辑回归模型
lib_lr = LogisticRegression(solver='liblinear', random_state=42)

# 初始化决策树模型
dt = DecisionTreeClassifier(random_state=42)

# 初始化神经网络模型
mlp = MLPClassifier(max_iter=5000, random_state=42)

# 初始化SVM模型
svm = SVC(probability=True, random_state=42)

# 初始化KNN模型
knn = KNeighborsClassifier()

# 初始化随机森林模型
rf = RandomForestClassifier(random_state=42)

# 定义参数网格
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],  
    'penalty': ['l1', 'l2'],  
    'max_iter': [100, 200, 500],  
}

param_grid_dt = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,)],
    'alpha': [0.0001, 0.001, 0.01]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20],    
    'min_samples_split': [2, 5, 10] 
}

# 使用网格搜索进行参数调优
grid_lr = GridSearchCV(lib_lr, param_grid_lr, cv=5, scoring='roc_auc')
grid_lr.fit(X_val, y_val)
best_lr = grid_lr.best_estimator_
print(f"最佳逻辑回归参数: {grid_lr.best_params_}")

grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='roc_auc')
grid_dt.fit(X_val, y_val)
best_dt = grid_dt.best_estimator_
print(f"最佳决策树参数: {grid_dt.best_params_}")

grid_mlp = GridSearchCV(mlp, param_grid_mlp, cv=5, scoring='roc_auc')
grid_mlp.fit(X_val, y_val)
best_mlp = grid_mlp.best_estimator_
print(f"最佳神经网络参数: {grid_mlp.best_params_}")

grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='roc_auc')
grid_svm.fit(X_val, y_val)
best_svm = grid_svm.best_estimator_
print(f"最佳SVM参数: {grid_svm.best_params_}")

grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='roc_auc')
grid_knn.fit(X_val, y_val)
best_knn = grid_knn.best_estimator_
print(f"最佳KNN参数: {grid_knn.best_params_}")

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc')
grid_rf.fit(X_val, y_val)
best_rf = grid_rf.best_estimator_
print(f"最佳随机森林参数: {grid_rf.best_params_}")



# 训练所有模型在训练集上
best_lr.fit(X_train, y_train)
best_dt.fit(X_train, y_train)
best_mlp.fit(X_train, y_train)
best_svm.fit(X_train, y_train)
best_knn.fit(X_train, y_train)
best_rf.fit(X_train, y_train)


# 自定义逻辑回归已经在训练过程中使用了训练集和验证集

# 定义一个函数来计算评估指标
def evaluate_model(name, model, X, y):
    if name == '自定义逻辑回归':
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
    else:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:,1]
        else:
            # 对于没有predict_proba的方法，如某些SVM
            y_proba = model.decision_function(X)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_proba)
    
    return accuracy, precision, recall, f1, auc

# 评估所有模型在测试集上的表现
results = {}

# 自定义逻辑回归
results['自定义逻辑回归'] = evaluate_model('自定义逻辑回归', custom_lr, X_test, y_test)

# 库函数逻辑回归
results['逻辑回归（库函数）'] = evaluate_model('逻辑回归（库函数）', best_lr, X_test, y_test)

# 决策树
results['决策树'] = evaluate_model('决策树', best_dt, X_test, y_test)

# 神经网络
results['神经网络'] = evaluate_model('神经网络', best_mlp, X_test, y_test)

# SVM
results['SVM'] = evaluate_model('SVM', best_svm, X_test, y_test)

# KNN
results['KNN'] = evaluate_model('KNN', best_knn, X_test, y_test)

#随机森林
results['随机森林'] = evaluate_model('随机森林', best_rf, X_test, y_test)


# 打印评估结果
print("\n模型评估结果（在测试集上）:")
print("{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}".format('模型', '准确率', '精确率', '召回率', 'F1分数', 'AUC'))
for model_name, metrics in results.items():
    accuracy, precision, recall, f1, auc = metrics
    print("{:<20} {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}".format(
        model_name, accuracy, precision, recall, f1, auc
    ))