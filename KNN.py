"""
数据集：breast_cancer
正确率最高为98.26%，此时k=6
"""

from sklearn import datasets   # 导入scikit-learn库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

breast_cancer_data = datasets.load_breast_cancer()
features = breast_cancer_data.data   # 特征
targets = breast_cancer_data.target  # 类别
mm = MinMaxScaler()
features = mm.fit_transform(features)   # 对特征进行归一化，统一量纲
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)
y_predict = np.zeros_like(y_test)       # 存放预测数据
for k in range(1, 20):
    for i in range(y_test.size):
        dist = np.linalg.norm(x_train - x_test[i], axis=1)  # 计算训练集到预测数据的欧氏距离
        order = np.argsort(dist)                            # 为距离排序
        count_1 = np.sum(y_train[order[:k]] == 1)           # 取前k个元素，统计标签为1的个数
        count_0 = k - count_1                               # 再统计标签为0的个数
        if count_0 >= count_1:
            y_predict[i] = 0
        else:
            y_predict[i] = 1
    print("k为", k, "时正确率", np.sum(y_predict == y_test)/y_test.size * 100, "%")
