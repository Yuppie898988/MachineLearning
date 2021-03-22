"""
数据集：breast_cancer
正确率最高为95.61%
"""

from sklearn import datasets   # 导入scikit-learn库
from sklearn.model_selection import train_test_split
import numpy as np
breast_cancer_data = datasets.load_breast_cancer()
features = breast_cancer_data.data   # 特征
targets = breast_cancer_data.target  # 类别
feature_nums = features.shape[1]
targets[targets == 0] = -1
f_train, f_test, t_train, t_test = train_test_split(features, targets, test_size=0.2, random_state=0)
w = np.zeros(feature_nums + 1)      # 设定参数，偏移量b作为31号元素，对应输入量恒为1
Loss = 0.0
Learning_rate = 0.01
for i in range(100000):
    it = np.random.randint(0, t_train.size)     # 选取随机行
    x = np.hstack((f_train[it, :], 1))          # 增加值为1的输入量
    y_predict = x @ w
    if y_predict * t_train[it] > 0:             # 标签匹配则重新选取输入样本
        continue
    Loss = -1 * t_train[it] * y_predict
    # print('Loss:', Loss)
    w_grad = -1 * t_train[it] * x
    w -= Learning_rate * w_grad
test_conclusion = np.zeros_like(t_test)         # 用于存储预测值
for i in range(t_test.size):
    test_conclusion[i] = np.sign(np.hstack((f_test[i, :], 1)) @ w)
print(t_test.size, '组test中预测数据与实际数据不相符的有', np.sum(t_test != test_conclusion), '组')
print('正确率为：', (1 - np.sum(t_test != test_conclusion)/t_test.size)*100, '%')
print('w权重如下：')
print(w)        # 31个，其中最后一个为偏移量b
