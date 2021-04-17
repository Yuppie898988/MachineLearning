"""
数据集：iris
采用OvR策略将三分类问题转化为二分类问题
正确率为96.67%
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def Lw(x, y, w):
    """对数似然函数计算

    :param x: 行对应样本，列对应样本的特征
    :param y: 与样本集合x对应的标签值
    :param w: 权重，每一列表示对应标签下的分类器
    :return: 各分类器下的对数似然函数值
    """

    labels_amount = w.shape[1]              # 统计标签数
    L = np.zeros(labels_amount)
    for label in range(labels_amount):      # 遍历各个标签，以该标签作为正例，其余标签作为负例
        temp_y = np.zeros_like(y)
        temp_y[y == label] = 1
        weight = w[:, label]                # 读取对应标签下的分类器权重
        L[label] = np.sum(temp_y * (x @ weight) - np.log(1 + np.exp(x @ weight)), axis=0)
    return L



def grad_Lw(x, y, w):
    """计算似然函数梯度

    :param x: 行对应样本，列对应样本的特征
    :param y: 与样本集合x对应的标签值
    :param w: 权重，每一列表示对应标签下的分类器
    :return: 各分类器下的对数似然函数梯度值，每一列表示对应标签下的分类器
    """
    labels_amount = w.shape[1]
    grad_L = np.zeros_like(w)
    for label in range(labels_amount):
        temp_y = np.zeros_like(y)
        temp_y[y == label] = 1
        weight = w[:, label]
        grad_L[:, label] = np.sum(np.transpose(x) * temp_y - x.T * np.exp(x @ weight) / (1 + np.exp(x @ weight)), axis=1)
    return grad_L


def predict(x, w):
    """对新样本集合进行预测

    :param x: 行对应样本，列对应样本的特征
    :param w: 权重，每一列表示对应标签下的分类器
    :return: 预测结果
    """
    size = x.shape[0]                       # 新样本量
    labels_amount = w.shape[1]
    P = np.zeros((size, labels_amount))     # 对每个样本遍历所有分类器，选出概率最大的标签为预测值
    for label in range(labels_amount):
        weight = w[:, label]
        P[:, label] = np.exp(x @ weight) / (1 + np.exp(x @ weight))
    y = np.argmax(P, axis=1)                # P的每一行表示为各分类器计算出的概率，选出最大概率的对应索引
    return y


data, target = datasets.load_iris(return_X_y=True)
features_amount = data.shape[1]
labels_amount = len(set(target))
iterations = 1000000
eta = 0.00001
epsilon = 1e-6
data_ = np.hstack((data, np.ones((data.shape[0], 1))))      # 将全1向量合并进入原始数据（作为最后一列），便于将bias合并入w
x_train, x_test, y_train, y_test = train_test_split(data_, target, test_size=0.2)
w = np.zeros((features_amount+1, labels_amount))    # 每列代表以对应label作为正例的分类器，行数代表特征数+1（将bias合并入w）
for it in range(iterations):
    L = Lw(x_train, y_train, w)
    print(L)
    grad = grad_Lw(x_train, y_train, w)
    if np.max(eta * grad) < epsilon:                      # 当梯度变化小于epsilon时可认为训练完成
        break
    for label in range(labels_amount):
        w[:, label] += eta * grad[:, label]         # 因为此时要使对数似然函数最大，所以是向梯度上升的方向移动
y_predict = np.zeros_like(y_test)
y_predict = predict(x_test, w)
print('正确率：', np.sum(y_predict == y_test) / np.size(y_test) * 100, '%')
