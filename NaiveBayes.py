"""
数据集：iris
五折五次交叉验证
每折正确率最高为100%
平均正确率为95.33%
"""
from sklearn import datasets
from sklearn.model_selection import KFold
import numpy as np


def gaussian(x, std, mean):     # 对向量进行高斯运算并相乘
    return np.prod(1 / (np.sqrt(2 * np.pi) * std) * np.exp(-1 * (x - mean)**2 / (2 * std**2)))


class NaiveBayes:
    def __init__(self, x, y):
        self.std = np.zeros((len(set(y)), x.shape[1]))  # 行表示y值，列表示样本的特征，(i,j)表示在y=i条件下的j分量的标准差
        self.mean = np.zeros_like(self.std)             # 行表示y值，列表示样本的特征，(i,j)表示在y=i条件下的j分量的均值
        self.P_y = np.zeros(3)
        for i in range(3):
            self.P_y[i] = np.sum(y_train == i)/y_train.size
        self.calculate(x, y)

    def calculate(self, x, y):
        labels = np.array(list(set(y)))
        for i in range(labels.size):
            self.std[i] = np.std(x[y == labels[i]], axis=0)
            self.mean[i] = np.mean(x[y == labels[i]], axis=0)

    def prob(self, x_test):
        probability = np.zeros((x_test.shape[0], 3))    # 行表示x_test各样本值，列表示y取值，(i,j)表示第i个样本在y=j下各分量概率乘积
        for i in range(x_test.shape[0]):
            for j in range(3):
                probability[i][j] = gaussian(x_test[i], self.std[j], self.mean[j])  # gaussian()返回向量各特征的概率的连乘
            probability[i] *= self.P_y
        return probability

    def predict(self, x_test, y_test):
        accuracy = self.estimate(x_test, y_test)
        print("正确率为", accuracy * 100, "%")
        return accuracy

    def estimate(self, x_test, y_test):
        prob = self.prob(x_test)
        y_pred = np.argmax(prob, axis=1)
        return np.sum(y_pred == y_test)/y_test.size


data, target = datasets.load_iris(return_X_y=True)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
accuracy = np.zeros(5)
it = 0
for train_index, test_index in kf.split(data):
    x_train = data[train_index]
    y_train = target[train_index]
    x_test = data[test_index]
    y_test = target[test_index]
    model = NaiveBayes(x_train, y_train)
    accuracy[it] = model.predict(x_test, y_test)
    it = it + 1
print("平均正确率", np.sum(accuracy)/5 * 100, "%")
