import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def Lw(x, y, w):
    x_y0 = x[y == 0, :]
    x_y1 = x[y == 1, :]
    return np.sum(x_y0 @ w[:, 0]) + np.sum(x_y1 @ w[:, 1]) - np.sum(np.log(1 + np.exp(x @ w[:, 0]) + np.exp(x @ w[:, 1])))


def grad_Lw(x, y, w):
    x_y0 = x[y == 0, :]
    x_y1 = x[y == 1, :]
    grad0 = np.sum(x_y0, axis=0) - np.sum(np.transpose(x) * np.exp(x @ w[:, 0]) / (1 + np.exp(x @ w[:, 0]) + np.exp(x @ w[:, 0])), axis=1)
    grad1 = np.sum(x_y1, axis=0) - np.sum(np.transpose(x) * np.exp(x @ w[:, 1]) / (1 + np.exp(x @ w[:, 1]) + np.exp(x @ w[:, 1])), axis=1)
    return grad0, grad1


def predict(x, w):
    size = x.shape[0]
    P = np.zeros((3, size))
    P[0, :] = np.exp(x @ w[:, 0]) / (1 + np.exp(x @ w[:, 0]) + np.exp(x @ w[:, 1]))
    P[1, :] = np.exp(x @ w[:, 1]) / (1 + np.exp(x @ w[:, 0]) + np.exp(x @ w[:, 1]))
    P[2, :] = 1 / (1 + np.exp(x @ w[:, 0]) + np.exp(x @ w[:, 1]))
    y = np.argmax(P, axis=0)
    return y


data, target = datasets.load_iris(return_X_y=True)
features_amount = data.shape[1]
labels_amount = len(set(target))
iterations = 1000000
eta = 0.000001
data_ = np.hstack((data, np.ones((data.shape[0], 1))))
x_train, x_test, y_train, y_test = train_test_split(data_, target, test_size=0.2)
w = np.zeros((features_amount+1, labels_amount-1))    # 每列代表每个标签下的label，行数代表特征数+1（将bias合并入w）
for it in range(iterations):
    L = Lw(x_train, y_train, w)
    # print(L)
    if L > -76.5:
        break
    grad0, grad1 = grad_Lw(x_train, y_train, w)
    w[:, 0] += eta * grad0
    w[:, 1] += eta * grad1
y_predict = np.zeros_like(y_test)
y_predict = predict(x_test, w)
print('正确率：', np.sum(y_predict == y_test) / np.size(y_test))