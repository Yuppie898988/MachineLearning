"""
数据集：play_tennis
五折五次交叉验证
每折正确率最高为100%
平均正确率为76.66%
由于数据量过小，划分出的决策树会出现未包含所有特征的情况，故对未知特征采取多数表决法确定标签
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class DecisionTree:
    def __init__(self):
        self.feature = ""           # 表示待划分特征
        self.children = {}          # 表示子树，key为具体特征值，value为对应子树
        self.amount = 0             # 表示当前样本数量，用于多数投票表决确定未知特征的标签


def Entropy(data):                  # 对列向量计算经验熵
    items = list(set(data))
    p = np.zeros(len(items))
    size = data.size
    for i in range(len(items)):
        p[i] = np.sum(data == items[i])/size
    return np.sum(-1 * np.log2(p) * p)


def Conditional_E(data):
    H_Y_X = np.zeros(data.shape[1] - 1)
    i = 0
    for col in data.columns[:-1]:   # 最后一列不取
        column = data.loc[:, col]   # 遍历每一列，当前列存为column
        items = list(set(column))
        e = np.zeros(len(items))
        p = np.zeros_like(e)
        for it in range(len(items)):
            e[it] = Entropy(data[column == items[it]].iloc[:, -1])  # 根据当前列是否等于items特征来选取data，再取最后一列计算经验熵
            p[it] = data[column == items[it]].shape[0]/data.shape[0]
        H_Y_X[i] = e @ p
        i += 1
    return H_Y_X


def CreatTree(data, epsilon, tree):
    if len(data.columns) == 1:                      # 如果数据只有一列（标签列），则说明所有特征均被划分
        target = data.iloc[:, -1].value_counts()    # 统计结果，取最多数为结果
        tree.feature = target.index[0]
        return
    H = Entropy(data.iloc[:, -1])
    H_Y_X = Conditional_E(data)
    Gain = H - H_Y_X
    index = np.argmax(Gain)
    if Gain[index] < epsilon:
        target = data.iloc[:, -1].value_counts()    # 统计结果，取最多数为结果
        tree.feature = target.index[0]
        tree.amount = data.shape[0]
        return
    feature = data.columns[index]       # 确定划分特征
    tree.feature = feature
    tree.amount = data.shape[0]
    tree.children = {item: DecisionTree() for item in list(set(data.iloc[:, index]))}
    for key, value in tree.children.items():
        CreatTree(data[data.loc[:, feature] == key].drop(feature, axis=1), epsilon, value)


def Predict(x, tree):
    try:
        next_tree = tree.children[x[tree.feature]]
    except KeyError:                # 此时x的待划分特征不在决策树中
        vote = {'Yes': 0, 'No': 0}  # 统计投票结果
        Vote(tree, vote)
        result = 'Yes' if vote['Yes'] >= vote['No'] else 'No'
        return result
    if not next_tree.children:
        return next_tree.feature
    result = Predict(x, next_tree)
    return result


def Vote(tree, vote_result):                    # 当此时决策树中划分特征中出现未知值，则遍历该所有子树，统计样本数量，投票表决含未知值的样本标签
    for node in tree.children.values():
        if not node.children:
            vote_result[node.feature] += node.amount
            continue
        Vote(node, vote_result)


df = pd.read_csv("play_tennis.csv", index_col="day")
kf = KFold(n_splits=5, shuffle=True, random_state=1)
accuracy = np.zeros(5)
it = 0
for train_index, test_index in kf.split(df):
    train = df.iloc[train_index, :]
    root = DecisionTree()
    CreatTree(train, 0.1, root)
    for test in test_index:
        if Predict(df.iloc[test, :], root) == df.iloc[test, -1]:
            accuracy[it] += 1
    accuracy[it] /= len(test_index)
    print('正确率为', accuracy[it] * 100, '%')
    it += 1
print('平均正确率为', np.sum(accuracy) / 5 * 100, '%')
