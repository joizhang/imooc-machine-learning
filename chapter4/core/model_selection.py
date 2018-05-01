import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """
    将数据 X 和 y 按照 test_ratio 分割成 X_train, X_test, y_train, y_test
    :param X: 特征
    :param y: 标记
    :param test_ratio: 切分比例
    :param seed:
    :return:
    """
    assert X.shape[0] == y.shape[0], "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, "test_ratio must be valid"
    if seed:
        np.random.seed(seed)
    # train test split
    # 0 - len(X) 索引的随机排列
    shuffle_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    # 测试数据集索引
    test_indexes = shuffle_indexes[:test_size]
    # 训练数据集索引
    train_indexes = shuffle_indexes[test_size:]

    # 测试数据
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    # 训练数据
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
