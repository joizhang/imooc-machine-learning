"""
7-4 求数据的前n个主成分
"""
import matplotlib.pyplot as plt
import numpy as np


# demean
def demean(X):
    return X - np.mean(X, axis=0)


# 梯度上升法
def f(w, X):
    return np.sum((X.dot(w) ** 2)) / len(X)


def df(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)


# 单位向量
def direction(w):
    # w 向量除以w的模
    return w / np.linalg.norm(w)


def first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        # 注意1：每次求一个单位方向
        w = direction(w)
        if (abs(f(w, X)) - f(last_w, X)) < epsilon:
            break
        cur_iter += 1
    return w


if __name__ == "__main__":
    X1 = np.empty((100, 2))
    X1[:, 0] = np.random.uniform(0., 100., size=100)
    X1[:, 1] = 0.75 * X1[:, 0] + 3. + np.random.normal(0, 10., size=100)

    X1 = demean(X1)
    plt.scatter(X1[:, 0], X1[:, 1])
    plt.show()
    initial_w1 = np.random.random(X1.shape[1])
    eta1 = 0.001
    w1 = first_component(X1, initial_w1, eta1)
    print(w1)

    # X2 = np.empty(X1.shape)
    # for i in range(len(X1)):
    #     X2[i] = X1[i] - X1[i].dot(w1) * w1
    # 向量化的求法
    X2 = X1 - X1.dot(w1).reshape(-1, 1) * w1
    plt.scatter(X2[:, 0], X2[:, 1])
    plt.show()
    initial_w2 = np.random.random(X2.shape[1])
    w2 = first_component(X2, initial_w2, eta1)
    print(w2)
    print(w1.dot(w2))
