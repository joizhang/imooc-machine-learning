"""
7-3 求数据的主成分PCA
"""
import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:, 0] = np.random.uniform(0., 100., size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

plt.scatter(X[:, 0], X[:, 1])
plt.show()


# demean
def demean(X_arg):
    return X_arg - np.mean(X_arg, axis=0)


X_demean = demean(X)
plt.scatter(X_demean[:, 0], X_demean[:, 1])
plt.show()


# 梯度上升法
def f(w, X_arg):
    return np.sum((X_arg.dot(w) ** 2)) / len(X_arg)


def df_math(w, X_arg):
    return X_arg.T.dot(X_arg.dot(w)) * 2. / len(X_arg)


def df_debug(w, X_arg, epsilon=0.0001):
    res = np.empty(len(w))
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X_arg) - f(w_2, X_arg)) / (2 * epsilon)
    return res


# 单位向量
def direction(w):
    # w 向量除以w的模
    return w / np.linalg.norm(w)


def gradient_ascent(df, X_arg, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    global w
    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X_arg)
        last_w = w
        w = w + eta * gradient
        # 注意1：每次求一个单位方向
        w = direction(w)
        if (abs(f(w, X)) - f(last_w, X)) < epsilon:
            break
        cur_iter += 1
    return w


# 注意2：不能从0向量开始
initial_w1 = np.random.random(X.shape[1])
eta1 = 0.001
# 注意3：不能使用StandardScaler标准化数据
print(gradient_ascent(df_debug, X_demean, initial_w1, eta1))
w = gradient_ascent(df_math, X_demean, initial_w1, eta1)
print(w)

plt.scatter(X_demean[:, 0], X_demean[:, 1])
plt.plot([0, w[0] * 30], [0, w[1] * 30], color='r')
plt.show()
