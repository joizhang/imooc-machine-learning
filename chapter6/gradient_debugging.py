"""
6-8  如何确定梯度计算的准确性？调试梯度下降法
"""
import numpy as np

np.random.seed(666)
X = np.random.random(size=(1000, 10))
true_theta = np.arange(1, 12, dtype=float)
X_b = np.hstack([np.ones((len(X), 1)), X])
y = X_b.dot(true_theta) + np.random.normal(size=1000)


def J(theta, X_b_arg, y_arg):
    try:
        return np.sum((y_arg - X_b_arg.dot(theta)) ** 2) / len(X_b_arg)
    except:
        return float('inf')


def dJ_math(theta, X_b_arg, y_arg):
    return X_b_arg.T.dot(X_b_arg.dot(theta) - y_arg) * 2. / len(y_arg)


def dJ_debug(theta, X_b_arg, y_arg, epsilon=0.01):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2[i] -= epsilon
        res[i] = (J(theta_1, X_b_arg, y_arg) - J(theta_2, X_b_arg, y_arg)) / (2 * epsilon)
    return res


def gradient_descent(dJ, X_b_arg, y_arg, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    theta = initial_theta
    cur_iter = 0
    while cur_iter < n_iters:
        gradient = dJ(theta, X_b_arg, y_arg)
        last_theta = theta
        theta = theta - eta * gradient
        if abs(J(theta, X_b_arg, y_arg) - J(last_theta, X_b_arg, y_arg)) < epsilon:
            break
        cur_iter += 1
    return theta


initial_theta1 = np.zeros(X_b.shape[1])
eta1 = 0.01
theta1 = gradient_descent(dJ_math, X_b, y, initial_theta1, eta1)
print(theta1)
theta2 = gradient_descent(dJ_debug, X_b, y, initial_theta1, eta1)
print(theta2)
