"""
6-6 随机梯度下降法
"""
import numpy as np

m = 100000

x = np.random.normal(size=m)
X = x.reshape(-1, 1)
y = 4. * x + 3. + np.random.normal(0, 3, size=m)


def J(theta, X_b_arg, y_arg):
    try:
        return np.sum((y_arg - X_b_arg.dot(theta)) ** 2) / len(y_arg)
    except:
        return float('inf')


def dJ(theta, X_b_arg, y_arg):
    return X_b_arg.T.dot(X_b_arg.dot(theta) - y_arg) * 2. / len(y_arg)


def gradient_descent(X_b_arg, y_arg, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
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


# 随机梯度下降
def dJ_sgd(theta, X_b_i, y_i):
    return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.


def sgd(X_b_arg, y_arg, initial_theta, n_iters):
    t0 = 5
    t1 = 50

    def learning_rate(t):
        return t0 / (t + t1)

    theta = initial_theta
    for cur_iter in range(n_iters):
        rand_i = np.random.randint(len(X_b_arg))
        gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])
        theta = theta - learning_rate(cur_iter) * gradient
    return theta


X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta1 = np.zeros(X_b.shape[1])
eta1 = 0.01
theta1 = gradient_descent(X_b, y, initial_theta1, eta1)
print(theta1)

initial_theta2 = np.zeros(X_b.shape[1])
theta2 = sgd(X_b, y, initial_theta2, n_iters=len(X_b) // 3)
print(theta2)
