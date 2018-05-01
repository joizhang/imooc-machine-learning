import numpy as np
from chapter4.core.metrics import r2_score


class LinearRegression:
    def __init__(self):
        # 系数
        self.coef_ = None
        # 截距
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """
        根据训练数据集 X_train, y_train 训练 Linear Regression 模型
        :param X_train:
        :param y_train:
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """
        根据训练数据集 X_train, y_train，使用梯度下降训练Linear Regression 模型
        :param X_train:
        :param y_train:
        :param eta:
        :param n_iters:
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b_arg, y):
            try:
                return np.sum((y - X_b_arg.dot(theta)) ** 2) / len(X_b_arg)
            except:
                return float('inf')

        def dJ(theta, X_b_arg, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b_arg.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b_arg.dot(theta) - y).dot(X_b_arg[:, i])
            return res * 2 / len(X_b_arg)

        def gradient_descent(X_b_arg, y, initial_theta_arg, eta_arg, n_iters_arg=1e4, epsilon=1e-8):
            theta = initial_theta_arg
            i_iter = 0
            while i_iter < n_iters_arg:
                gradient = dJ(theta, X_b_arg, y)
                last_theta = theta
                theta = theta - eta_arg * gradient
                if abs(J(theta, X_b_arg, y) - J(last_theta, X_b_arg, y)) < epsilon:
                    break
                i_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        """
        给定带预测数据集 X_predict，返回表示 X_predict 的结果向量
        :param X_predict:
        :return:
        """
        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """
        根据测试数据集 X_test, y_test 确定当前模型的准确度
        :param X_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
