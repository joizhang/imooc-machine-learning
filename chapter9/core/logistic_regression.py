import numpy as np
from chapter4.core.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    def __init__(self):
        # 系数
        self.coef_ = None
        # 截距
        self.interception_ = None
        self._theta = None

    @staticmethod
    def _sigmoid(t):
        return 1. / (1. + np.exp((-t)))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """
        根据训练数据集 X_train, y_train，使用梯度下降训练 Logistic Regression 模型
        :param X_train:
        :param y_train:
        :param eta:
        :param n_iters:
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b_arg, y):
            y_hat = self._sigmoid(X_b_arg.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(X_b_arg)
            except:
                return float('inf')

        def dJ(theta, X_b_arg, y):
            return X_b_arg.T.dot(self._sigmoid(X_b_arg.dot(theta)) - y) / len(X_b_arg)

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

    def predict_probability(self, X_predict):
        """
        给定带预测数据集 X_predict，返回表示 X_predict 的结果向量
        :param X_predict:
        :return:
        """
        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """
        给定带预测数据集 X_predict，返回表示 X_predict 的结果向量
        :param X_predict:
        :return:
        """
        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        probability = self.predict_probability(X_predict)
        return np.array(probability >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """
        根据测试数据集 X_test, y_test 确定当前模型的准确度
        :param X_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


def polynomial_logistic_regression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])
