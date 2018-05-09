import numpy as np


class PCA:
    def __init__(self, n_components):
        """
        初始化PCA
        :param n_components:
        """
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X_pca, eta_pca=0.1, n_iters_pca=1e4):
        """
        获取数据集X的前n个主成分
        :param X_pca:
        :param eta_pca:
        :param n_iters_pca:
        :return:
        """
        assert self.n_components <= X_pca.shape[1], "n_components must not be greater than the feature number of X"

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

        X_pca = demean(X_pca)
        self.components_ = np.empty(shape=(self.n_components, X_pca.shape[1]))
        for i in range(self.n_components):
            initial_w_pca = np.random.random(X_pca.shape[1])
            w_pca = first_component(X_pca, initial_w_pca, eta_pca, n_iters_pca)
            self.components_[i, :] = w_pca
            X_pca = X_pca - X_pca.dot(w_pca).reshape(-1, 1) * w_pca
        return self

    def transform(self, X):
        """
        将给定的X，映射到各个主成分分量中
        :param X:
        :return:
        """
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """
        将给定的X，反向映射回原来的特征空间
        :param X:
        :return:
        """
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
