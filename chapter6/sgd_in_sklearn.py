"""
6-7  scikit-learn中的随机梯度下降法
"""
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

from chapter4.core.model_selection import train_test_split
from chapter5.core.linear_regression import LinearRegression

m = 100000

x = np.random.normal(size=m)
X = x.reshape(-1, 1)
y = 4. * x + 3. + np.random.normal(0, 3, size=m)

lin_reg = LinearRegression()
lin_reg.fit_sgd(X, y, n_iters=2)
print(lin_reg.coef_, lin_reg.interception_)

# 真实使用我们自己的 SGD
boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

lin_reg = LinearRegression()
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)
lin_reg.fit_sgd(X_train_standard, y_train, n_iters=5)
print(lin_reg.score(X_test_standard, y_test))

# sklearn 中的 SGD
sgd_reg = SGDRegressor(max_iter=5)
sgd_reg.fit(X_train_standard, y_train)
print(sgd_reg.score(X_test_standard, y_test))
