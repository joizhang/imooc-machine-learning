"""
8-8 模型泛化与岭回归
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from chapter8.core.polynomial_regression import polynomial_regression
from chapter8.core.ridge_regression import plot_model
from chapter8.core.ridge_regression import ridge_regression

np.random.seed(42)
x = np.random.uniform(-3., 3., size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)

np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)

poly20_reg = polynomial_regression(degree=20)
poly20_reg.fit(X_train, y_train)
y20_predict = poly20_reg.predict(X_test)
print(mean_squared_error(y_test, y20_predict))

plot_model(x, y, poly20_reg)

# 岭回归
ridge1_reg = ridge_regression(20, 0.0001)
ridge1_reg.fit(X_train, y_train)
y1_predict = ridge1_reg.predict(X_test)
print(mean_squared_error(y_test, y1_predict))

plot_model(x, y, ridge1_reg)
