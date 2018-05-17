"""
8-5 学习曲线
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from chapter8.core.learning_curve import plot_learning_curve
from chapter8.core.polynomial_regression import polynomial_regression

np.random.seed(666)
x = np.random.uniform(-3., 3., size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

# 线性回归学习曲线示例
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.25)
print(X_train.shape)
plot_learning_curve(LinearRegression(), X_train, X_test, y_train, y_test)

# 多项式回归学习曲线示例
poly2_reg = polynomial_regression(degree=2)
plot_learning_curve(poly2_reg, X_train, X_test, y_train, y_test)

# 更大的degree
poly20_reg = polynomial_regression(degree=20)
plot_learning_curve(poly20_reg, X_train, X_test, y_train, y_test)
