"""
9-6 在逻辑回归中使用多项式特征
"""
import matplotlib.pyplot as plt
import numpy as np

from chapter9.core.decision_boundary import plot_decision_boundary
from chapter9.core.logistic_regression import LogisticRegression
from chapter9.core.logistic_regression import polynomial_logistic_regression

np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array(X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5, dtype='int')

log_reg = LogisticRegression()
log_reg.fit(X, y)
print(log_reg.score(X, y))

plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

# 多项式特征
poly_log_reg = polynomial_logistic_regression(degree=2)
poly_log_reg.fit(X, y)
print(poly_log_reg.score(X, y))
plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
