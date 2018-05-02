"""
6-4 实现线性回归中的梯度下降
"""
import numpy as np
import matplotlib.pyplot as plt
from chapter5.core.linear_regression import LinearRegression

np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3. + 4. + np.random.normal(size=100)
X = x.reshape(-1, 1)
plt.scatter(x, y)
plt.show()

# 使用梯度下降法训练
lin_reg = LinearRegression()
lin_reg.fit_gd(X, y)
print(lin_reg.coef_, lin_reg.interception_)
