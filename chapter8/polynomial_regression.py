"""
8-1 什么是多项式回归
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict = lin_reg.predict(X)

plt.scatter(x, y)
plt.plot(x, y_predict, color='r')
plt.show()

# 线性回归不理想，解决方案，添加一个特征
X2 = np.hstack([X, X ** 2])
print(X2.shape)
lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()
# 分别是[X, X ** 2]前面的系数
print(lin_reg2.coef_)
