"""
5-3 简单线性回归的实现
"""
import numpy as np
import matplotlib.pyplot as plt
from chapter5.core.simple_linear_regression import SimpleLinearRegression1
from chapter5.core.simple_linear_regression import SimpleLinearRegression2

x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

# ########## 使用我们自己的 SimpleLinearRegression ##########
reg1 = SimpleLinearRegression1()
reg1.fit(x, y)
x_predict = 6
print(reg1.predict(np.array([x_predict])))
y_hat1 = reg1.predict(x)
plt.scatter(x, y)
plt.plot(x, y_hat1, color='r')
plt.axis([0, 6, 0, 6])
plt.show()

# ########## 向量化实现的 SimpleLinearRegression ##########
reg2 = SimpleLinearRegression2()
reg2.fit(x, y)
print(reg2.predict(np.array([x_predict])))

m = 1000000
big_x = np.random.random(size=m)
big_y = big_x * 2.0 + 3.0 + np.random.normal(size=m)
