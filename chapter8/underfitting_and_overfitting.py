"""
 8-3 过拟合与欠拟合
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(666)
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

# 欠拟合
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict = lin_reg.predict(X)
plt.scatter(x, y)
plt.plot(x, y_predict, color='r')
plt.show()

# 过拟合
poly10_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=100)),
    ("std_scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])
poly10_reg.fit(X, y)
y10_predict = poly10_reg.predict(X)
plt.scatter(x, y)
plt.plot(np.sort(x), y10_predict[np.argsort(x)], color='r')
plt.show()
