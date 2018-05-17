"""
8-4 为什么要有训练数据集与测试数据集
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

np.random.seed(666)
x = np.random.uniform(-3., 3., size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.2)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_predict = lin_reg.predict(X_test)
print(mean_squared_error(y_test, y_predict))

poly2_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("std_scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])
poly2_reg.fit(X_train, y_train)
y2_predict = poly2_reg.predict(X_test)
print(mean_squared_error(y_test, y2_predict))

poly10_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=10)),
    ("std_scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])
poly10_reg.fit(X_train, y_train)
y10_predict = poly10_reg.predict(X_test)
print(mean_squared_error(y_test, y10_predict))

poly100_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=100)),
    ("std_scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])
poly100_reg.fit(X_train, y_train)
y100_predict = poly100_reg.predict(X_test)
print(mean_squared_error(y_test, y100_predict))
