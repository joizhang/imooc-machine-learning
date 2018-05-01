"""
5-8 实现多元线性回归
"""
from sklearn import datasets
from chapter4.core.model_selection import train_test_split
from chapter5.core.linear_regression import LinearRegression

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

reg = LinearRegression()
reg.fit_normal(X_train, y_train)
print(reg.score(X_test, y_test))
