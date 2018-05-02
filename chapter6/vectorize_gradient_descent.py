"""
6-5 梯度下降法的向量化和数据标准化
"""
import numpy as np
from sklearn import datasets
from chapter4.core.model_selection import train_test_split
from chapter5.core.linear_regression import LinearRegression
from sklearn.preprocessing import StandardScaler

boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

lin_reg1 = LinearRegression()
lin_reg1.fit_normal(X, y)
print(lin_reg1.score(X_test, y_test))

standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
lin_reg2 = LinearRegression()
lin_reg2.fit_gd(X_train_standard, y_train)
X_test_standard = standardScaler.transform(X_test)
print(lin_reg2.score(X_test_standard, y_test))
