"""
9-5 决策边界
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from chapter9.core.logistic_regression import LogisticRegression
from chapter9.core.logistic_regression import plot_decision_boundary


def x2(x1):
    return (-log_reg.coef_[0] * x1 - log_reg.interception_) / log_reg.coef_[1]


iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y < 2, :2]
y = y[y < 2]
print(X.shape)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))
print(log_reg.coef_)
print(log_reg.interception_)

x1_plot = np.linspace(4, 8, 1000)
x2_plot = x2(x1_plot)
plt.plot(x1_plot, x2_plot)
plt.show()

plot_decision_boundary(log_reg, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

# kNN的决策边界
