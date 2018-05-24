"""
9-4 实现逻辑回归算法
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from chapter9.core.logistic_regression import LogisticRegression

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y < 2, :2]
y = y[y < 2]
print(X.shape)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))
print(log_reg.predict_probability(X_test))
print(y_test)
print(log_reg.predict(X_test))


