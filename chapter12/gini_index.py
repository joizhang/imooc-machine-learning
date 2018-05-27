"""
12-4 基尼系数
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from chapter12.core.gini import gini
from chapter12.core.gini import split
from chapter12.core.gini import try_split
from chapter9.core.decision_boundary import plot_decision_boundary

iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target

dt_clf = DecisionTreeClassifier(max_depth=2, criterion="gini")
dt_clf.fit(X, y)

plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.show()

best_g, best_d, best_v = try_split(X, y)
print("best_g = ", best_g)
print("best_d = ", best_d)
print("best_v = ", best_v)
X1_l, X1_r, y1_l, y1_r = split(X, y, best_d, best_v)

print(gini(y1_l))

print(gini(y1_r))
