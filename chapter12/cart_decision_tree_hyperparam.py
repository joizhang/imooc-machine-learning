"""
12-5 CART与决策树中的超参数
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from chapter9.core.decision_boundary import plot_decision_boundary

X, y = datasets.make_moons(noise=0.25, random_state=666)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X, y)
plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

# max_depth
# min_samples_split：对于一个节点至少需要多少个样本才继续拆分
# min_samples_leaf：对于一个叶子节点至少需要多少个样本
# max_leaf_nodes：最大叶子节点数
