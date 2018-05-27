"""
12-3 使用信息熵寻找最优划分
"""
from sklearn import datasets

from chapter12.core.entropy import try_split
from chapter12.core.entropy import split
from chapter12.core.entropy import entropy

iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target
best_entropy, best_d, best_v = try_split(X, y)
print("best_entropy = ", best_entropy)
print("best_d = ", best_d)
print("best_v = ", best_v)

X1_l, X1_r, y1_l, y1_r = split(X, y, best_d, best_v)

print(entropy(y1_l))

print(entropy(y1_r))
