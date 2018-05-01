"""
3-12 数据加载与简单数据探索
"""
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()
print(iris.keys())
print(iris.data.shape)
print(iris.target.shape)

# 前两个维度
X = iris.data[:, :2]
print(X.shape)
y = iris.target
# fancy indexing
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="o")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="+")
plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", marker="x")
plt.show()

# 后两个维度
X = iris.data[:, 2:]
print(X.shape)
y = iris.target
# fancy indexing
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="o")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="+")
plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", marker="x")
plt.show()
