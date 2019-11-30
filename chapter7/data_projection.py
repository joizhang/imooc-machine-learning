"""
7-5 高维数据映射为低维数据
"""
import matplotlib.pyplot as plt
import numpy as np

from chapter7.core.pca import PCA

X1 = np.empty((100, 2))
X1[:, 0] = np.random.uniform(0., 100., size=100)
X1[:, 1] = 0.75 * X1[:, 0] + 3. + np.random.normal(0, 10., size=100)

pca = PCA(n_components=2)
pca.fit(X1)
print(pca.components_)

pca = PCA(n_components=1)
pca.fit(X1)
print(pca.components_)

X_reduction = pca.transform(X1)
print(X_reduction.shape)

X_restore = pca.inverse_transform(X_reduction)
print(X_restore.shape)

plt.scatter(X1[:, 0], X1[:, 1], color='b', alpha=0.5)
plt.scatter(X_restore[:, 0], X_restore[:, 1], color='r', alpha=0.5)
plt.show()
