"""
7-8 使用PCA对数据进行降噪
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

X1 = np.empty((100, 2))
X1[:, 0] = np.random.uniform(0., 100., size=100)
X1[:, 1] = 0.75 * X1[:, 0] + 3. + np.random.normal(0, 10., size=100)

pca = PCA(n_components=1)
pca.fit(X1)

X_reduction = pca.transform(X1)
X_restore = pca.inverse_transform(X_reduction)
