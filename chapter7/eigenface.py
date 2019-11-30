"""
7-9 人脸识别与特征脸
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(data_home='d:/')
print(faces.keys())
print(faces.data.shape)
print(faces.images.shape)

random_indexes = np.random.permutation(len(faces.data))
X = faces.data[random_indexes]

example_faces = X[:36, :]
print(example_faces.shape)


def plot_faces(faces_):
    fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces_[i].reshape(62, 47), cmap='bone')
    plt.show()


# plot_faces(example_faces)


# 特征脸
pca = PCA(svd_solver='randomized')
pca.fit(X)
print(pca.components_.shape)
plot_faces(pca.components_[:36, :])

# 某个人至少有60个样本照片
faces2 = fetch_lfw_people(min_faces_per_person=60)
