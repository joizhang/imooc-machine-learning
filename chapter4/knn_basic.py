"""
4-1 k近邻算法基础
4-2 scikit-learn中的机器学习算法封装
"""
import matplotlib.pyplot as plt
import numpy as np

from chapter4.core import knn
from sklearn.neighbors import KNeighborsClassifier

raw_data_X = [[3.39, 2.33],
              [3.11, 1.78],
              [1.34, 3.37],
              [3.58, 4.68],
              [2.28, 2.87],
              [7.42, 4.70],
              [5.74, 3.53],
              [9.17, 2.51],
              [7.79, 3.42],
              [7.94, 0.79]]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='g')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='r')

x = np.array([8.09, 3.37])
x_predict = x.reshape(1, -1)
plt.scatter(x[0], x[1], color='b')

# ########## 自实现 kNN ##########
knn_clf = knn.KNNClassifier(k=6)
knn_clf.fit(X_train, y_train)
print(knn_clf.predict(x_predict))

# ########## 使用 scikit-learn 中的 kNN ##########
kNN_classifier = KNeighborsClassifier(n_neighbors=6)
kNN_classifier.fit(X_train, y_train)
print(kNN_classifier.predict(x_predict))
