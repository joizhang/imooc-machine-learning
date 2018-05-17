"""
8-6 验证数据集与交叉验证
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

digits = datasets.load_digits()
X = digits.data
y = digits.target

best_score, best_p, best_k = 0, 0, 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=666)

# for k in range(2, 10):
#     for p in range(1, 6):
#         knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
#         knn_clf.fit(X_train, y_train)
#         score = knn_clf.score(X_test, y_test)
#         if score > best_score:
#             best_score, best_p, best_k = score, p, k
# print("Best K = ", best_k)
# print("Best P = ", best_p)
# print("Best score = ", best_score)

# 使用交叉验证
knn_clf = KNeighborsClassifier()
cross_val_score(knn_clf, X_train, y_train)
for k in range(2, 10):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        scores = cross_val_score(knn_clf, X_train, y_train)
        score = np.mean(scores)
        if score > best_score:
            best_score, best_p, best_k = score, p, k
print("Best K = ", best_k)
print("Best P = ", best_p)
print("Best score = ", best_score)

best_knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=2, p=2)
best_knn_clf.fit(X_train, y_train)
best_knn_clf.score(X_test, y_test)
