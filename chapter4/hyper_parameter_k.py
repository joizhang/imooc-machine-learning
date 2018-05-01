"""
4-4 分类准确度
4-5 超参数
"""
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model_selection
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import chapter4.core.model_selection as my_model_selection
from chapter4.core import knn
from chapter4.core import metrics

plt.ion()

digits = datasets.load_digits()
X = digits.data
print(X.shape)
y = digits.target
print(y.shape)

some_digit = X[666]
print(y[666])
some_digit_image = some_digit.reshape(8, 8)
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
# plt.show()

# 自定义的 accuracy_score
X_train, X_test, y_train, y_test = my_model_selection.train_test_split(X, y, test_ratio=0.2)
my_knn_clf = knn.KNNClassifier(k=3)
my_knn_clf.fit(X_train, y_train)
y_predict = my_knn_clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))
print(my_knn_clf.score(X_test, y_test))

# scikit-learn 中的 accuracy_score
X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y, test_size=0.2, random_state=666)
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_test)
print(accuracy_score(y_test, y_predict))

# 寻找最好的 k，如果在边界，需要继续寻找
best_score = 0.0
best_k = -1
for k in range(1, 11):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    score = knn_clf.score(X_test, y_test)
    if score > best_score:
        best_k = k
        best_score = score
print("best_k = ", best_k, "\nbest_score = ", best_score)

# 考虑欧拉距离
print("\n考虑欧拉距离")
best_method = ""
best_score = 0.0
best_k = -1
for method in ["uniform", "distance"]:
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_method = method
print("best_k = ", best_k, "\nbest_score = ", best_score, "\nbest_method = ", best_method)

# 考虑明可夫斯基距离
print("\n考虑明可夫斯基距离")
best_p = ""
best_score = 0.0
best_k = -1
for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_p = p
print("best_k = ", best_k, "\nbest_score = ", best_score, "\nbest_p = ", best_p)
