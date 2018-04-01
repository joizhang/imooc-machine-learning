"""
4-4 分类准确度
"""
import matplotlib
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model_selection
from sklearn import datasets

import chapter4.core.model_selection as my_model_selection
from chapter4.core import knn
from chapter4.core import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

plt.ion()

digits = datasets.load_digits()
X = digits.data
print(X.shape)
y = digits.target
print(y.shape)

some_digit = X[666]
print(y[666])
some_digit_image = some_digit.reshape(8, 8)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
# plt.show()

# 自定义
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
