"""
4-3 训练数据集 测试数据集
"""

import sklearn.model_selection as sk_model_selection
from sklearn import datasets

import chapter4.core.model_selection as my_model_selection
from chapter4.core import knn

iris = datasets.load_iris()
X = iris.data
y = iris.target

# 自定义的 train_test_split
X_train, X_test, y_train, y_test = my_model_selection.train_test_split(X, y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

my_knn_clf = knn.KNNClassifier(k=3)
my_knn_clf.fit(X_train, y_train)
y_predict = my_knn_clf.predict(X_test)
print(sum(y_predict == y_test) / len(y_test))

# sklearn 中的 train_test_split
X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y, test_size=0.2, random_state=666)
