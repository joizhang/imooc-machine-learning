"""
10-8 多分类问题中的混淆矩阵
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=666)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))
y_predict = log_reg.predict(X_test)

# 多分类的精准率
print(precision_score(y_test, y_predict, average='micro'))

# 多分类的混淆矩阵
cfm = confusion_matrix(y_test, y_predict)
print(cfm)
# plt.matshow(cfm, cmap=plt.cm.gray)
# plt.show()

row_sums = np.sum(cfm, axis=1)
err_matrix = cfm / row_sums
np.fill_diagonal(err_matrix, 0)
# plt.matshow(err_matrix, cmap=plt.cm.gray)
# plt.show()
