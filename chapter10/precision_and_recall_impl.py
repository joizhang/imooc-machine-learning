"""
10-3 实现混淆矩阵，精准率和召回率
"""
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from chapter4.core.metrics import FN
from chapter4.core.metrics import FP
from chapter4.core.metrics import TN
from chapter4.core.metrics import TP
from chapter4.core.metrics import confusion_matrix
from chapter4.core.metrics import precision_score
from chapter4.core.metrics import recall_score

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()
# 构造具有极度偏斜特征的数据
y[digits.target == 9] = 1
y[digits.target != 9] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))

y_log_predict = log_reg.predict(X_test)

print(TN(y_test, y_log_predict))
print(FP(y_test, y_log_predict))
print(FN(y_test, y_log_predict))
print(TP(y_test, y_log_predict))
print(confusion_matrix(y_test, y_log_predict))
print(precision_score(y_test, y_log_predict))
print(recall_score(y_test, y_log_predict))

# sklearn 中的混淆矩阵，精准率和召回率
