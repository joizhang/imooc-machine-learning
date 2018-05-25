from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

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

print(confusion_matrix(y_test, y_log_predict))
print(precision_score(y_test, y_log_predict))
print(recall_score(y_test, y_log_predict))