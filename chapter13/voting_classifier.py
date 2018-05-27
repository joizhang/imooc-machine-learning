"""
13-1 什么是集成学习
"""
import numpy as np
from sklearn import datasets
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 逻辑回归
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# SVM
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# 决策树
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X, y)

# 集成
y_predict1 = log_reg.predict(X_test)
y_predict2 = svm_clf.predict(X_test)
y_predict3 = dt_clf.predict(X_test)
y_predict = np.array((y_predict1 + y_predict2 + y_predict3) >= 2, dtype='int')
print(y_predict[:10])
print(accuracy_score(y_test, y_predict))

# 使用 Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier())
], voting='hard')
voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_test, y_test))
