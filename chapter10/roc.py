"""
10-7 ROC曲线
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from chapter4.core.metrics import FPR
from chapter4.core.metrics import TPR

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target == 9] = 1
y[digits.target != 9] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
decision_scores = log_reg.decision_function(X_test)

fprs = []
tprs = []
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
for threshold in thresholds:
    y_predict = np.array(decision_scores >= threshold, dtype='int')
    fprs.append(FPR(y_test, y_predict))
    tprs.append(TPR(y_test, y_predict))
plt.plot(fprs, tprs)
plt.show()

# sklearn 中的 ROC
fprs, tprs, threshold = roc_curve(y_test, decision_scores)
plt.plot(fprs, tprs)
plt.show()
# 求ROC曲线下面的面积
print(roc_auc_score(y_test, decision_scores))
