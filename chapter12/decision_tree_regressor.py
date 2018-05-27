"""
12-6 决策树解决回归问题
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

boston = datasets.load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
# 过拟合了，需要调参
print(dt_reg.score(X_train, y_train))
print(dt_reg.score(X_test, y_test))

# TODO 学习曲线
# TODO 模型复杂度曲线
