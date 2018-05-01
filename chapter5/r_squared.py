from sklearn import datasets

from chapter4.core.metrics import r2_score
from chapter4.core.model_selection import train_test_split
from chapter5.core.simple_linear_regression import SimpleLinearRegression2

boston = datasets.load_boston()
x = boston.data[:, 5]
y = boston.target

x = x[y < 50.0]
y = y[y < 50.0]

x_train, x_test, y_train, y_test = train_test_split(x, y, seed=666)

reg = SimpleLinearRegression2()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

print(r2_score(y_test, y_predict))
print(reg.score(x_test, y_test))

# scikit-learn 中的 R Square
