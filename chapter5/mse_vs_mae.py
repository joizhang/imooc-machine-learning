import matplotlib.pyplot as plt
from sklearn import datasets

from chapter4.core.metrics import mean_absolute_error
from chapter4.core.metrics import mean_squared_error
from chapter4.core.metrics import root_mean_squared_error
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
print(reg.a_, reg.b_)

plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color="r")
plt.show()

y_predict = reg.predict(x_test)

# MSE
mse_test = mean_squared_error(y_test, y_predict)
print(mse_test)

# RMSE
rmse_test = root_mean_squared_error(y_test, y_predict)
# 平均误差就在下面这个数字万美元左右
print(rmse_test)

# MAE
mae_test = mean_absolute_error(y_test, y_predict)
print(mae_test)

# scikit-learn 中的 MSE 和 MAE

