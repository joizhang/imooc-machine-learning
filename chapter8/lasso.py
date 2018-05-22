"""
8-9 LASSO
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from chapter8.core.lasso_regression import lasso_regression
from chapter8.core.ridge_regression import plot_model

np.random.seed(42)
x = np.random.uniform(-3., 3., size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)

np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# LASSO
lasso_reg = lasso_regression(20, 0.01)
lasso_reg.fit(X_train, y_train)
y1_predict = lasso_reg.predict(X_test)
print(mean_squared_error(y_test, y1_predict))
plot_model(x, y, lasso_reg)
