import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print(boston.feature_names[np.argsort(lin_reg.coef_)])