"""
5-9 使用scikit-learn解决回归问题
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

# scikit-learn 中的线性回归
reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))

# kNN解决回归问题
# kNN Regressor
knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train, y_train)
print(knn_reg.score(X_test, y_test))

# Grid Search
if __name__ == "__main__":
    param_grid = [
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 11)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 11)],
            'p': [i for i in range(1, 6)]
        }
    ]
    knn_reg = KNeighborsRegressor()
    grid_search = GridSearchCV(knn_reg, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_.score(X_test, y_test))
