"""
4-6 网格搜索与 k 近邻算法中更多超参数
"""
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
sk_knn_clf = KNeighborsClassifier(n_neighbors=4, weights="uniform")
sk_knn_clf.fit(X_train, y_train)
sk_knn_clf.score(X_test, y_test)

# Grid Search
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
knn_clf = KNeighborsClassifier()

if __name__ == "__main__":
    grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    # 网格搜索中的最佳分类器
    print(grid_search.best_estimator_)
    # 最佳分类器的准确度
    print(grid_search.best_score_)
    # 最佳参数
    print(grid_search.best_params_)



