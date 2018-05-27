"""
13-4 oob (Out-of-Bag) 和关于 Bagging 的更多讨论
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

if __name__ == "__main__":
    # oob_score：使用oob
    # n_jobs：使用多核运算
    # max_features：对特征随机取样（对图像识别任务比较适合）
    # bootstrap_features：True 表示放回
    bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                    n_estimators=500,
                                    max_samples=100,
                                    bootstrap=True,
                                    oob_score=True,
                                    n_jobs=-1)
    bagging_clf.fit(X, y)
    print(bagging_clf.oob_score_)

    random_subspace_clf = BaggingClassifier(DecisionTreeClassifier(),
                                            n_estimators=500,
                                            max_samples=500,
                                            bootstrap=True,
                                            oob_score=True,
                                            n_jobs=-1,
                                            max_features=1,
                                            bootstrap_features=True)
    random_subspace_clf.fit(X, y)
    print(random_subspace_clf.oob_score_)

    random_patches_clf = BaggingClassifier(DecisionTreeClassifier(),
                                           n_estimators=500,
                                           max_samples=100,
                                           bootstrap=True,
                                           oob_score=True,
                                           n_jobs=-1,
                                           max_features=1,
                                           bootstrap_features=True)
    random_patches_clf.fit(X, y)
    print(random_patches_clf.oob_score_)
