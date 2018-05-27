"""
13-5 随机森林和 Extra-Trees
"""
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

if __name__ == "__main__":
    # 随机森林
    rf_clf = RandomForestClassifier(n_estimators=500,
                                    random_state=666,
                                    oob_score=True,
                                    n_jobs=-1)
    rf_clf.fit(X, y)
    print(rf_clf.oob_score_)

    rf_clf2 = RandomForestClassifier(n_estimators=500,
                                     max_leaf_nodes=16,
                                     random_state=666,
                                     oob_score=True,
                                     n_jobs=-1)
    rf_clf2.fit(X, y)
    print(rf_clf2.oob_score_)

    # 使用 Extra_Trees
    et_clf = ExtraTreesClassifier(n_estimators=500,
                                  bootstrap=True,
                                  oob_score=True,
                                  random_state=666)
    et_clf.fit(X, y)
    print(et_clf.oob_score_)
