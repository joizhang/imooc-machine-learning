"""
4-8 scikit-learn中的Scaler
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X[:10, :])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

# ########## scikit-learn 中的 StandardScalar ##########
standardScaler = StandardScaler()
standardScaler.fit(X_train)
# 均值
print(standardScaler.mean_)
# 标准差
print(standardScaler.scale_)
# 进行归一化处理
X_train = standardScaler.transform(X_train)
print(X_train[:10, :])
# 要获得较高的准确率，那么测试数据集也必须归一化
X_test_standard = standardScaler.transform(X_test)
# 用kNN分类
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test_standard, y_test))
