"""
11-4 scikit-learn中的SVM
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from chapter9.core.decision_boundary import plot_decision_boundary
from chapter9.core.decision_boundary import plot_svc_decision_boundary

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y < 2, :2]
y = y[y < 2]

standardScaler = StandardScaler()
standardScaler.fit(X)
X_standard = standardScaler.transform(X)

svc = LinearSVC(C=1e9)
svc.fit(X_standard, y)
plot_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y == 0, 0], X_standard[y == 0, 1], color='red')
plt.scatter(X_standard[y == 1, 0], X_standard[y == 1, 1], color='blue')
plt.show()

plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y == 0, 0], X_standard[y == 0, 1], color='red')
plt.scatter(X_standard[y == 1, 0], X_standard[y == 1, 1], color='blue')
plt.show()

svc2 = LinearSVC(C=0.01)
svc2.fit(X_standard, y)
plot_decision_boundary(svc2, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y == 0, 0], X_standard[y == 0, 1], color='red')
plt.scatter(X_standard[y == 1, 0], X_standard[y == 1, 1], color='blue')
plt.show()

plot_svc_decision_boundary(svc2, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y == 0, 0], X_standard[y == 0, 1], color='red')
plt.scatter(X_standard[y == 1, 0], X_standard[y == 1, 1], color='blue')
plt.show()
