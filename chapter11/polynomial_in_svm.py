"""
11-5 SVM中使用多项式特征和核函数
"""
import matplotlib.pyplot as plt
from sklearn import datasets


from chapter11.core.svm import polynomial_svc
from chapter11.core.svm import polynomial_kernel_svc
from chapter9.core.decision_boundary import plot_decision_boundary

X, y = datasets.make_moons(noise=0.15, random_state=666)
print(X.shape)

# 使用多项式特征的SVM
poly_svc = polynomial_svc(degree=3)
poly_svc.fit(X, y)
plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

# 使用多项式核函数的SVM
poly_kernel_svc = polynomial_kernel_svc(degree=6)
poly_kernel_svc.fit(X, y)
plot_decision_boundary(poly_kernel_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
