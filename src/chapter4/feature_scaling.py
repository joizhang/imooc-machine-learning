"""
4-7 数据归一化
"""
import numpy as np
import matplotlib.pyplot as plt

# 最值归一化
x = np.random.randint(0, 100, size=100)
print((x - np.min(x)) / (np.max(x) - np.min(x)))

x = np.random.randint(0, 100, (50, 2))
x = np.array(x, dtype=float)
x[:, 0] = (x[:, 0] - np.min(x[:, 0])) / (np.max(x[:, 0]) - np.min(x[:, 0]))
x[:, 1] = (x[:, 1] - np.min(x[:, 1])) / (np.max(x[:, 1]) - np.min(x[:, 1]))
print(x[0:10, :])
plt.scatter(x[:, 0], x[:, 1])
plt.show()
print(np.mean(x[:, 0]))
print(np.std(x[:, 0]))
print(np.mean(x[:, 1]))
print(np.std(x[:, 1]))

# 均值方差归一化
x2 = np.random.randint(0, 100, (50, 2))
x2 = np.array(x2, dtype=float)

x2[:, 0] = (x2[:, 0] - np.mean(x2[:, 0])) / np.std(x2[:, 0])
x2[:, 1] = (x2[:, 1] - np.mean(x2[:, 1])) / np.std(x2[:, 1])
plt.scatter(x2[:, 0], x2[:, 1])
plt.show()
print(np.mean(x2[:, 0]))
print(np.std(x2[:, 0]))
print(np.mean(x2[:, 1]))
print(np.std(x2[:, 1]))
