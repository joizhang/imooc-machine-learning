### 多项式回归与模型泛化

#### 多项式回归

![多项式回归](images/多项式回归.png)

PolynomialFeatures(degree=3)

![PolynomialFeatures](images/PolynomialFeatures.png)

#### 过拟合与欠拟合

过拟合与欠拟合的泛化能力都较差

过拟合（overfitting）：算法所训练的模型过多的表达了数据见的噪音关系

#### 为什么要有训练数据集与测试数据集

![为什么要有训练数据集和测试数据集](images/为什么要有训练数据集和测试数据集.png)

#### 学习曲线

随着训练样本的逐渐增都，算法训练出的模型的表现能力

欠拟合

![欠拟合](images/欠拟合.png)

过拟合

![欠拟合](images/过拟合.png)

#### 验证数据集与交叉验证

可能发生针对特定测试数据集过拟合了

![验证数据集](images/验证数据集.png)

![交叉验证](images/交叉验证.png)

##### k-folds 交叉验证

把训练数据集分成k份，称为k-folds cross validation

缺点，每次训练k个模型，相当于整体性能慢了k倍

##### 留一法 LOO-CV

把训练数据集分成m份，称为留一法（Leave-One-Out Cross Validation），完全不受随机的影响，最接近模型真正的性能指标，缺点是计算量巨大

#### 偏差和方差

![偏差和方差](images/偏差和方差.png)