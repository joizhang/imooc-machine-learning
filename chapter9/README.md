# 逻辑回归 Logistic Regression

逻辑回归：解决分类问题

回归问题怎么解决分类问题：将样本的特征和样本发生的概率联系起来，概率是一个数

![逻辑回归](images/逻辑回归.png)

逻辑回归既可以看作是回归算法，也可以看作是分类算法，通常作为分类算法用，只可以解决二分类问题

## Sigmoid 函数

![逻辑回归1](images/逻辑回归1.png)

![Sigmoid](images/Sigmoid.png)

sigmoid 函数值域(0,1)，t > 0 时，p > 0.5，t < 0 时，p < 0.5

![Sigmoid1](images/Sigmoid1.png)

## 损失函数

![损失函数1](images/损失函数1.png)

![损失函数2](images/损失函数2.png)

![损失函数3](images/损失函数3.png)

![损失函数4](images/损失函数4.png)

前一部分的求导过程

![损失函数5](images/损失函数5.png)

![损失函数6](images/损失函数6.png)

后一部分的求导过程

![损失函数7](images/损失函数7.png)

最终结果

![损失函数8](images/损失函数8.png)

![损失函数9](images/损失函数9.png)

![损失函数10](images/损失函数10.png)

向量化

![损失函数11](images/损失函数11.png)

## 决策边界

![决策边界](images/决策边界.png)

## 逻辑回归中使用正则化

![正则化](images/正则化.png)

## 逻辑回归解决多分类

### OvR(One vs Rest)

![OvR](images/OvR.png)

### OvO()

![OvO](images/OvO.png)