[Linear classification: Support Vector Machine, Softmax](http://cs231n.github.io/linear-classify/)
====
---
# 1.Linear Classification-线性分类概述
线性分类主要由两部分组成，**评分函数(score function)和损失函数(loss function)**。其中 **评分函数(score function)** 是原始图像数据(raw data)到类别分值(class score)的映射；而 **损失函数(loss function)** 是用来量化预测分类标签的得分与真实标签之间的一致性。所有的NN和CNN问题实际上都是围绕这两个function进行的。
# 2.评分函数(score function)

评分函数将图像的像素值映射为各个分类类别的得分，得分高低代表图像属于该类别的可能性高低。下面将举例说明(以CIFAR-10为例)：

假设我们有N(5000)个图像样例，每个图像维度为D(32*32*3=3072)，有K(10)个分类，定义**评分函数**为 $f=R^D\rarr R^K $ 表示原始图像像素到分类分值的映射。
## 线性分类器
提到 **评分函数** 我们最先想到的就是linear function。
$$ f(x_i,W,b)=Wx_i+b$$

在这个式子中每个图像数据被处理成长度为D的列向量，故$X_i$大小为[D X 1].
其中大小为[K x D]的矩阵$W$和大小为[K x 1]的列向量$b$为该函数的**参数(parameters)** .参数$W$被称为权重(weights)。$b$被称为偏差向量(bias vector),它影响输出数值，但是并不和原始数据$x_i$产生关联。
$Wx_i+b$=(1,3072)* (3072,10)+(1,10)=(1,10)+(1,10)=(1,10)表示 $x_i$分别属于10个类别下的得分。

![](https://pic3.zhimg.com/80/7c204cd1010c0af1e7b50000bfff1d8e_hd.jpg)
----
## 偏差和权重的合并技巧
分类评分函数定义为$ f(x_i,W,b)=Wx_i+b$ ，分开处理$W$和 $b$比较麻烦，一般常用的方法是**把两个参数放到同一个矩阵中**，同时$x_i$向量就要增加一个维度，这个维度的数值是常量1，即默认的偏差维度。此时新公式简化为:$f(x_i,W)=Wx_i$.

以CIFAR-10为例,$x_i$ 变为[3073x1],$W$变为[10x3073]
![](https://pic2.zhimg.com/80/3c69a5c87a43bfb07e2b59bfcbd2f149_hd.jpg)

## 图像数据预处理
对输入特征做归一化(normalization)处理，所有数据都减去均值使得数据**中心化**，再让数值分布区间变为[-1,1]。零均值的中心化的重要性在梯度下降中体现。

---

# 2.损失函数

我们将使用损失函数（Loss Function）来衡量我们对结果的不满意程度。直观地讲，当评分函数输出结果与真实结果之间差异越大，损失函数输出越大，反之越小。
## Multiclass Support Vector Machine Loss-多类支持向量机损失
损失函数的具体形式多种多样。首先，介绍常用的多类支持向量机(SVM损失函数。SVM的损失函数想要SVM在正确分类上的得分始终比不正确分类上的得分高出一个边界值$\Delta$。
设第$i$个数据，包含图像$x_i$的像素和代表正确类别的标签$y_i$.评分函数$f(x_i,W)$得到不同分类类别的分值，分值记作$s$。则 $x_i$对应第$j$个类别的得分为 $s_j=f(x_i,W) _ j $.针对第$i$个数据的多类SVM的损失函数定义如下：
$$
L_i=\displaystyle\sum_{j\not=y_i}\max(0,s_j-s_{y_i}+\Delta)
$$

---
# Reference
1. [CS231n 官方笔记主页](http://cs231n.github.io/)
1. [CS231n 官方笔记授权翻译](https://zhuanlan.zhihu.com/p/21930884?refer=intelligentunit)
2. [CS231n课程及作业详细解读](https://github.com/FortiLeiZhang/cs231n)
