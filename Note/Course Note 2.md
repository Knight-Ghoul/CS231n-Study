[Linear classification: Support Vector Machine, Softmax](http://cs231n.github.io/linear-classify/)
====
---
# 1.Linear Classification-线性分类概述
线性分类主要由两部分组成，**评分函数(score function)和损失函数(loss function)**。其中 **评分函数(score function)** 是原始图像数据(raw data)到类别分值(class score)的映射；而 **损失函数(loss function)** 是用来量化预测分类标签的得分与真实标签之间的一致性。所有的NN和CNN问题实际上都是围绕这两个function进行的。
# 2.score function-评分函数

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
## 多类SVM
损失函数的具体形式多种多样。首先，介绍常用的多类支持向量机(SVM)损失函数。SVM的损失函数想要SVM在正确分类上的得分始终比不正确分类上的得分高出一个边界值$\Delta$。
设第$i$个数据，包含图像$x_i$的像素和代表正确类别的标签$y_i$.评分函数$f(x_i,W)$得到不同分类类别的分值，分值记作$s$。则 $x_i$对应第$j$个类别的得分为 $s_j=f(x_i,W) _ j $.针对第$i$个数据的多类SVM的损失函数定义如下：
$$
L_i=\displaystyle\sum_{j\not=y_i}\max(0,s_j-s_{y_i}+\Delta)
$$
*补充说明* :$\max(0,-)$ 函数，常被称为**折叶损失(hinge loss)**

### Regularization-正则化
为了使得W的系数尽量小，模型尽量趋向简单，从而避免overfitting问题。我们向损失函数增加一个正则化惩罚$R(W)$,常用的正则化惩罚是**L2范式** :$R(W)=\displaystyle\sum_k \sum_l W_{k.l}^2,$.

最终完整的多类SVM损失函数有两个部分组成：**数据损失(data loss)** 和 **正则化损失(regularization loss)**,完整公式如下：
$$
L=\underbrace{\dfrac{1}{N}\displaystyle\sum_i L_i}_{data\ \  loss} +\underbrace{\lambda R(W)}_{regularization\ \ loss}
$$
其中，$N$是训练集的数据量。现在正则化惩罚添加到了损失函数里面，并用超参数$\lambda$来计算其权重。该超参数无法简单确定，需要通过交叉验证来获取。除了上述理由外，引入正则化惩罚还带来很多良好的性质，其中最好的性质就是对大数值权重进行惩罚，可以提升其泛化能力。
需要注意的是，和权重不同，偏差没有这样的效果，因为它们并不控制输入维度上的影响强度。因此通常只对权重$W$正则化，而不正则化偏差$b$。在实际操作中，可发现这一操作的影响可忽略不计。最后，因为正则化惩罚的存在，不可能在所有的例子中得到0的损失值，这是因为只有当$W=0$的特殊情况下，才能得到损失值为0。

## Softmax分类器
SVM是最常用的两个分类器之一，而另一个就是Softmax分类器，它的损失函数与SVM的损失函数不同。对于SVM而言，由score function得到的score仅仅相对大小是有意义的，每一项的绝对值并不表达任何意义。通过softmax function，可以将score与概率联系起来。softmax函数如下：
 $$ P(Y=k\mid X=x_i)= \frac{e^{s_k}}{\sum_j{e^{s_j}}}, $$
 先把所有scores用e指数化，得到均为正数，然后分母为所有指数的和，使得他们归一化。这样，每一class的score都变成了一个大于0小于1的数，而所有class score之和等于1，得到一个标准的概率分布。
 然后取softmax处理后的score的交叉熵损失(cross entropy)作为**loss function**：
 $$
 L_i = -\log \left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right )
 $$
补充说明：我们希望我们的概率尽可能地集中在正确的标签上，即$ P(Y=k\mid X=x_i)\to 1$。之所以对$P$取$L_i=-\log P$ ,是因为损失函数是用来衡量坏的程度，我们要将其尽可能最小化，即当$P\to 1$时，$L_i\to 0$(而当$P\to 0$时，$L_i\to +\infin$).

**视频中提到的问题(推荐思考)：**
>问：softmax损失函数的最大值和最小值是多少？
答：$min = 0$,$max \to +\infin$,
解：当true class的score为 $+\infty$ 时，$P=1$,loss为0；当true class的score为 $-\infty$ 时，loss为$+\infty$。注意，这里的min/max都是理论上的极限值，并不能取到。不便于理解的话，画一下 $y=-\log x$的图，看区间[0，1]即可。
---
# Reference
1. [CS231n 官方笔记主页](http://cs231n.github.io/)
1. [CS231n 官方笔记授权翻译](https://zhuanlan.zhihu.com/p/21930884?refer=intelligentunit)
2. [CS231n课程及作业详细解读](https://github.com/FortiLeiZhang/cs231n)