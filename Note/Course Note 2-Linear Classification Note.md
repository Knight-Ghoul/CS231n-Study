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
 然后取softmax处理后的score的交叉熵损失(cross entropy loss)作为**loss function**：
 $$
 L_i = -\log \left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right )
 $$
补充说明：我们希望我们的概率尽可能地集中在正确的标签上，即$ P(Y=k\mid X=x_i)\to 1$。之所以对$P$取$L_i=-\log P$ ,是因为损失函数是用来衡量坏的程度，我们要将其尽可能最小化，即当$P\to 1$时，$L_i\to 0$(而当$P\to 0$时，$L_i\to +\infin$).
>**视频中提到的问题(推荐思考)：**
- 问：softmax损失函数的最大值和最小值是多少？
- 答：$min = 0$,$max \to +\infin$,
- 解：当true class的score为 $+\infty$ 时，$P=1$,loss为0；当true class的score为 $-\infty$ 时，loss为$+\infty$。注意，这里的min/max都是理论上的极限值，并不能取到。不便于理解的话，画一下 $y=-\log x$的图，看区间[0，1]即可。

**补充：这里解释一下为什么cross entropy可以用作loss function。**
### cross entropy 交叉熵

熵(entropy)是一个事件所包含的信息量，定义为:$S(x) = -\sum_i{p(x_i)\log{p(x_i)}}.$
相对熵(relative entropy)又称为KL散度，是两个随机分布间距离大小的度量，定义为:
$$
D_{KL}(p||q) = E_p \left ( \log \frac{p(x)}{q(x)} \right ) = \sum_{x\in X}p(x) \log \left ( \frac{p(x)}{q(x)} \right ) \\
= \sum_{x\in X}p(x)\log p(x) - \sum_{x\in X}p(x)\log q(x),
$$
前一部分是 $p(x)$ 负熵，如果p(x)是已知的分布，这个值是定值；

**第二部分就是 $p(x)$ 和 $q(x)$ 之间的交叉熵(cross entropy)， 定义为:
$$
H(x) = - \sum_{x\in X}p(x)\log q(x)
$$**
$D_{KL}$ 越大，表示用 $q(x)$ 来近似 $p(x)$ 的差距越大，因为第一部分为定值，也就等价于第二部分cross entropy越大， 用 $q(x)$ 来近似 $p(x)$ 的差距越大，即$q(x)$ 和 $p(x)$ 的不相似度。

机器学习的过程是希望用一个模型的分布 $P(model)$ 来近似实际事件的分布 $P(real)$，但是自然界中实际事件的分布 $P(real)$ 是不可能得到的，退而求其次，我们收集该事件的大量样本，并用该样本的分布 $P(sample)$ 来代替 $P(real)$， 即 $P(sample) \cong P(real)$。从而机器学习的目的蜕化为使 $P(model)$ 与 $P(sample)$ 分布一致。 最小化  $P(model)$ 与 $P(sample)$ 的差异，等价于最小化两者的KL散度，也即最小化两者之间的cross entropy。

再回到softmax，这里 $p(x)$ 就是ground truth lable。以CIFAR10为例，这里 $p(x_i) = [0, 0, 0, 1, ... , 0]$，即truth class为1，其余都为0。因此，这里机器学习的目标就是用一个分布 $q(x)$ 来近似这个 $p(x)$ 。这里要求 $q(x)$ 是一个概率分布，这就是要将scores通过softmax function变成一个概率密度函数的原因。
接下来，要用 $q(x)$ 近似 $p(x)$，就要使两者之间的KL距离最小，等价于最小化两者之间的cross entropy。而$
\begin{aligned}
H(x) &= - \sum_{x\in X}p(x)\log q(x) \newline
&= - \left ( 0\times \log q(x_0) + 0\times \log q(x_0) + ... + 1\times \log q(x_i) + ... \right ) \newline
&= - \log q(x_i) \newline
&= -\log \left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right )
\end{aligned}
$
此即使用softmax和cross entropy的loss function。

###实操事项:数值稳定
编程实现softmax函数计算的时候，中间项$e^{f_{y_{i}}}$和$\displaystyle\sum_{j} e^{f_j}$因为存在指数函数，所以数值可能非常大。除以大数值可能导致数值计算的不稳定，所以学会使用归一化技巧非常重要。如果在分式的分子和分母都乘以一个常数$C$，并把它变换到求和之中，就能得到一个数学上等价的公式：
$$
\dfrac{e^{f_{y_i}}}{\sum_je^{f_j}}=\dfrac{Ce^{f_{y_i}}}{C\sum_je^{f_j}}=\dfrac{e^{f_{y_i}+\log C}}{\sum_je^{f_j+\log C}}
$$
$C$的值可自由选择，不会影响计算结果，通过使用这个技巧可以提高计算中的数值稳定性。通常将$C$设为$\log C=- \max_jf_j$。简单讲，就是将向量$f$中的数值全部进行平移，使得$f$取值为$(-\infty,0]$,$e^{f_{y_{i}}}$的取值为$(0,1]$

>**视频中提到的问题(和上式中的C无关)：**
- 问：通常初始化的时候$W$很小，所以所有的$s\thickapprox0$，那么损失值是多少？
- 解：$\log C$
###SVM与Softmax比较
![](https://pic1.zhimg.com/80/a90ce9e0ff533f3efee4747305382064_hd.png)

主要区别：我们如何解释这些class score 进而量化度量到底有多坏。
**在实际使用中，SVM和Softmax经常是相似的**

>**视频中提到的问题：**
- 问：假设我收到了一个数据点，并稍微改变了下它的分数，在这两钟不同情况下，会发生什么？
$$
\begin{array}{cc}
[\textcolor{#228B22}{10},-2,3]\\
[\textcolor{#228B22}{10},9,9]\\
[\textcolor{#228B22}{10},-100,-100]
\end{array}
and\ \ \ \ y_i=0
$$
- 答：对于SVM而言，即使正确分数稍微改变，也并不会对结果产生根本改变。对于softmax而言，给正确分类更高的分值，同时给不正确分类更低的分值，softmax仍会在正确的类上积累更多的概率质量，正确分类更加趋近于无穷大，不正确分类趋向于负无穷，永不满足。
- 解：对于SVM而言，唯一关心的是正确分类的分值要比不正确分类分值要高出一个安全边际；
对于softmax而言,softmax的目标是将概率质量函数分布等于1，SVM够大就行，softmax永不满足

#总结

- 定义了从图像像素映射到不同类别的分类评分的评分函数。在本节中，评分函数是一个基于权重W和偏差b的线性函数。
- 与kNN分类器不同，参数方法的优势在于一旦通过训练学习到了参数，就可以将训练数据丢弃了。同时该方法对于新的测试数据的预测非常快，因为只需要与权重W进行一个矩阵乘法运算。
- 介绍了偏差技巧，让我们能够将偏差向量和权重矩阵合二为一，然后就可以只跟踪一个矩阵。
- 定义了损失函数（介绍了SVM和Softmax线性分类器最常用的2个损失函数）。损失函数能够衡量给出的参数集与训练集数据真实类别情况之间的一致性。在损失函数的定义中可以看到，对训练集数据做出良好预测与得到一个足够低的损失值这两件事是等价的。
- 现在我们知道了如何基于参数，将数据集中的图像映射成为分类的评分，也知道了两种不同的损失函数，它们都能用来衡量算法分类预测的质量。但是，如何高效地得到能够使损失值最小的参数呢？这个求得最优参数的过程被称为最优化，将在下个笔记中讲到。
---

# Reference
1. [CS231n 官方笔记主页](http://cs231n.github.io/)
1. [CS231n 官方笔记授权翻译](https://zhuanlan.zhihu.com/p/21930884?refer=intelligentunit)
2. [CS231n课程及作业详细解读](https://github.com/FortiLeiZhang/cs231n)
