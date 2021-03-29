[Neural Networks Part 1: Setting up the Architecture](http://cs231n.github.io/neural-networks-1/)
===
[TOC]
#1. 神经网络
##1.1 概念简介
![](https://pic2.zhimg.com/80/d0cbce2f2654b8e70fe201fec2982c7d_hd.png)
- 大脑的基本计算单位是**神经元（neuron）**.。人类的神经系统中大约有860亿个神经元，它们被大约$10^{14}$ - $10^{15}$个 **突触（synapses）** 连接起来。每个神经元都从它的**树突**获得输入信号，然后沿着它唯一的**轴突（axon）**产生输出信号。轴突在末端会逐渐分枝，通过突触和其他神经元的树突相连。
- 在神经元的计算模型中，沿轴突传播的**信号($x_0$)** 将基于突触的**突触强度($w_0$)** 与其他神经元进行交互，**突触强度($w_0$)** 是可学习的且可以控制一个神经元对于另一个神经元的影响强度(通过控制$w_0$控制其兴奋或抑制)。之后**树突** 将信号传递到细胞体，多个信号累加之和若**高于某阈值** ，则**激活神经元**，向轴突输出一个峰值信号。。而神经元的激活率建模为**激活函数（activation function）$f$** 由于历史原因，激活函数常常选择使用**sigmoid函数$\sigma$**
##**1.2 常用激活函数**
### Sigmoid 
sigmoid函数的数学公式是$\sigma(x)=\dfrac{1}{1+e^{-x}}$，函数图像如下图所示，它输入实数值，并将输出限定在(0,1)，其中 $x \rightarrow -\infty$时取0，$x \rightarrow +\infty$时取1，梯度为$(1 - \sigma(x)) \cdot \sigma(x)$，**实际已经很少使用了**因为它有两个主要缺点
- 当 $\sigma(x)$ 为0或者1时，由 $(1 - \sigma(x)) \cdot \sigma(x)$ 可以看出，它的梯度都是0。**Sigmoid函数饱和会使梯度消失**，在反向传播时，与该局部梯度相乘的结果会接近0，无信号回传到数据。这就要求在初始化weight是要格外小心，以防 $\sigma(x)$ 函数进入饱和区。
- **Sigmoid函数的输出不是零中心的**，取值为(0,1),所以经过sigmoid以后的输出全部变成了正值，导致 $\mathrm{d} w$ 的符号完全取决于local grad。(要么全部是正数，要么去全部为负数，这将会导致梯度下降权重更新时出现z字型的下降。)
![](https://pic3.zhimg.com/80/677187e96671a4cac9c95352743b3806_hd.png)
### Tanh
Tanh函数可以看作是一个简单放大的sigmoid神经元，其函数表示为：$\tanh (x) = 2\sigma(2x) - 1$。值域为$(-1, 1)$，，和sigmoid神经元不同的是，它的输出是零中心的。但是依然有饱和区。

---
![](https://pic3.zhimg.com/80/83682a138f6224230f5b0292d9c01bd2_hd.png)
左边是ReLU（校正线性单元：Rectified Linear Unit）激活函数，当$x=0$时函数值为0。当$x>0$函数的斜率为1。右边是从 Krizhevsky等的论文中截取的图表，指明使用ReLU比使用tanh的收敛快6倍。
###ReLU
函数公式为$f(x)=\max(0,x)$,简单说，就是一个关于0的阈值。在近些年ReLU变得非常流行，它有以下一些优缺点:
- 优点：计算简单(减少计算资源的耗费)，没有饱和区，无需额外参数，对随机梯度下降的收敛有巨大加速作用
- 缺点：ReLU函数在<0的区域函数值和grad都为0，如果在ReLU前的节点输出全部为负值，那么所有流过该节点的梯度都变为0，从此之后再也不会被update到，对learning再无贡献，而且不会有任何机会被重新激活，也就是说，这个ReLU单元死亡，这将导致数据多样化的丢失。这种情况通常发生在relu backprop一个很大的grad，或者是learning rate选取的太大。
###Leaky ReLU
Leaky ReLU是为解决“ReLU死亡”问题的尝试。ReLU中当x<0时，函数值为0。而Leaky ReLU则是给出一个很小的负数梯度值，比如0.01。
其函数公式为$f(x)=1(x<0)(\alpha x)+1(x>=0)(x)$,其中$\alpha$是一个小的常量,表现得很不错，但其效果不是很稳定
###Maxout
函数：$\max(w^T_1x+b_1,w^T_2x+b_2)$,输出ReLU和leaky ReLU两组函数中取值较大的一个,然而和ReLU对比，它拥有ReLU单元的所有优点（线性操作和不饱和），而没有它的缺点（死亡的ReLU单元）,然而它的每个神经元的参数数量增加了一倍，这就导致整体参数的数量激增。
###总结
用ReLU，但是要注意learning rate的选取并且关注dead unit的比例。Leaky ReLU和Maxout可以试试，不要用sigmoid。
# 2 神经网络结构
神经元通过全连接层连接，层间神经元两两相连，但是层内神经元不连接；分层的结构能够让神经网络高效地进行矩阵乘法和激活函数运算；
##层组织
###将神经网络算法以神经元的形式图形化
- 神经网络被建模为神经元集合，神经元之间以无环图形式进行连接，通常神经网络模型中神经元是分层的。
- 全连接层（fully-connected layer）
是最普通的层的类型，全连接层中的神经元与其前后两层的神经元是完全成对连接的，但是在同一个全连接层内的神经元之间没有连接。下面是两个神经网络的图例，都使用的全连接层：
![](https://pic3.zhimg.com/80/ccb56c1fb267bc632d6d88459eb14ace_hd.png)
左边是一个2层神经网络，隐层由4个神经元（也可称为单元（unit））组成，输出层由2个神经元组成，输入层是3个神经元。右边是一个3层神经网络，两个含4个神经元的隐层。注意：层与层之间的神经元是全连接的，但是层内的神经元不连接。

    - 命名规则
        N层神经网络，不算输入层，单层神经网络即是输入直接映射到输出。
- 输出层 
和神经网络中其他层不同，输出层的神经元一般是不会有激活函数，因为最后的输出层大多用于表示分类评分值，因此是任意值的实数，或者某种实数值的目标数（比如在回归中）。
- 确定网络尺寸
用来度量神经网络的尺寸的标准主要有两个：一个是神经元的个数，另一个是参数的个数，用上面图示的两个网络举例：
    - 第一个网络有4+2=6个神经元（输入层不算），[3x4]+[4x2]=20个权重，还有4+2=6个偏置，共26个可学习的参数。
    - 第二个网络有4+4+1=9个神经元，[3x4]+[4x4]+[4x1]=32个权重，4+4+1=9个偏置，共41个可学习的参数。
## 设置层的数量和尺寸
不应该因为害怕出现过拟合而使用小网络。相反，应该进尽可能使用大网络，然后使用正则化技巧来控制过拟合。
#总结
- 介绍了生物神经元的粗略模型；
- 讨论了几种不同类型的激活函数，其中ReLU是最佳推荐；
- 介绍了**神经网络**，神经元通过全连接层连接，层间神经元两两相连，但是层内神经元不连接；
- 理解了分层的结构能够让神经网络高效地进行矩阵乘法和激活函数运算；
- 理解了神经网络是一个通用函数近似器，但是该性质与其广泛使用无太大关系。之所以使用神经网络，是因为它们对于实际问题中的函数的公式能够某种程度上做出“正确”假设。
- 讨论了更大网络总是更好的这一事实。然而更大容量的模型一定要和更强的正则化（比如更高的权重衰减）配合，否则它们就会过拟合。在后续章节中我们讲学习更多正则化的方法，尤其是dropout。

# Reference
1. [我的主页](https://github.com/Knight-Ghoul/CS231n-Study)
1. [CS231n 官方笔记主页](http://cs231n.github.io/)
1. [CS231n 官方笔记授权翻译](https://zhuanlan.zhihu.com/p/21930884?refer=intelligentunit)
2. [CS231n课程及作业详细解读](https://github.com/FortiLeiZhang/cs231n)