[Optimization: Stochastic Gradient Descent](http://cs231n.github.io/optimization-1/)
====
---
# 1.Introduction-简介
通过线性分类，我们得知了图像分类任务中的两个关键部分:
1. 基于参数的评分函数(score function)。该函数将原始图像像素映射为分类评分值（例如：一个线性函数）。
2. 损失函数(loss function)。该函数能够根据分类评分和训练集图像数据实际分类的一致性，衡量某个具体参数集的质量好坏。损失函数有多种版本和不同的实现方式（例如：Softmax或SVM）。
其中,线性函数的形式是$ f(x_i,W)=Wx_i$
而SVM的实现的公式是:
$$
\begin{aligned}
L&=\underbrace{\dfrac{1}{N}\displaystyle\sum_i L_i}_{data\ \  loss} +\underbrace{\lambda R(W)}_{regularization\ \ loss}\\
&={\dfrac{1}{N}\displaystyle\sum_i\displaystyle\sum_{j\not=y_i}\max(0,f(x_i;W)_{j}-f(x_i;W)_{y_i}+\Delta)} +{\alpha R(W)}_{regularization\ \ loss}
\end{aligned}
$$
对于图像数据$x_i$，如果基于参数集$W_i$做出的分类预测与真实情况比较一致，那么计算出来的损失值$L$就很低。现在介绍第三个，也是最后一个关键部分：**最优化(Optimization)**,*最优化是寻找能使得损失函数值最小化的参数$W$的过程。*

后续安排：一旦理解了这三个部分是如何相互运作的，我们将会回到第一个部分（基于参数的函数映射），然后将其拓展为一个远比线性函数复杂的函数：首先是神经网络，然后是卷积神经网络。而损失函数和最优化过程这两个部分将会保持相对稳定。

---
# 2.Optimization-最优化
## 2.1 [略]随机搜索
- 随机尝试很多不同的权重，然后看其中哪个最好。
- **核心思路：迭代优化。** 当然，我们肯定能做得更好些。核心思路是：虽然找到- 最优的权重W非常困难，甚至是不可能的（尤其当W中存的是整个神经网络的权重的时候），但如果问题转化为：对一个权重矩阵集W取优，使其损失值稍微减少。那么问题的难度就大大降低了。换句话说，我们的方法从一个随机的W开始，然后对其迭代取优，每次都让它的损失值变得更小一点。
- **蒙眼徒步者的比喻** ：一个助于理解的比喻是把你自己想象成一个蒙着眼睛的徒步者，正走在山地地形上，目标是要慢慢走到山底。在CIFAR-10的例子中，这山是30730维的（因为W是3073x10）。我们在山上踩的每一点都对应一个的损失值，该损失值可以看做该点的海拔高度。
## 2.2 [略]随机本地搜索
第一个策略可以看做是每走一步都尝试几个随机方向，如果某个方向是向山下的，就向该方向走一步。这次我们从一个随机$W$开始，然后生成一个随机的扰动$\delta W$ ，只有当$W+\delta W$的损失值变低，我们才会更新。
## 2.3 [※]跟随梯度
- 前两个策略中，我们是尝试在权重空间中找到一个方向，沿着该方向能降低损失函数的损失值。
- 其实我们可以直接从数学上计算出最好的即最陡峭的方向。这个方向就是损失函数的**梯度（gradient）** 。
- 在一维函数中，斜率是函数在某一点的瞬时变化率。梯度是函数的斜率的一般化表达，它不是一个值，而是一个向量。在输入空间中，梯度是各个维度的斜率组成的向量（或者称为导数derivatives）;当函数有多个参数的时候，我们称导数为偏导数。而梯度就是在每个维度上偏导数所形成的向量。

# **3.梯度计算**
计算梯度有两种方法：一个是缓慢的近似方法（**数值梯度法**），但实现相对简单。另一个方法（**分析梯度法**）计算迅速，结果精确，但是实现时容易出错，且需要使用微分。
## 数值梯度法
- **利用有限差值计算梯度**$\dfrac{df(x)}{dx}=\dfrac{f(x+h)-f(x)}{h}$
根据上面的梯度公式，代码对所有维度进行迭代，在每个维度上产生一个很小的变化h，通过观察函数值变化，计算函数在该维度上的偏导数。
- **实践考量**：在数学公式中，$h\to 0$，然而在实际中，用一个很小的数值（比如例子中的1e-5）就足够了。且在实际中用中心差值公式（centered difference formula）$\dfrac{[f(x+h)-f(x-h)]}{2h}$效果较好。
- **在梯度负方向上更新** ：为了计算新的W，向着梯度df的负方向去更新，因为我们是希望损失函数值降低而不是升高。
- **步长的影响** ：梯度指明了函数在哪个方向是变化率最大的，但是没有指明在这个方向上应该走多远。步长（也叫作学习率）将会是神经网络训练中最重要（也是最头痛）的超参数设定之一。小步长，情况可能比较稳定但是进展较慢（这就是步长较小的情况）。相反，大步长结果也不一定尽如人意。在某些点如果步长过大，反而可能越过最低点导致更高的损失值。
- **效率问题**：计算数值梯度的复杂性和参数的量线性相关。有N个参数，所以损失函数每走一步就需要计算N次损失函数的梯度。现代神经网络很容易就有上千万的参数，因此这个问题只会越发严峻。显然这个策略不适合大规模数据，我们需要更好的策略。

## 分析梯度法
**微分分析计算梯度**:利用微分来分析，能得到计算梯度的公式（不是近似），用公式计算梯度速度很快，唯一不好的就是实现的时候容易出错。为了解决这个问题，在实际操作时常常将分析梯度法的结果和数值梯度法的结果作比较，以此来检查其实现的正确性，这个步骤叫做**梯度检查**。
### **举例(公式推导)**
以SVM的损失函数在某个数据点上的计算来说：
- 损失函数:
$$
\begin{aligned}
L_i = & \sum_{j \neq y_i}^C \max\left( 0, w_j x_i - w_{y_i} x_i + \Delta \right) \newline
= & \max\left( 0, w_0 x_i - w_{y_i} x_i + \Delta \right) + ... + \max\left( 0, w_j x_j - w_{y_i} x_i + \Delta \right) + ...
\end{aligned}
$$
- $L_i$ 对 $w_{y_i}$ 进行求导得到：
$$
\begin{aligned}
\mathrm{d}w_{y_i} =& \frac{\partial L_i}{\partial w_{y_i}} =
\mathbb{1} \left( w_0 x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) +
 ... + \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) + ... \newline
 =& - \left(  \sum_{j \neq y_i}^C  \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \right) \cdot x_i
 \end{aligned}
$$
- $L_i$ 对 $w_{j}$ 进行求导得到：
$$
\mathrm{d}w_j =  \frac{\partial L_i}{\partial w_j} = 0 + 0 + ... +
 \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \cdot x_i
$$
>其中 $\mathbb{1}$ 是一个示性函数，如果括号中的条件为真，那么函数值为1，如果为假，则函数值为0。虽然上述公式看起来复杂，但在代码实现的时候比较简单：只需要计算没有满足边界值的分类的数量（因此对损失函数产生了贡献），然后乘以$x_i$就是梯度了。将梯度的公式微分出来，代码实现公式并用于梯度更新就比较顺畅了。
### 梯度下降
**梯度下降**：程序重复地计算梯度然后对参数进行更新
```python
#普通的梯度下降

while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # 进行梯度更新
```
**小批量数据梯度下降**：在大规模的应用中，训练数据可以达到百万级量级。计算整个数据集去更新一个参数太过浪费，常用方法是计算训练集中的 **小批量数据(batcher)**
```python
# 普通的小批量数据梯度下降

while True:
  data_batch = sample_training_data(data, 256) # 256个数据
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # 参数更新
```

# 代码实现
这里完全照搬[CS231n课程及作业详细解读](https://github.com/FortiLeiZhang/cs231n)中的Course Note 3.md中的内容，写的实在是太好了
###### svm_naive

$\mathrm{d}W$ 必定与 $W$ 有同样的shape，这一点是今后计算grad必须要首先确定的。在这里，$\mathrm{d}W$的shape是(3073, 10)。接下来看 $L$ 的下标是 $i \in [0, N)$，即是N个sample之一，$w$ 的下标是 $j \in [0, C)$，即10个class之一。如果此列对应的不是true class，并且score大于0，就把这个sample的$x_i$加到 $\mathrm{d}W$ 的此列；如果此列对应的是true class，要计算其余9个class中，有几个的score大于0，然后与这个sample的$x_i$相乘，放到 $\mathrm{d}W$ 对应列。如此遍历N个sample结束。

###### svm_vectorize
这里介绍非常重要的维数分析法，该方法可大大简化vectorize的分析过程，而且不易出错。首先score是X和W的函数，即：
$$
Score = X.dot(W)
$$
所以，$\mathrm{d}W$必定是由 $\mathrm{d} Score$ 和X计算得出。这里X是(N, 3073)，W是(3073, 10)，所以Score是(N, 10)，而 $\mathrm{d} Score$ 必定与Score的shape相同，所以 $\mathrm{d} Score$ 也是(N, 10)，这样，根据矩阵相乘的维数限制，可以得到
$$
\mathrm{d} W = X.T.dot(\mathrm{d} Score)。
$$
由公式推导可以得到 $\mathrm{d} Score$：
$$
\mathrm{d}s_j = \mathbb{1} \left( s_j - s_{y_i} + \Delta > 0\right)
$$
$$
\mathrm{d}s_{y_i}
 = - \sum_{j \neq y_i}^C  \mathbb{1} \left( s_j - s_{y_i} + \Delta > 0\right)
$$
即对Score的每一列，如果不是true class，且score>0，该位置 $\mathrm{d} Score$ 为1，否则为0；如果是true class，该位置的数值是此列不为0的个数。


#### [Assignment 1: softmax grads的计算](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/softmax.ipynb)

##### 公式推导
还是要stage到score级别，然后再用 $\mathrm{d} W = X.T.dot(\mathrm{d} Score)$来计算 $\mathrm{d} W$，这样可以在推导的时候不用考虑如何计算对两个矩阵相乘。
$$
L_i = - \log \left( \ p_{y_i} \right) = -\log \left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right )
$$

$L_i$ 对任意 $s_k$ 求导：
$$
\begin{aligned}
\mathrm{d} s_k =& \frac{\partial L_i}{\partial s_k} = - \frac{\partial}{\partial s_k} \left( \log \left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right ) \right) \newline
=& - \frac{\sum_j e^{s_j}}{e^{s_{y_i}}} \cdot \frac{\left( {e^{s_{y_i}}}\right)^{'} \cdot {\sum_j e^{s_j}} - {e^{s_{y_i}}} \cdot \left( {\sum_j e^{s_j}} \right)^{'}}{\left( {\sum_j e^{s_j}}\right)^2} \newline
=&\frac{\frac{\partial}{\partial s_k}\left( {\sum_j e^{s_j}} \right)}{{\sum_j e^{s_j}}} - \frac{ \frac{\partial }{\partial s_k} \left({e^{s_{y_i}}} \right)}{{e^{s_{y_i}}}} \newline
=&\frac{\frac{\partial}{\partial s_k}\left( e^{s_0} + e^{s_1} + e^{s_{y_i}} + ... \right)}{{\sum_j e^{s_j}}} - \frac{ \frac{\partial }{\partial s_k} \left({e^{s_{y_i}}} \right)}{{e^{s_{y_i}}}}
\end{aligned}
$$
当 $y_i = k$时：
$$
\mathrm{d} s_k = \frac{{e^{s_{y_i}}}}{{\sum_j e^{s_j}}} - 1
$$
当 $y_i \neq k$时：
$$
\mathrm{d} s_k = \frac{{e^{s_k}}}{{\sum_j e^{s_j}}}
$$
综上，
$$
\mathrm{d} s_k = \frac{{e^{s_k}}}{{\sum_j e^{s_j}}} - \mathbb{1} \left( y_i = k \right)
$$

##### [代码实现](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/cs231n/classifiers/softmax.py)

###### softmax_naive
###### softmax_vectorize
有了上面的公式推导和svm的经验，这里的代码不难写。注意，我们这里都是先去计算 $\mathrm{d} Score$, 然后再用 $\mathrm{d} W = X.T.dot(\mathrm{d} Score)$ 来计算。

---

# Reference
1. [CS231n 官方笔记主页](http://cs231n.github.io/)
1. [CS231n 官方笔记授权翻译](https://zhuanlan.zhihu.com/p/21930884?refer=intelligentunit)
2. [CS231n课程及作业详细解读](https://github.com/FortiLeiZhang/cs231n)
