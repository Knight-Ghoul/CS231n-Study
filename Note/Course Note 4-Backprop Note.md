[Backpropagation, Intuitions](http://cs231n.github.io/optimization-2/)
=====
---

[toc]
---

# 1.简介

- **反向传播** 是利用链式法则递归计算表达式的梯度的方法。对于理解、实现、设计和调试神经网络*非常关键*。
- **核心问题**：给定函数$f(x)$,其中$x$是输入数据的向量，需要计算函数$f$关于$x$的梯度$\nabla f(x)$
- **研究目的**：在神经网络中$f$对应的是损失函数$L$，输入$x$里面包含训练数据和神经网络的权重。(例如SVM的损失函数，输入包含了训练数据$(x_i,y_i),i=1...N$、权重$W$和偏差$b$.)但训练集是给定的，而权重为可控制变量，**所以实践中为了进行参数更新，通常只计算参数(比如$W$,$B$)的梯度**
---

# 2.简单表达式和理解梯度
对于函数$f(x,y)=xy$。对于两个输入变量$x$$、y$分别求偏导数：
$$
\begin{matrix}
f(x,y)=xy\ \ \to\ \ \dfrac{df}{fx}=y & \dfrac{df}{dy}=x
\end{matrix}
$$
**函数关于每个变量的导数指明了整个表达式对于该变量的敏感程度。**
# 3.用链式法则计算复合表达式
- **链式法则**：考虑到更复杂的包含多个函数的复合函数，例如$f(x,y,z)=(x+y)z$可分为两部分$q=x+y$和$f=qz$;由$f=qz$可以得到$\dfrac{\partial f}{\partial q}=z$和$\dfrac{\partial f}{\partial z}=q$;又因为$q=x+y$，所以$\dfrac{\partial q}{\partial x}=1$,$\dfrac{\partial q}{\partial y}=1$;
不需要关心中间量q的梯度，关注$f$关于$x$,$y$,$z$的梯度。所以由链式法则可以得到$\dfrac{\partial f}{\partial x}=\dfrac{\partial f}{\partial q}\dfrac{\partial q}{\partial x}$,示例代码如下：
```python
# 设置输入值
x = -2; y = 5; z = -4

# 进行前向传播
q = x + y # q becomes 3
f = q * z # f becomes -12

# 进行反向传播:
# 首先回传到 f = q * z
dfdz = q # df/dz = q, 所以关于z的梯度是3
dfdq = z # df/dq = z, 所以关于q的梯度是-4
# 现在回传到q = x + y
dfdx = 1.0 * dfdq # dq/dx = 1. 这里的乘法是因为链式法则
dfdy = 1.0 * dfdq # dq/dy = 1
```
最后得到变量的梯度 **[dfdx, dfdy, dfdz]** ，它们告诉我们函数f对于变量 **[x, y, z]** 的敏感程度。这是一个最简单的反向传播。一般会使用一个更简洁的表达符号，用**dq**来代替**dfdq**，且总是假设梯度是关于最终输出的。以上代码能被可视化为计算图
![](https://pic4.zhimg.com/80/213da7f66594510b45989bd134fc2d8b_hd.jpg)
**前向传播**(绿色)：从输入计算到输出，**反向传播**(红色):从最终输出开始，根据链式法则递归地向前计算梯度，一直到网络的输入端。

# 4.反向传播的直观理解
- 反向传播是一个局部过程，在整个计算线路图中，每个**门单元**都会得到一些输入并立即计算两个东西：1. 这个门的输出值，和2.其输出值关于输入值的局部梯度。它完成这两件事是完全独立的，不需要知道计算线路中的其他细节，然而，一旦前向传播完毕，在反向传播的过程中，门单元将最终获得整个网络的最终输出值在自己的输出值上的梯度。*链式法则指出，门单元应该将回传的梯度乘以它对其的输入的局部梯度，从而得到整个网络的输出对该门单元的每个输入值的梯度。*
- 反向传播可以看做是门单元之间在通过梯度信号相互通信，只要让它们的输入沿着梯度方向变化，无论它们自己的输出值在何种程度上升或降低，都是为了让整个网络的输出值更高。

# 模块化 Sigmoid例子

![alt](https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/w%3D268%3Bg%3D0/sign=ba0ac7a864061d957d46303e43cf6dec/d009b3de9c82d158dfb4e7218a0a19d8bc3e426f.jpg)
函数 $f(x)=\dfrac{1}{1+e^{-x}} 被称为$sigmoid激活函数   $\sigma (x)$；其关于其输入的求导是可以简化的(这里只记结论)：
$$
\begin{aligned}
\sigma (x)=\dfrac{1}{1+e^{-x}}
 \to \frac{d\sigma (x)}{dx}=\dfrac{e^{-x}}{(1+e^{-x})^2}=(\dfrac{1+e^{-x}-1}{1+e^{-x}})(\dfrac{1}{1+e^{-x}})=(1-\sigma (x))\sigma (x)
\end{aligned}
$$
反向传播代码如下：
```python
w = [2,-3,-3] # 假设一些随机数据和权重
x = [-1, -2]

# 前向传播
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid函数

# 对神经元反向传播
ddot = (1 - f) * f # 点积变量的梯度, 使用sigmoid函数求导
dx = [w[0] * ddot, w[1] * ddot] # 回传到x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # 回传到w
# 完成！得到输入的梯度
```


- **实现提示:分段反向传播** :为了使反向传播过程更加简洁，把向前传播分成不同的阶段。比如我们创建了一个中间变量dot，它装着w和x的点乘结果。在反向传播的时，就可以（反向地）计算出装着w和x等的梯度的对应的变量（比如ddot，dx和dw）。
- **对前向传播变量进行缓存** ：在计算反向传播时，前向传播过程中得到的一些中间变量非常有用。在实际操作中，最好代码实现对于这些中间变量的缓存，这样在反向传播的时候也能用上它们。如果这样做过于困难，也可以（但是浪费计算资源）重新计算它们。
- **在不同分支的梯度要相加**：如果变量x，y在前向传播的表达式中出现多次，那么进行反向传播的时候就要非常小心，使用+=而不是=来累计这些变量的梯度（不然就会造成覆写）。这是遵循了在微积分中的多元链式法则，该法则指出如果变量在线路中分支走向不同的部分，那么梯度在回传的时候，就应该进行累加。
## 举例[便于理解，可略]：
$$
f(x,y)=\dfrac{x+\sigma (y)}{\sigma (x)+(x+y)^2}
$$
- **向前传播**
```python
x = 3 # 例子数值
y = -4

# 前向传播
sigy = 1.0 / (1 + math.exp(-y)) # 分子中的sigmoi          #(1)
num = x + sigy # 分子                                    #(2)
sigx = 1.0 / (1 + math.exp(-x)) # 分母中的sigmoid         #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # 分母                                #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # 搞定！ 
```
在构建代码s时创建了多个中间变量，每个都是比较简单的表达式，它们计算局部梯度的方法是已知的。这样计算反向传播就简单了：我们对前向传播时产生每个变量(sigy, num, sigx, xpy, xpysqr, den, invden)进行回传。我们会有同样数量的变量，但是都以d开头，用来存储对应变量的梯度。注意在反向传播的每一小块中都将包含了表达式的局部梯度，然后根据使用链式法则乘以上游梯度。对于每行代码，我们将指明其对应的是前向传播的哪部分。
- **反向传播**
```python
# 回传 f = num * invden
dnum = invden # 分子的梯度                                         #(8)
dinvden = num                                                     #(8)
# 回传 invden = 1.0 / den 
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# 回传 den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# 回传 xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# 回传 xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# 回传 sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# 回传 num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# 回传 sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# 完成! 嗷~~
```

# 回传流中的模式
`down_diff = local_diff * up_diff`其中up_diff是从上一层block传递下来的，local_diff要通过计算得到，并且和输入值有关，两者相乘传递给下一层的down_lock。
![](https://pic2.zhimg.com/80/39162d0c528144362cc09f1965d710d1_hd.jpg)
- **加法门单元[add gate]** : 
local_diff=1,up_diff不做任何改变均匀的传向两个分支。
- **取最大值门单元[max gate]** :
取最大值门将up_diff转给前向传播中的最大输入，因为最大值的局部梯度是1.0，其余的是0。上例中，取最大值门将梯度2.00转给了z变量，因为z的值比w高，于是w的梯度保持为0。
- **乘法门单元[multiply gate]**
local_diff，即该分支的grade为另一个分支的输入值。所以一个分支的输入如果过大的话，会导致另外一个分支的grad很大，造成梯v度爆炸。
具体的，score = wx，x是training data，通常是已知的不变的，所以不会计算对x的grad，只计算对w的grad $\mathrm{d} w$。如果输入数据x很大的话，由于w和x相乘，会造成 $\mathrm{d} w$ 很大.这样的后果就是，要么梯度会爆炸，要么要大大降低learning rate，使得学习变慢。**所以我们要对原始的输入数据进行预处理，减去均值；**
- **Sigmoid门单元**
$$
\begin{aligned}
 \frac{d\sigma (x)}{dx}=(1-\sigma (x))\sigma (x)
\end{aligned}
$$

# 用向量化操作计算梯度
- **矩阵相乘的梯度** ：可能最有技巧的操作是矩阵相乘（也适用于矩阵和向量，向量和向量相乘）的乘法操作：
```python
# 前向传播
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# 假设我们得到了D的梯度
dD = np.random.randn(*D.shape) # 和D一样的尺寸
dW = dD.dot(X.T) #.T就是对矩阵进行转置
dX = W.T.dot(dD)
```
不用记忆$\mathrm{dW}$和$\mathrm{dX}$的表达，用维度进行推导。权重的梯度$\mathrm{dW}$的尺寸肯定和权重矩阵$\mathrm{W}$的尺寸是一样的，而这又是由$\mathrm{X}$和$\mathrm{dD}$的矩阵乘法决定的,先算$\mathrm{dD}$,通过**维度分析**来计算 $\mathrm{dW}$。例如，X是(N, D)，W是(D, C)，那么Score = X.dot(W)是(N, C)。根据**维度分析**：想要$\mathrm{dW}$的尺寸为(D,C)
$$
\begin{aligned}
\mathrm{dW} &= X.T.\mathrm{dot} (\mathrm{dD} ) \\
\mathrm{dX} &= \mathrm{dD}.\mathrm{dot}(W.T)
\end{aligned}
$$
如果实在避免不了计算对vector的grad，那么就要项note里说的，先写出对vector中每一项的grad，然后再去general成vector形式，这里常用到的公式是视频里板书的那个:
$$
\frac{\partial f}{\partial X} = \sum_i \frac{\partial f}{\partial q_i} \cdot \frac{\partial q_i}{\partial X}，
$$
这里要注意的是写代码时np.sum()要对哪个axis进行。

---

# Reference
1. [CS231n 官方笔记主页](http://cs231n.github.io/)
1. [CS231n 官方笔记授权翻译](https://zhuanlan.zhihu.com/p/21930884?refer=intelligentunit)
2. [CS231n课程及作业详细解读](https://github.com/FortiLeiZhang/cs231n)
