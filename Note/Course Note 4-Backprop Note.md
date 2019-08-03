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

