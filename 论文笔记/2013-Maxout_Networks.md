<!-- toc -->
[toc]
# 1. Maxout Networks
>arXiv:1302.4389 [stat.ML]
>tensorflow2代码：https://github.com/zhangkaihua88/ML_Paper

---

## 1.1. 摘要
maxout：
- 旨在通过dropout来加快优化过程，并提高准确度（与drop共同使用）；
- **模型的输出是模型输入的最大值**

---

## 1.2. 介绍
dropout
- 可以训练集成模型
- 共享参数并近似的对这些模型的预测进行了平均
- **一种不加区分的适用型工具，几乎可以应用于任何模型，都可以产生一定的性能改进。**


dropout与SDG
- **dropout**
在更新时最有效的方式是**使用更大的步长**，因为这样可以在不同的训练子集上对不同的模型有明显的影响来使得目标函数有持续的波动性，理想情况下整个训练过程就类似于使用[bagging](https://zh.wikipedia.org/wiki/Bagging%E7%AE%97%E6%B3%95)来训练集成的模型（带有参数共享的约束）。
- SGD
更新时一般会使用更小的步长，来使得目标函数平滑的下降。

对于深度网络模型，dropout只能作为模型平均的一种近似，显式的设计模型来最小化这种近似误差也可以提高dropout的性能。

---

## 1.3. 回顾droupt
在给定输入向量$v$后，输出预测向量$y$，该构成包含了一系列的隐含层$h=\{h^{(1)},...,h^{(L)}\}$。
Dropout训练一组由包含$v$和$h$中变量的子集组成的所有模型组成的模型。使用同一组参数$\theta$来标识一组分组$p(y | v ; \theta, \mu)$，其中$\mu \in M$是一个二进制掩码，用来决定模型中哪些变量参与运算。
每次在训练集上进行训练时，我们都按照$\log p(y | v ; \theta, \mu)$的梯度对不同的$\mu$随机取样训练不同的子模型。
可以通过$v$和$h$和掩码的按元素相乘得到不同子模型$p(y | v ; \theta, \mu)$的实例

当集合需要将所有子模型的预测平均起来进行预测时，函数形式就变得非常重要。
多个指数模型的预测平均值可以简单的通过运行权重除以2的完整模型来得到。即当$p(y|v;\theta) = softmax(v^Tw+b)$时，通过重整$p(y|v;\theta,\mu)$的几何平均定义的预测分布，可以很简单的由$softmax(v^tw/2+b)$



dropout与bagging
- 都是在不同子集上训练出不同模型
- dropout只训练一次，且所有模型共享参数。像是在训练一个模型集合而不是训练单个模型，每次的更新都必须有重大的影响，这样才能使得该子模型能较好的拟合当前的输入$v$
- bagging对子模型的输出进行算数平均，dropout是几何平均

---

## 1.4. 模型maxout描述
- 是一个简单的前馈框架模型
- 使用了一个新的激活函数：**maxout unit**
给定一个输入$ x \in \mathbb{R}^d$ （$x$可能是输入$v$,也可能是隐含层的状态），maxout隐含层的采用下式实现：
$$
h_{i}(x)=\max _{j \in[1, k]} z_{i j}
$$
其中$z_{i j}=x^{T} W_{\ldots i j}+b_{i j}$，$x \in \mathbb{R}^{d \times n}$，$W \in \mathbb{R}^{d \times m \times k}$, $b \in \mathbb{R}^{m \times k}$。$w$，$b$都是可训练参数。$k$表示每个隐藏节点对应k个“隐隐层”节点，这$k$个“隐隐层”节点都是线性输出。maxout的每个节点就从这k个“隐隐层”节点输出值中取最大的。所以使得maxout为一种非线性的变换

**Notes**
- maxout因为有参数同时为非线性，所以既可以是网络也可以是激活器
- 单个maxout单元可以解释为对任意凸函数进行线性逼近。（任意的凸函数都可由分段线性函数来拟合）。它在每处都是局部线性的（k个“隐隐层”节点都是线性的，取其最大值则为局部线性，分段的个数与k值有关），而一般的激活函数都有明显的曲率。
![20200206171729.png](https://image.zkhweb.top/20200206171729.png)
- 如同MLP一样，maxout网络也可以拟合任意连续函数。只要maxout单元含有任意多个“隐隐层”节点，那么只要两个隐层的maxout网络就可以实现任意连续函数的近似。
- maxout网络不仅可以学习到隐层之间的关系，还可以学习到每个隐层单元的激活函数。
- maxout放弃了传统激活函数的设计，它产生的表示不再是稀疏的，但是它的梯度是稀疏的，且dropout可以将它稀疏化。
- maxout没有上下界，所以让它在某一端饱和是零概率事件。
- 如果训练时使用dropout，则dropout操作在矩阵相乘之前，而并不对max操作的输入执行dropout。
- 使用maxout会默认一个先验：样本集是凸集可分的。

---

## 1.5. Maxout是一个通用的近似器

![20200206165136.png](https://image.zkhweb.top/20200206165136.png)

**命题1**：对于任意的正整数$m$,$n$,都存在两组$n+1$维的实数参数向量$[W_{1j}, b_{1j}], j \in [1, k]$和$[W_{2j}, b_{2j}], j \in [1, k]$使得$$ g(v) = h_1(v) - h_(v)$$ 即任意的分段连续函数都可以使用两个凸分段线性函数的差来表示。

**命题2**：根据Stone-Weierstrass近似定理，令$C$属于[紧空间(compact domain)](https://zh.wikipedia.org/wiki/%E7%B4%A7%E7%A9%BA%E9%97%B4)$C \subset \mathbb{R}^{n}$, 函数$f: C \rightarrow \mathbb{R}$是一个连续函数，一个正实数$\epsilon>0$。存在分段线性函数(PWL function)$g$，使得$v \in C,|f(v)-g(v)|<\epsilon$(取决于$\epsilon$)

**命题3**：万能近似理论：任何连续函数$f$，在紧空间上都可以使用具有两个maxout单元的maxout网络近似。

**证明**：
- 命题2，一个分段线性函数可以尽可能近似（取决于$\epsilon$）一个连续函数；
- 命题1，一个分段线性函数的表示正好和一个maxout网络完全匹配，该maxout网络具有两个maxout单元$h_1(v)$和$h_2(v)$，且k足够大的，可以达到所需的近似程度$ \epsilon$。
- 综上所述，我们可以得出结论：一个具有两个maxout单元的maxout网络可以任意程度的逼近任何一个紧空间内的连续函数$f(v)$。通常情况下，近似程度越大（即$\epsilon \rightarrow 0$），k越大（即$ k \rightarrow \infty$）。

---

## 1.6. maxout与relu
relu表达式$h_{i}(x)=\operatorname{relu}\left(x^{T} W_{\cdots i}+b_{i}\right)=\max \left(x^{T} W_{\cdots i}+b_{i}, 0\right)$
maxout表达式$h_{i}(x)=\max _{j \in[1, k]}\left(x^{T} W \ldots i j+b_{i j}\right)$

唯一区别
- relu使用的max(x,0)是对隐层每一个单元执行的与0比较最大化操作
- maxout是对$k$个“隐隐层”单元的值执行最大化操作(k为最大池化步长,max pooling stride。一般最大池化步长与k相等，如果步长小，则会有重叠最大池化)
（**一种实现方式**2-4-1，maxout是对5个“隐隐层”单元的值执行最大化操作。如果将“隐隐层”单元在隐层展开，那么隐层就有20个“隐隐层”单元，maxout做的就是在这20个中每5个取一个最大值作为最后的隐层单元，最后的隐层单元仍然为4个。实现的时候，可以将隐层单元数设置为20个，权重维度（2，20）偏置维度（1，20），然后在20个中每5个取一个最大值得到4个隐层单元。）

---

## 1.7. 模型平均
- 单层softmax有对模型进行平均的能力，但是通过观察，多层模型中使用dropout也存在这样的模型平均，只是有拟合精度的问题。
- 训练中使用dropout使得maxout单元有了更大的输入附近的线性区域，因为每个子模型都要预测输出，每个maxout单元就要学习输出相同的预测而不管哪些输入被丢弃。改变dropout mask将经常明显移动有效输入，从而决定了输入被映射到分段线性函数的哪一段。使用dropout训练的maxout具有一种特性，即当dropout mask改变时每个maxout单元的最大化滤波器相对很少变化。
- **maxout网络中的线性和最大化操作可以让dropout的拟合模型平均的精度很高**。而一般的激活函数几乎处处都是弯曲的，因而dropout的拟合模型平均的精度不高。

---

## 1.8. 优化
- 训练中使用dropout时，maxout的优化性能比relu+max pooling好
- dropout使用更大的步长最有效，使得目标函数有持续的波动性。而一般的SGD会使用更小的步长，来使得目标函数平滑的下降。dropout快速的探索着许多不同的方向然后拒绝那些损害性能的方向，而SGD缓慢而平稳的朝向最可能的方向移动。
- 实验中SGD使得relu饱和在0值的时间少于5%，而dropout则超过60%。由于relu激活函数中的0值是一个常数，这就会阻止梯度在这些单元上传播（无论正向还是反向），这也就使得这些单元很难再次激活，这会导致很多单元由激活转变为非激活。而maxout就不会存在这样的问题，梯度在maxout单元上总是能够传播，即使maxout出现了0值，但是这些0值是参数的函数可以被改变，从而maxout单元总是激活的。单元中较高比例的且不易改变的0值会损害优化性能。
- dropout要求梯度随着dropout mask的改变而明显改变，而一旦梯度几乎不随着dropout mask的改变而改变时，dropout就简化成为了SGD。relu网络的低层部分会有梯度衰减的问题（梯度的方差在高层较大而反向传播到低层后较小）。maxout更好的将变化的信息反向传播到低层并帮助dropout以类似bagging的方式训练低层参数。relu则由于饱和使得梯度损失，导致dropout在低层的训练类似于一般的SGD。



---

## 1.9. 参考资料
[Paper---Maxout Networks](https://arxiv.org/abs/1302.4389)
[CSD---Maxout Networks](https://blog.csdn.net/zhufenghao/article/details/52527047)
[CSDN---论文笔记_Maxout Networks](https://blog.csdn.net/maqian5/article/details/91880468)