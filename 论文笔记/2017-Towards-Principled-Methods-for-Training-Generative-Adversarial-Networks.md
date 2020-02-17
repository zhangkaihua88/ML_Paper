<!-- toc -->
[toc]

---

# 1. Towards Principled Methods for Training Generative Adversarial Networks
>arXiv:1701.04862 [cs, stat]
>tensorflow2代码：https://github.com/zhangkaihua88/ML_Paper

---

# 2. 总结
- 要解决什么问题
    - 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
        - 第一种生成器loss面临梯度消失问题
        - 第二种生成器loss面临优化目标荒谬、梯度不稳定、对多样性与准确性惩罚不平衡导致mode collapse这几个问题。
    - 基本解决了collapse mode的问题，确保了生成样本的多样性
    - 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高
    - 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到
    
- 用什么方法解决
- 还存在什么问题

- 算法流程(相对于原始GAN)
    - 判别器最后一层去掉sigmoid
    - 生成器和判别器的loss不取log
    - 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
    - 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行
![20200213234842.png](https://image.zkhweb.top/20200213234842.png)

---

# 摘要
为全面理解生成式对抗网络的训练动力学做出理论上的一步
- 介绍了眼前的问题
- 专门研究和严格证明训练生成对抗网络时出现的不稳定和饱和问题
- 探讨了解决这些问题的实用且理论基础的方向

---

# 引言
- GAN难以训练，解决问题还依赖于对修改极其敏感的启发式方法
- 没有理论分析GAN训练的不稳定原因

GAN所用的生成器和其他方法(VAE)并没有明显差别
- 都是从一个简单先验进行采样$z\sim p(z)$，然后输出最后的采样$g_{\theta}(z)$，有时候会在最后加上噪音.总之，$g_{\theta}$是一个受到参数$\theta$控制的神经网络，主要的差别就是$g_{\theta}$是如何训练的。

- 生成模型依赖最大似然
- 等效于最小化未知的真实数据分布$\mathbb{P}_r$和生成器分布$\mathbb{P}_g$（依赖于$\theta$）之间的KL散度。如果假设这两个分布都是连续密度$P_r$和$P_g$，那么这些方法就是最小化：
$$
K L\left(\mathbb{P}_{r} \| \mathbb{P}_{g}\right)=\int_{\mathcal{X}} P_{r}(x) \log \frac{P_{r}(x)}{P_{g}(x)} \mathrm{d} x
$$
这个损失函数有很好的特性，其有唯一最小值，当且仅当$\mathbb{P}_r = \mathbb{P}_g$，而且优化时，不需要事先知道$P_r(x)$的相关信息，只需要采样。
但是**当$\mathbb{P}_r$和$\mathbb{P}_g$之间不是对称的时候就有趣了**：
    - 如果$P_r(x)>P_g(x)$，那么$x$是一个数据点，其来自真实数据的概率要远大于生成的。该现象的本质通常被描述为“mode dropping”[^modedropping]：当很大区域是$P_r$的高值，而$P_g$是很小或者零值。当$P_r(x)>0$但是$P_g(x)\rightarrow 0$时，KL内部的被积函数迅速增长到无穷大，这意味着这个损失函数赋予生成器的分布（那些没有覆盖到数据的部分）极大的cost值。
    - 如果$P_r(x)<P_g(x)$，那么$x$表示来自生成的概率远大于真实数据的。这种情况下我们可以看到生成器的输出基本看上去很假。当$P_r(x) \rightarrow 0$且$P_g(x)>0$时，发现KL中的值接近为0，意味着损失函数会将极低的值给生成器（此时生成的数据看上去很假）。
[^modedropping]:样本多样性缺失，

**最小化$KL(\mathbb{P}_g||\mathbb{P}_r)$**
损失函数的权重就会反过来，即损失函数会在生成很假样本的时候给予很高的cost值。

GAN是优化（最原始形态）Jensen-Shannon散度(Jensen-shannon divergence, JSD)，这两个cost的转换形式为：
$$
J S D\left(\mathbb{P}_{r} \| \mathbb{P}_{g}\right)=\frac{1}{2} K L\left(\mathbb{P}_{r} \| \mathbb{P}_{A}\right)+\frac{1}{2} K L\left(\mathbb{P}_{g} \| \mathbb{P}_{A}\right)
$$
这里$\mathbb{P}_A$是平均分布，密度为$\frac{P_r+P_g}{2}$。
**GAN成功生成看似真实图像的原因**：是由于传统的最大似然方法的转换

GAN形式化成2个步骤，首先训练一个判别器D去最大化：
$$L(D,g_{\theta})=\mathbb{E}_{x\sim \mathbb{P}_r}\left[ \log D(x)\right]+\mathbb{E}_{x\sim \mathbb{P}_g}\left[ \log(1-D(x))\right] $$
可以发现最优判别器形如：
$$D^*(x)=\frac{P_r(x)}{P_r(x)+P_g(x)} $$

$$L(D^*,g_{\theta})=2JSD(\mathbb{P}_r||\mathbb{P}_g)-2\log2$$
当鉴别器是最优的时，最小化等式$J S D\left(\mathbb{P}_{r} \| \mathbb{P}_{g}\right)=\frac{1}{2} K L\left(\mathbb{P}_{r} \| \mathbb{P}_{A}\right)+\frac{1}{2} K L\left(\mathbb{P}_{g} \| \mathbb{P}_{A}\right)$看成最小化Jensen-Shannon散度的$\theta$的函数。

理论上，我们期望首先尽可能最优的训练判别器（所以$\theta$时候的cost函数近似JSD），然后在$\theta$上进行梯度迭代，然后交替这2个事情。然而，这并不work。判别器目标函数越小，则实际上就是$P_r$和$P_g$之间的JS散度越小，通过优化JS散度就能将$P_g$“拉向”$P_r$，最终以假乱真。
实际上，判别器越好，对生成器的更新就会越糟糕，原始GAN论文认为这个问题主要来自饱和，换成另一个相似cost函数就不会有这个问题了。然而即使使用新的cost函数，更新还是会变得更糟，优化也会变得更不稳定。


**就有下面几个问题**：

- 为什么判别器变得越来越好，而更新却越来越差？在原始cost函数和新的cost函数都是这样；
- 为什么GAN训练这么不稳定；
- 是否新的cost函数和JSD是一样的相似散度？如果是，他的特性是什么？；
- 有方法避免这些问题么？

---

# 不稳定的来源
>**概念术语**
>- 支撑集（support）其实就是函数的非零部分子集，比如ReLU函数的支撑集就是(0, +$\infty$)，一个概率分布的支撑集就是所有概率密度非零部分的集合。
>- 流形（manifold）是高维空间中曲线、曲面概念的拓广，我们可以在低维上直观理解这个概念，比如我们说三维空间中的一个曲面是一个二维流形，因为它的本质维度（intrinsic dimension）只有2，一个点在这个二维流形上移动只有两个方向的自由度。同理，三维空间或者二维空间中的一条曲线都是一个一维流形。
>- 测度（measure）是高维空间中长度、面积、体积概念的拓广，可以理解为“超体积”

## 判别器越好，生成器梯度消失越严重

理论上训练的判别器的cost基本就是$2\log2-2JSD(\mathbb{P}_r||\mathbb{P}_g)$。
实际上，如果只训练D直到收敛，它的误差接近0。可以得知判别器已经完全胜出了(D训练得更较精确，G的更新会变得越差)，并不是均衡。
这种情况发生的时候是
- 分布是非连续的（附录B1）
- 支撑集（supports）不相交。

>1通过连续，我们实际上将引用一个绝对连续的随机变量（即具有密度的变量），如通常所做的那样。 有关进一步的说明，请参见附录B。常见的分布一般都有密度函数

**分布非连续的原因**(没有密度函数)
- 他们的支撑集（supports）位于低维度流形(流形维度低于全空间)上。文献中证明$\mathbb{P}_r$确实非常集中在低维流形上


在GAN中，$\mathbb{P}_g$的定义是从一个简单先验$z\sim p(z)$进行采样，然后应用一个函数$g:\mathcal{Z}\rightarrow \mathcal{X}$，所以$\mathbb{P}_g$的支撑集被包含在$g(\mathcal{Z})$里面。如果$\mathcal{Z}$的维度小于$\mathcal{X}$的维度（通常都是这样，采样128维，然后生成28x28的图片），那么是不可能让$\mathbb{P}_g$变成连续的 。这是因为在大多数情况下$g(\mathcal{Z})$会被包含在一个低维度流形的联合体上，因此在$\mathcal{X}$中有测度0存在。而直观上，这是高度非平凡的，因为一个$n$维的参数绝对不会意味着图片会处于$n$维流形上。事实上，有许多简单的反例，如Peano曲线，lemniscates等等。

>**$\downarrow$对应白话文$\downarrow$**
- 如果两个分布完全没有重叠的部分，或者它们重叠的部分可忽略（下面解释什么叫可忽略），它们的**JS散度是$\log 2$**(见[附录](#Appendix_1))
    - **总之**：无论$P_r$跟$P_g$是远在天边，还是近在眼前，只要它们俩没有一点重叠或者重叠部分可忽略，JS散度就固定是常数$\log 2$，**而这对于梯度下降方法意味着——梯度为0**此时对于最优判别器来说，生成器肯定是得不到任何梯度信息的；即使对于接近最优的判别器来说，生成器也有很大机会面临梯度消失的问题。

- $P_r$与$P_g$不重叠或重叠部分可忽略的可能性有多大？
    - 不严谨的答案是：非常大。
    - 比较严谨的答案是：当$P_r$与$P_g$的支撑集是高维空间中的低维流形时，$P_r$与$P_g$重叠部分测度为0的概率为1。
- 解释“**当$P_r$与$P_g$的支撑集是高维空间中的低维流形**”
    GAN中的生成器一般是从某个低维(比如100维)的随机分布中采样出一个编码向量，==>再经过一个神经网络生成出一个高维样本(比如64x64的图片就有4096维)。当生成器的参数固定时，生成样本的概率分布虽然是定义在4096维的空间上，但它本身所有可能产生的变化已经被那个100维的随机分布限定了，其本质维度就是100，再考虑到神经网络带来的映射降维，最终可能比100还小，所以生成样本分布的支撑集就在4096维空间中构成一个最多100维的低维流形，“撑不满”整个高维空间。
- “撑不满”就会导致真实分布与生成分布难以“碰到面”，这很容易在二维空间中理解：一方面，二维平面中随机取两条曲线，它们之间刚好存在重叠线段的概率为0；另一方面，虽然它们很大可能会存在交叉点，但是相比于两条曲线而言，交叉点比曲线低一个维度，长度（测度）为0，可忽略。三维空间中也是类似的，随机取两个曲面，它们之间最多就是比较有可能存在交叉线，但是交叉线比曲面低一个维度，面积（测度）是0，可忽略。从低维空间拓展到高维空间，就有了如下逻辑：因为一开始生成器随机初始化，所以$P_g$几乎不可能与$P_r$有什么关联,它们的支撑集之间的重叠部分
    - **要么不存在**
    - **要么就比$P_r$和$P_g$的最小维度还要低至少一个维度，故而测度为0**。所谓“重叠部分测度为0”，就是上文所言“不重叠或者重叠部分可忽略”的意思。
- **关于生成器梯度消失的第一个论证**：在（近似）最优判别器下，最小化生成器的loss等价于最小化$P_r$与$P_g$之间的JS散度，而由于$P_r$与$P_g$几乎不可能有不可忽略的重叠，所以无论它们相距多远JS散度都是常数$\log 2$，最终导致生成器的梯度（近似）为0，梯度消失。
> **$\uparrow$对应白话文$\uparrow$**



**假设$g$是一个NN**。下面给出引理：
- **引理1**：令$g:\mathcal(Z)\rightarrow \mathcal{X}$是一个函数，其由仿射映射(线性变换+平移)，逐点非线性（可以是ReLU，LReLU，或者就是严格平滑增加的函数如sigmoid，tanh，softplus等等）。然后$g(\mathcal{Z})$是包含在一个维度接近$\mathcal{Z}$的可数流形并集上（这里就有点测度论的”可数无穷个不相交的集合的测度之和等于其集合并集的测度“意思了。）。因此，如果$\mathcal{Z}$的维度小于$\mathcal{X}$，那么$g(\mathcal{Z})$会让$\mathcal{X}$中有许多测度为0的存在。
**证明**：附录A
    - **Relu或Leak Relu**： $\sigma(x)=\mathbb{1}[x<0] c_{1} x+\mathbb{1}[x \geq 0] c_{2} x$其中$c_1, c_2 \in \mathbb{R}$
    $g(z)=\mathbf{D}_{n} \mathbf{W}_{n} \ldots \mathbf{D}_{1} \mathbf{W}_{1} z,$ 其中$\mathbf{W}_{i}$仿射变换, $\mathbf{D}_{i}$为取决于$z$对角矩阵，具有对角项$c_1, c_2$。如果我们认为$\mathcal{D}$是所有(有限)集合拥有$c_1, c_2$的对角矩阵，那么$g(\mathcal{Z}) \subseteq \bigcup_{D_{i} \in \mathcal{D}} \mathbf{D}_{n} \mathbf{W}_{n} \ldots \mathbf{D}_{1} \mathbf{W}_{1} \mathcal{Z}$是线性整流的有限集合
    - **严格平滑增加的函数**：$\sigma$是逐点平滑严格单调增加的欸线性时，然后将其矢量化应用到他的图像上是一个微分同胚。因此，它将d维可数集合(countable union)的流形映射到另一个d维可数集合的流形。如果可以证明仿射变换具有相同的意义，那么将证毕。由于$g(\mathcal{Z}) $只是将其应用于$\mathcal{Z}$维的流形。当然，这足以证明放射变化将不增加维度的使流形映射到可数集合的流形，因为可数个可数集的并是可数集。此外，只需要对线性变换展现这一点，因为应用偏置项使微分同构
    - 设$\mathbf{W} \in \mathbb{R}^{n \times m}$，通过奇异值分解$\mathbf{W}=\mathbf{U} \boldsymbol{\Sigma} \mathbf{V}$，其中$\sum$为具有对角正项的平方对角矩阵，$U,V$是由基础变化、包含（意味着在新坐标上加0）和投影到坐标子集的组成。与基数相乘并乘以是微分同构，并且将0加到新坐标上是流形嵌入，因此我们只需要证明对投影到该坐标子集上的陈述即可。$\pi: \mathbb{R}^{n+k} \rightarrow \mathbb{R}^{n}$， 其中$\pi\left(x_{1}, \ldots, x_{n+k}\right)=\left(x_{1}, \ldots, x_{n}\right)$是投影。$\mathcal{M} \subseteq \mathbb{R}^{n+k}$为$d$维流形。
        - 如果$n \leq d$得证，因为映射$\pi$包含在所有的$\mathbb{R}^n$,其中流形最大为d维
        - 如果$n > d$,$\pi_{i}(x)=x_{i}$为第$i$个坐标上的投影，如果$x$为$\pi$的临界点，由于$\pi$的最表是独立的，所以$x$是$\pi_i$的临界点。由于Morse引理，$\pi_i$的临界点被隔离了，因此$\pi$中的任意一个也被隔离，意味着它们的数量最多的可数数字。因此$\pi$将非临界点映射到了d维流形(因为它扮演者嵌入)，并且将可数的边界点，映射到可计数的点数(或0维流形)


**如果$\mathbb{P}_r$和$\mathbb{P}_g$的支撑集是不相交或者位于低维流形上，那么总是存在一个完美的判别器**

## 完美的判别理论
>**$\downarrow$脉络白话文$\downarrow$**
- 首先，$\mathbb{P}_r$和$\mathbb{P}_g$之间几乎不可能有不可忽略的重叠，所以无论它们之间的“缝隙”多狭小，都肯定存在一个最优分割曲面把它们隔开，最多就是在那些可忽略的重叠处隔不开而已。
- 由于判别器作为一个神经网络可以无限拟合这个分隔曲面，所以存在一个最优判别器，对几乎所有真实样本给出概率1，对几乎所有生成样本给出概率0，而那些隔不开的部分就是难以被最优判别器分类的样本，但是它们的测度为0，可忽略。
- 最优判别器在真实分布和生成分布的支撑集上给出的概率都是常数（1和0），导致生成器的loss梯度为0，梯度消失。
> **$\uparrow$脉络白话文$\uparrow$**
- 解释下什么时候$\mathbb{P}_r$和$\mathbb{P}_g$会存在不相交支撑集
我们说一个判别器$D:\mathcal{X}\rightarrow [0,1]$的准确值为1时，即此时判别器在包含$\mathbb{P}_r$的支撑集上判定其为1，而在包含$\mathbb{P}_g$的支撑集上判定其为0.即$\mathbb{P}_r[D(x)=1]=1$和$\mathbb{P}_g[D(x)=0]=1$。

- **定理2.1**：如果两个分布$\mathbb{P}_r$和$\mathbb{P}_g$的支撑集分别包含在两个不相交且紧凑的集合$\mathcal{M}$和$\mathcal{P}$，那么存在一个平滑最优的判别器$D^*:\mathcal{X}\rightarrow [0,1]$，其准确值为1（即此时一定存在一个判别器能够完全划分这两个集合），且此时对于所有的$x\in\mathcal{M}\cup\mathcal{P}$有$\bigtriangledown_x D^*(x)=0$
**证明**：判别器是训练并且最大化
$$\mathbb{E}_{x\sim\mathbb{P}_r}[\log D(x)]+\mathbb{E}_{x\sim\mathbb{P}_g}[\log(1-D(x))]$$
因为$\mathcal{M}$和$\mathcal{P}$都是紧凑且不相关的，那么$z$这两个集合之间的距离存在$0<\delta =d(\mathcal{P},\mathcal{M})$。现在定义：
$$\begin{aligned}
\hat{\mathcal{M}} = \{ x:d(x,M) \leq \frac{\delta}{3} \} \\ \hat{\mathcal{P}}= \{x:d(x,P)\leq \frac{\delta}{3}\} 
\end{aligned}$$
通过$\delta$的定义，$\hat{\mathcal{M}}$和$\hat{\mathcal{P}}$清晰的是两个不相关紧凑集合。因此，通过Urysohn的平滑理论，会存在一个平滑函数$D^*:\mathcal{X}\rightarrow [0,1]$,使得$D^*|_{\hat{\mathcal{M}}}\equiv 1$和$D^*|_{\hat{\mathcal{P}}}\equiv 0$。因为对于所有的位于$\mathcal{P}_r$的支撑集中的变量$x$，都有$\log D^*(x)=0$，而对于所有位于$\mathcal{P}_g$的支撑集中的变量$x$，都有$\log (1-D^*(x))=1$，判别器是完全最优且准确值为1。令$x$位于$\mathcal{M}\cup \mathcal{P}$。假设$x\in\mathcal{M}$，存在一个开区间球$B=B(x,\frac{\delta}{3})$，且$D^*|_{B}$是一个常量。此时$\triangledown_xD^*(x) \equiv 0$，即梯度就是为0，如果$x\in\mathcal{P}$那么结果也是一样。得证。

在下一个理论中，先放弃不相交的假设，将其推广到更一般的情况，
假设是2个不同的流形。如果这两个流形在很大部分空间上都是完美匹配的，那么意味着没有判别器可以将它们进行区分。直观上，具有这种情况的两个低维度流形还是很少存在的：对于在特定段中空间匹配的两条曲线，它们不能以遭受任何任意的小方式扰动下还能满足该属性。（即在低维流形中进行稍微扰动下，就分开了）。为此，将**定义**两个流形完美对齐的概念，并表明在任意的小扰动下，该属性永远不会以概率1保持。（即一扰动，该属性就会被破坏）.

- **定义2.1**：我们首先需要回顾一下横向性(transversality)的定义。令$\mathcal{M}$和$\mathcal{P}$是两个关于$\mathcal{F}$的无边界常规子流形，这里简单认为$\mathcal{F}=\mathbb[R]^d$。$x\in \mathcal{M}\cap\mathcal{P}$是这两个流形的交叉点。如果有$\mathcal{T}_x\mathcal{M}+\mathcal{T}_x\mathcal{P}=\mathcal{T}_x\mathcal{F}$，我们就说$\mathcal{M}$和$\mathcal{P}$在$x$上横向交叉，这里$\mathcal{T}_x\mathcal{M}$表示$\mathcal{M}$上围绕$x$的切线空间。

- **定义2.2**我们说两个没有边界的流形$\mathcal{M}$和$\mathcal{P}$是完美对齐是，如果有$x\in \mathcal{M}\cap\mathcal{P}$，那么$\mathcal{M}$和$\mathcal{P}$不在$x$上横向交叉。

这里将流形$M$的边界表示为$\partial M$，内部表示为$Int M$。我们说两个流形$\mathcal{M}$和$\mathcal{P}$（不管是否有边界）完美对齐是基于下面四组中（$Int \mathcal{M}$, $Int \mathcal{P}$）,（$Int \mathcal{M}$,$\partial \mathcal{P}$），（$\partial \mathcal{M}$, $Int \mathcal{P}$），（$\partial \mathcal{M}$,$\partial \mathcal{P}$）任意一组无边界流形对完全对齐成立前提下。[^yinli2]

[^yinli2]:有趣的是，在实际中，我们可以很安全的假设任意两个流形不是完美对齐的，因为在这两个流形上任意一个小的扰动都会导致他们有横向交叉或者甚至不交叉。这可以通过引理2进行叙述和证明。

如引理3所述，如果两个流形不完美对齐，那么他们的交集$\mathcal{L}=\mathcal{M}\cap\mathcal{P}$是一个有限流形的并集，其中维度严格小于$\mathcal{M}$和$\mathcal{P}$。

- **引理2**：令$\mathcal{M}$和$\mathcal{P}$是$\mathbb{R}^d$的两个常规子流形，且没有所有的维度。同时令$\eta$,$\eta^{'}$是任意独立的连续随机变量。因此定义扰动的流形为$\tilde{\mathcal{M}}=\mathcal{M}+\eta$和$\tilde{\mathcal{P}}=\mathcal{P}+\eta^{'}$，那么：
$$\mathbb{P}_{\eta,\eta^{'}}(\tilde{\mathcal{M}}与\tilde{\mathcal{P}}不是完美对齐)=1$$
**证明**见附录A

- **引理3**：令$\mathcal{M}$和$\mathcal{P}$是$\mathbb{R}^d$的两个常规子流形，他们不是完美对齐且没有所有维度。令$\mathcal{L}=\mathcal{M}\cap\mathcal{P}$。如果$\mathcal{M}$和$\mathcal{P}$没有边界，那么$\mathcal{L}$同样也是一个流形，并且维度严格低于$\mathcal{M}$和$\mathcal{P}$。如果他们有边界，那么$\mathcal{L}$是一个最多4个（可数的）严格更低维度流形的并集。在这两种情况中，$\mathcal{L}$在$\mathcal{M}$和$\mathcal{P}$上的测度为0.
**证明**见附录A

现在叙述下在这种情况下，基于两个流形上最优判别器结果。

- **定理2.2**：令$\mathbb{P}_r$和$\mathbb{P}_g$是两个分布，其支撑集包含在两个封闭的流形$\mathcal{M}$和$\mathcal{P}$上，这两个流形没有完美对齐，且没有所有维度。并假设$\mathbb{P}_r$和$\mathbb{P}_g$在他们各自流形内是连续的，意味着如果有一个集合$A$，其在$\mathcal{M}$上的测度为0，那么$\mathbb{P}_r(A)=0$（对于$\mathbb{P}_g$也是一样）。然后，存在一个最优判别器$D^*:\mathcal{X}\rightarrow [0,1]$的准确度为1，且对于几乎任意$\mathcal{M}$和$\mathcal{P}$中的变量$x$，$D^*$在$x$周边是平滑的，且$\bigtriangledown_x D^*(x)=0$。
**证明**：通过**引理3**我们知道$\mathcal{L}=\mathcal{M} \cap \mathcal{P}$是比$\mathcal{M}$和$\mathcal{P}$严格更低维度,并在两个位置上都测得0。通过连续性，得知$\mathbb{P}_{r}(\mathcal{L})=0$和$\mathbb{P}_{g}(\mathcal{L})=0$。注意这暗示着$\mathbb{P}_{r}$ 暗含在 $\mathcal{M} \backslash \mathcal{L}$中，同时也是$\mathbb{P}_{q}$ 暗含在 $\mathcal{P} \backslash \mathcal{L}$的支持。
令$x \in \mathcal{M} \backslash \mathcal{L} $因此$ x \in \mathcal{P}^{c} \left( \mathcal{P}\text {的补码}\right) $ 是一个开放集(open set), 因此存在一个半径为 $\epsilon_{x}$的球，使得$B\left(x, \epsilon_{x}\right) \cap \mathcal{P}=\emptyset$通过这种方式，我们定义了
$$
\hat{\mathcal{M}}=\bigcup_{x \in \mathcal{M} \backslash \mathcal{L}} B\left(x, \epsilon_{x} / 3\right)
$$
类似的定义$\hat{P}$. 注意，通过构造，这些都是在$\mathbb{R}^{d}$上的开放集 因此$\mathcal{M} \backslash \mathcal{L} \subseteq$
$\hat{\mathcal{M}},$ and $\mathcal{P} \backslash \mathcal{L} \subseteq \hat{\mathcal{P}},$ the support of $\mathbb{P}_{r}$ and $\mathbb{P}_{g}$ is contained in $\mathcal{M}$ and $\hat{\mathcal{P}}$ respectively. As well by construction, $\mathcal{M} \cap \hat{\mathcal{P}}=\emptyset$
Let us define $D^{*}(x)=1$ for all $x \in \hat{\mathcal{M}},$ and 0 elsewhere (clearly including $\hat{\mathcal{P}} .$ since $\log D^{*}(x)=0$ for all $x$ in the support of $\mathbb{P}_{r}$ and $\log \left(1-D^{*}(x)\right)=0$ for all $x$ in the support of $\mathbb{P}_{g},$ the discriminator is completely optimal and has accuracy 1. Furthermore, let $x \in \hat{\mathcal{M}}$. since $\mathcal{M}$ is an open set and $D^{*}$ is constant on $\mathcal{M},$ then $\left.\nabla_{x} D^{*}\right|_{\hat{\mathcal{M}}} \equiv 0 .$ Analogously, $\nabla_{x} D^{*} | \hat{p} \equiv 0 .$ Therefore, the set of points where $D^{*}$ is non-smooth or has non-zero gradient inside $\mathcal{M} \cup \mathcal{P}$ is contained in $\mathcal{L},$ which has null-measure in both manifolds, therefore concluding the theorem.

这两个定理告诉我们**存在一个最优判别器，其在$\mathbb{P}_r$和$\mathbb{P}_g$几乎任何地方都是平滑而且是常量**。所以事实就是该判别器在流形点上是常量，所以没法通过BP学到任何信息，同时在下面介绍的也是常量。下面的定理2.3是将整个理论进行总结得出的

- **定理2.3**：令$\mathbb{P}_r$和$\mathbb{P}_g$是两个分布，其支撑集包含在两个封闭的流形$\mathcal{M}$和$\mathcal{P}$上，这两个流形没有完美对齐，且没有所有维度。并假设$\mathbb{P}_r$和$\mathbb{P}_g$在他们各自流形内是连续的，那么：
$$ \begin{aligned}
JSD(\mathbb{P}_r||\mathbb{P}_g) =\log2 \\ KL(\mathbb{P}_r||\mathbb{P}_g) = +\infty \\ KL(\mathbb{P}_g||\mathbb{P}_r) = +\infty 
\end{aligned}
$$

注意到即使两个流形彼此靠得很近，这些散度也会maxed out。而就算生成器生成的样本看上去很好，可是此时两个KL散度可能很大。因此，定理2.3指出使用那些通常用来测试两个分布之间相似性的方法并不是一个好主意。更不用说，如果这些散度总是maxed out并试图通过梯度下降进行最小化也是不可能的。我们期望有一个softer的测度，可以包含流形中点之间距离的概念。我们将在第3节中稍后再讨论该主题，在该部分中我们将解释一个替代指标并提供我们可以分析和优化的范围。


>**$\downarrow$结论白话文$\downarrow$**
有了这些理论分析，原始GAN不稳定的原因就彻底清楚了：
- **判别器训练得太好，生成器梯度消失，生成器loss降不下去**
- **判别器训练得不好，生成器梯度不准，四处乱跑**
- **只有判别器训练得不好不坏才行，但是这个火候又很难把握，甚至在同一轮训练的前后不同阶段这个火候都可能不一样**
所以GAN才那么难训练。
> **$\uparrow$结论白话文$\uparrow$**


## 每个cost函数的结果和问题
**上述均为原始cost**
定理2.1和定理2.2得出一个很重要的事实。如果我们关心的两个分布的支撑集是不相关或者位于低维流形上的，最优判别器可能是完美的，而且梯度几乎在任何地方都是0.
### 原始的cost函数
**一句话概括：判别器越好，生成器梯度消失越严重**
接下来将介绍下当通过一个判别器将梯度传递给生成器时会发生什么。与目前为止的典型分析一个关键区别是，作者将开发一套理论来近似最优判别器，而不是使用（未知）真正的判别器。并证明随着近似越来越好，所观察到的梯度消失或者大规模不稳定的行为主要依赖使用的cost函数。
将$||D||$表示范数：
$$||D||=\underset{x\in\mathcal{X}}{sup}|D(x)|+||\bigtriangledown_x D(x)||_2$$
该范数的使用可以让证明变得更简单，但是可以在另一个Sobolev范数中完成$||\cdot||_{1,p}$,对于普遍逼近定理所涵盖的$p<\infty$，此时可以保证在这个范数中的神经网络近似[5]。

- **定理2.4**（在生成器上的梯度消失）:$g_{\theta}:\mathcal{Z}\rightarrow \mathcal{X}$是一个微分函数，表示分布一个分布$\mathbb{P}_g$。令$\mathbb{P}_r$表示真实数据分布，D表示一个可微分的判别器。如果定理2.1和2.2都满足，$||D-D^*||<\epsilon$且$\mathbb{E}_{z\sim p(z)}\left[ ||J_{\theta}g_{\theta}(z)||_2^2\right]\leq M^2$（因为M依赖于$\theta$,对于均匀分布先验和NN，该条件可以简单验证。而对于高斯先验需要更多的工作，因为我们需要限制$z$的增长，但是对当前结构同样适用。），那么：
$$||\bigtriangledown_{\theta}\mathbb{E}_{z\sim p(z)}\left[\log(1-D(g_{\theta}(z))\,\,) \right]||_2<M\frac{\epsilon}{1-\epsilon}$$
**证明**：在定理2.1和定理2.2的证明中，$D^*$在$\mathbb{P}_g$的支撑集上局部为0.那么在该支撑集上使用Jensen不等式和链式法则，得：
$$
\begin{aligned}
\left\|\nabla_{\theta} \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D\left(g_{\theta}(z)\right)\right)\right)\right\|_{2}^{2} & \leq \mathbb{E}_{z \sim p(z)}\left[\frac{\left\|\nabla_{\theta} D\left(g_{\theta}(z)\right)\right\|_{2}^{2}}{\left|1-D\left(g_{\theta}(z)\right)\right|^{2}}\right] \\
& \leq \mathbb{E}_{z \sim p(z)}\left[\frac{\left\|\nabla_{x} D\left(g_{\theta}(z)\right)\right\|_{2}^{2}\left\|J_{\theta} g_{\theta}(z)\right\|_{2}^{2}}{\left|1-D\left(g_{\theta}(z)\right)\right|^{2}}\right] \\
&<\mathbb{E}_{z \sim p(z)}\left[\frac{\left(\left\|\nabla_{x} D^{*}\left(g_{\theta}(z)\right)\right\|_{2}+\epsilon\right)^{2}\left\|J_{\theta} g_{\theta}(z)\right\|_{2}^{2}}{\left.\left(| 1-D^{*}(z)\right) |-\epsilon\right)^{2}}\right] \\
&=\mathbb{E}_{z \sim p(z)}\left[\frac{\epsilon^{2}\left\|J_{\theta} g_{\theta}(z)\right\|_{2}^{2}}{(1-\epsilon)^{2}}\right] \\
& \leq M^{2} \frac{\epsilon^{2}}{(1-\epsilon)^{2}}
\end{aligned}
$$将其开方得：$$||\bigtriangledown_{\theta}\mathbb{E}_{z\sim p(z)}\left[\log(1-D(g_{\theta}(z))\,\,)\right]||_2<M\frac{\epsilon}{1-\epsilon}$$
得证。

- **推理2.1**：基于与定理2.4一样的假设：
$$ \underset{||D-D^*||\rightarrow 0}{\lim} \bigtriangledown_{\theta}\mathbb{E}_{z\sim p(z)}\left[\log(1-D(g_{\theta}(z))\,\,)\right]=0 $$

可以发现判别器训练的越好，则生成器梯度就会消失，生成器的cost函数接近Jensen-Shannon散度取决于近似的质量好坏。这点告诉我们一个基础理论：要么判别器的更新是不准确的或者直接梯度会消失。这使得训练这个cost变得很困难或者需要用户来决定准确的判别器训练过程，从而让GAN训练变得很困难。

### 生成器的代替函数-log D（the -log D alternative）
**一句话概括：最小化第二种生成器loss函数，会等价于最小化一个不合理的距离衡量，导致两个问题，一是梯度不稳定，二是collapse mode即多样性不足。**
为了避免判别器很好时候梯度消失的问题，人们选择使用一个不同的cost函数：
$$ \Delta \theta=\bigtriangledown_{\theta}\mathbb{E}_{z\sim p(z)}[-\log D(g_{\theta}(z))] $$
现在先叙述并证明该梯度优化的cost函数，随后，证明虽然该梯度不一定会受到消失梯度的影响，但它确实会在最优判别器的噪音近似下导致大量不稳定的更新（在实验中已经广泛证实）。

- **定理2.5**:令$\mathbb{P}_r$和$\mathbb{P}_{g \theta}$表示2个连续分布，对应密度为$P_r$和$P_{g \theta}$。令$D^*=\frac{P_r}{P_{g\theta_0}+P_r}$是最优判别器，此时$\theta_0$是固定的（迭代生成器的时候，判别器是固定的）。那么：
$$ \mathbb{E}_{z\sim p(z)}\left[ -\bigtriangledown_{\theta}\log D^*(g_{\theta}(z))|_{\theta=\theta_0}\right]=\bigtriangledown_{\theta}\left[ KL(\mathbb{P}_{g\theta}||\mathbb{P}_r)-2JSD(\mathbb{P}_{g\theta}||\mathbb{P}_r)\right] |_{\theta=\theta_0} $$
或：
$$Loss_{ G }=KL({ P }_{ g }(x)||{ P }_{ r }(x))-2JS({ P }_{ r }||{ P }_{ g })+{P}_{r}(x)*log[D^*(x)]+2log2[1-D^{ * }(x)]$$(证明见GAN论文)
**证明**：原始生成器的loss为改成
$$\mathbb{E}_{x\sim P_g}[-\log D(x)]$$
且在得到最优判别器下
$$\mathbb{E}_{x\sim P_r}[\log D^*(x)]+\mathbb{E}_{x\sim P_g}[\log(1- D^*(x))]=2JSD(P_r||P_g)-2\log 2$$
将KL散度变换成最优判别器：$$
\begin{aligned}
K L\left(\mathbb{P}_{g_{\theta}} \| \mathbb{P}_{r}\right) &=\mathbb{E}_{x \sim \mathbb{P}_{g_{\theta}}}\left[\log \frac{P_{g_{\theta}}(x)}{P_{r}(x)}\right] \\
&=\mathbb{E}_{x \sim \mathbb{P}_{g_{\theta}}}\left[\log \frac{P_{g_{0}(x)}(x)}{P_{r}(x)}\right]-\mathbb{E}_{x \sim \mathbb{P}_{g_{\theta}}}\left[\log \frac{P_{g_{\theta}}(x)}{P_{g_{0}}(x)}\right] \\
&=-\mathbb{E}_{x \sim \mathbb{P}_{g_{\theta}}}\left[\log \frac{D^{*}(x)}{1-D^{*}(x)}\right]-K L\left(\mathbb{P}_{g_{\theta}} \| \mathbb{P}_{g_{\theta_{0}}}\right) \\
&=-\mathbb{E}_{z \sim p(z)}\left[\log \frac{D^{*}\left(g_{\theta}(z)\right)}{1-D^{*}\left(g_{\theta}(z)\right)}\right]-K L\left(\mathbb{P}_{g_{\theta}} \| \mathbb{P}_{g_{\theta_{0}}}\right)
\end{aligned}
$$Taking derivatives in $\theta$ at $\theta_0$ we get$$
\begin{aligned}
\left.\nabla_{\theta} K L\left(\mathbb{P}_{g s} \| \mathbb{P}_{r}\right)\right|_{\theta=\theta_{0}} &=-\left.\nabla_{\theta} \mathbb{E}_{z \sim p(z)}\left[\log \frac{D^{*}\left(g_{\theta}(z)\right)}{1-D^{*}\left(g_{\theta}(z)\right)}\right]\right|_{\theta=\theta_{0}}-\left.\nabla_{\theta} K L\left(\mathbb{P}_{g_{\theta}} \| \mathbb{P}_{g e_{0}}\right)\right|_{\theta=\theta_{0}} \\
&=\left.\mathbb{E}_{z \sim p(z)}\left[-\nabla_{\theta} \log \frac{D^{*}\left(g_{\theta}(z)\right)}{1-D^{*}\left(g_{\theta}(z)\right)}\right]\right|_{\theta=\theta_{0}}
\end{aligned}
$$将最后一个方程式与JSD的结果相减，我们便获得了所需的结果
得证。

>**$\downarrow$问题分析$\downarrow$**
- 第一：JS散度项
    一个拉近，一个推远。在数值上则会导致梯度不稳定
    - 最小化生成分布与真实分布的KL散度
    - 却又要最大化两者的JSD散度
- 第二，$KL(\mathbb{P}_{g\theta}||\mathbb{P}_r)$
    KL散度不是一个对称的衡量
    $KL(\mathbb{P}_{g\theta}||\mathbb{P}_r)$与$KL(\mathbb{P}_r||\mathbb{P}_g)$是不同的，以前者为例
    - $KL(\mathbb{P}_{g\theta}||\mathbb{P}_r)$
        这一放一打之下，生成器宁可多生成一些重复但是很“安全”的样本，也不愿意去生成多样性的样本，因为那样一不小心就会产生第二种错误，得不偿失。这种现象就是大家常说的collapse mode。
        - 当 $P_g(x)\rightarrow 0$而$P_r(x)\rightarrow 0$时，$P_g(x)\log\frac{P_g(x)}{P_r(x)}\rightarrow 0$,对$KL(\mathbb{P}_{g\theta}||\mathbb{P}_r)$贡献趋近于0；**“生成器没能生成真实的样本”，惩罚微小**；**缺乏多样性**
        - 当 $P_g(x)\rightarrow 1$而$P_r(x)\rightarrow 0$时，$P_g(x)\log\frac{P_g(x)}{P_r(x)}\rightarrow +\infty$,对$KL(\mathbb{P}_{g\theta}||\mathbb{P}_r)$贡献趋近于正无穷；**“生成器生成了不真实的样本” ，惩罚巨大** **缺乏准确性**
    - $KL(\mathbb{P}_r||\mathbb{P}_g)$
> **$\uparrow$问题分析$\uparrow$**

- **定理2.6**(生成器梯度更新的不稳定):令$g_{\theta}: \mathcal{Z} \rightarrow \mathcal{X}$是可生成$\mathbb{P}_g$的可微函数。令$\mathbb{P}_r$是真实数据分布，同时满足定理2.1和定理2.2的条件。令$D$是一个判别器，使得$D^*-D=\epsilon$是一个由x索引为中心的高斯过程，并且对于每一个x(通常称为白噪声)为独立。并且$\nabla_{x} D^{*}-\nabla_{x} D=r$是另一个由x索引为中心的独立高斯过程，且对于每一个x为独立。然后，每个坐标$$
\mathbb{E}_{z \sim p(z)}\left[-\nabla_{\theta} \log D\left(g_{\theta}(z)\right)\right]
$$是具有无限期望和方差的中心柯西分布(**Note**定理成立，与r和$\epsilon$的方差无关。 随着逼近度的提高，由于有限的精度，该误差越来越像是中心随机噪声。)
**证明**: 在这种情况下，$\mathbb{P}_g$的支撑集上$D$的局部常数等于0.使用$r\left(g_{\theta}(z)\right), \epsilon\left(g_{\theta}(z)\right)$ 表示随机变量$r(z), \epsilon(z)$。使用链式规则和$r,\epsilon$的定义，得到
$$
\begin{aligned}
\mathbb{E}_{z \sim p(z)}\left[-\nabla_{\theta} \log D\left(g_{\theta}(z)\right)\right] &=\mathbb{E}_{z \sim p(z)}\left[-\frac{J_{\theta g_{\theta}(z)} \nabla_{x} D\left(g_{\theta}(z)\right)}{D\left(g_{\theta}(z)\right)}\right] \\
&=\mathbb{E}_{z \sim p(z)}\left[-\frac{J_{\theta} g_{\theta}(z) r(z)}{\epsilon(z)}\right]
\end{aligned}
$$由于$r(z)$是中心高斯分布，因此它乘一个矩阵不会改变。此外，当除以$\epsilon$(独立于分子的中心高斯),在每个坐标上得到一个中心柯西随机变量.z上取平均值，不同的独立柯西随机变量再次产生中心柯西分布
证毕。

**Note**即使我们忽略了更新会有无限的变化（即方差很大），仍然认为更新的分布是可以中心化的，这意味着如果我们限定更新，更新的期望将为0，即不向梯度提供任何反馈。
因为关于$D$和$\nabla D$的噪音是去相关的假设太严格了，如图3.在训练稳定良好的DCGAN的任何阶段，除非已经收敛，否则当我们训练鉴别器接近最优时，梯度的范数会急剧增长。在所有情况下，使用此更新会导致样本质量不断下降。曲线中的噪音显示梯度的方差也在不断增长，而这会减缓收敛并且在优化阶段会有更多不稳定的行为

# 更柔和的指标和分布
一个很重要的问题是如何修复不稳定和梯度消失的问题。我们打破这些定理假设的一个方法就是**给判别器输入增加连续噪音**，因而平滑概率质量的分布。

- **定理3.1**：如果$X$分布为$\mathbb{P}_X$，其支撑集为$\mathcal{M}$,$\epsilon$是一个完全连续的随机变量，其密度为$P_{\epsilon}$，那么$\mathbb{P}_{X+\epsilon}$是完全连续的，其密度为：
$$ \begin{aligned}P_{X+\epsilon}(x) &= \mathbb{E}_{y\sim \mathbb{P}_X}[P_{\epsilon}(x-y)] \\ &= \int_{\mathcal{M}}P_{\epsilon}(x-y)d\mathbb{P}_X(y) \end{aligned}$$
**证明**:
    - 首先证明$\mathbb{P}_{X+\epsilon}$是绝对连续的，令$A$是Lebesgue测度为0的Borel集。那么，根据$\epsilon$和$X$是独立的事实，通过Fubini可以得到
    $$
    \begin{aligned}
    \mathbb{P}_{X+\epsilon}(A) &=\int_{\mathbb{R}^{d}} \mathbb{P}_{\epsilon}(A-x) \mathrm{d} \mathbb{P}_{X}(x) \\
    &=\int_{\mathbb{R}^{d}} 0 \mathrm{d} \mathbb{P}_{X}(x)=0
    \end{aligned}
    $$
    其中，如果$A$的Lebesgue的测度为0，那么$A-x$的测度也为0.并且由于$\mathbb{P}_{\epsilon}$是绝对连续的，所以$\mathbb{P}_{\epsilon}(A-x)=0$
    - 接着计算$\mathbb{P}_{X+\epsilon}$的密度。继续使用$X$和$\epsilon$独立性，对于任何Borel集B，有
    $$
    \begin{aligned}
    \mathbb{P}_{X+\epsilon}(B) &=\int_{\mathbb{R}^{d}} \mathbb{P}_{\epsilon}(B-y) \mathrm{d} \mathbb{P}_{X}(y) \\
    &=\mathbb{E}_{y \sim \mathbb{P}} x\left[\mathbb{P}_{\epsilon}(B-y)\right] \\
    &=\mathbb{E}_{y \sim \mathbb{P}_{x}}\left[\int_{B-y} P_{\epsilon}(x) d x\right] \\
    &=\left.\mathbb{E}_{y}\right|_{\Gamma_{x}}\left[\int_{B} P_{\epsilon}(x-y) d x\right] \\
    &=\int_{B} \mathbb{E}_{y \sim \mathbb{P}_{x}}\left[P_{\epsilon}(x-y)\right] d x
    \end{aligned}
    $$
    因此$\mathbb{P}_{X+\epsilon}(B)=\int_{B} P_{X+\epsilon}(x) d x$我们的目标是$P_{X+\epsilon}$和所有的Borel集$B$，通过Radon-Nikodym定理的唯一性，这意味着，所提出的$P_{x+\epsilon}$是$\mathbb{P}_{X+\epsilon}$的密度。根据期望定义以及$\mathbb{P}_X$在$\mathcal{M}$的支撑集$\mathbb{P}_X$, 改变$\int_{\mathcal{M}} \mathbb{P}_{X}$期望的公式的等价性是微不足道的。
    证毕。

- **推论3.1**
    - 如果$\epsilon\sim \mathcal{N}(0,\sigma^2I)$，那么：
    $$P_{X+\epsilon}(x)=\frac{1}{Z}\int_{\mathcal{M}}e^{-\frac{||y-x||^2}{2\sigma^2}}d\mathbb{P}_{X}(y)$$
    - 如果$\epsilon \sim \mathcal{N}(0, \Sigma)$，那么：
    $$
    P_{X+\epsilon}(x)=\frac{1}{Z} \mathbb{E}_{y \sim \mathbb{P}_{X}}\left[e^{-\frac{1}{2}\|y-x\|_{\Sigma-1}^{2}}\right]$$
    - 如果$P_{\epsilon}(x) \propto \frac{1}{\|x\|^{d+1}}$，那么：
    $$
    P_{X+\epsilon}(x)=\frac{1}{Z} \mathbb{E}_{y \sim \mathbb{P}_{X}}\left[\frac{1}{\|x-y\|^{d+1}}\right]
    $$

从定理得知密度$P_{X+\epsilon}(X)$与到支撑集$\mathbb{P}_X$的平均距离成反比，由临界点的概率加权。在支撑集$\mathbb{P}_X$是流形的情况下。我们将得到到沿着流形的点的距离的加权平均值。我们如何选择噪声的分布将影响我们所选择的距离的概念。例如，在我们的corolary中，我们可以看到通过改变指数内的范数来改变协方差矩阵的效果。因此，可以使用具有不同衰变类型的不同噪声。

因此，**最佳判别器处于$\mathbb{P}_{g+\epsilon}$和$\mathbb{P}_{r + \epsilon}$之间**
$$
D^{*}(x)=\frac{P_{r+\epsilon}(x)}{P_{r+\epsilon}(x)+P_{g+\epsilon}(x)}
$$

- **定理3.2**:令$\mathbb{P}_{g}$和$\mathbb{P}_{r}$为支撑集在$\mathcal{M}$和$\mathcal{P}$上的分布，其中$\epsilon \sim \mathcal{N}\left(0, \sigma^{2} I\right)$。那么，传递给生成器的梯度具有以下形式
$$
\begin{aligned}
&\mathbb{E}_{z \sim p(z)}\left[\nabla_{\theta} \log \left(1-D^{*}\left(g_{\theta}(z)\right)\right)\right]\\
&=\mathbb{E}_{z \sim p(z)}\left[a(z) \int_{\mathcal{M}} P_{\epsilon}\left(g_{\theta}(z)-y\right) \nabla_{\theta}\left\|g_{\theta}(z)-y\right\|^{2} \mathrm{d} \mathbb{P}_{r}(y)\right.\\
&-b(z) \int_{\mathcal{P}} P_{\epsilon}\left(g_{\theta}(z)-y\right) \nabla_{\theta}\left\|g_{\theta}(z)-y\right\|^{2} \mathrm{d} \mathbb{P}_{g}(y)]
\end{aligned}
$$,其中$a(z)$和$b(z)$是正函数。此外当且仅当$P_{r+\epsilon}>P_{g+\epsilon}$时$b>a$；当且仅当$P_{r+\epsilon} < P_{g+\epsilon}$时$b < a$；
**证明**：由于鉴别器在反向生成到发生器时假定是固定的，因此唯一依赖的是每个z的$g_\theta(z)$ 通过对我们的成本函数取导数
$$
\begin{array}{l}
{\mathbb{E}_{z \sim p(z)}\left[\nabla_{\theta} \log \left(1-D^{*}\left(g_{\theta}(z)\right)\right)\right]} \\
{=\mathbb{E}_{z \sim p(z)}\left[\nabla_{\theta} \log \frac{P_{g+\epsilon}\left(g_{\theta}(z)\right)}{P_{r+\epsilon}\left(g_{\theta}(z)\right)+P_{g+\epsilon}\left(g_{\theta}(z)\right)}\right]} \\
{=\mathbb{E}_{z \sim p(z)}\left[\nabla_{\theta} \log P_{g+\epsilon}\left(g_{\theta}(z)\right)-\nabla_{\theta+\epsilon}\left(g_{\theta}(z)\right)\right]} \\
{=\mathbb{E}_{z \sim p(z)}\left[\frac{\nabla_{\theta} P_{g+\epsilon}\left(g_{\theta}(z)\right)}{P_{g+\epsilon}\left(g_{\theta}(z)\right)}-\frac{\nabla_{\theta} P_{g+\epsilon}\left(g_{\theta}(z)\right)+\nabla_{\theta} P_{r+\epsilon}\left(g_{\theta}(z)\right)}{P_{g+\epsilon}\left(g_{\theta}(z)\right)+P_{r+\epsilon}\left(g_{\theta}(z)\right)}\right]} \\
{=\mathbb{E}_{z \sim p(z)}\left[\frac{1}{P_{g+\epsilon}\left(g_{\theta}(z)\right)+P_{r+\epsilon}\left(g_{\theta}(z)\right)} \nabla_{\theta}\left[-P_{r+\epsilon}\left(g_{\theta}(z)\right)\right]-\right.} \\
{\left.\frac{1}{P_{g+\epsilon}\left(g_{\theta}(z)\right)+P_{r+\epsilon}\left(g_{\theta}(z)\right)} \frac{P_{r+\epsilon}\left(g_{\theta}(z)\right)}{P_{g+\epsilon}\left(g_{\theta}(z)\right)} \nabla_{\theta}\left[-P_{g+\epsilon}\left(g_{\theta}(z)\right)\right]\right]}
\end{array}
$$令$\epsilon$的密度为$\frac{1}{Z} e^{-\frac{\|x\|^{2}}{2 \sigma^{2}}}$。定义$$
\begin{aligned}
&a(z)=\frac{1}{2 \sigma^{2}} \frac{1}{P_{g+\epsilon}\left(g_{\theta}(z)\right)+P_{r+\epsilon}\left(g_{\theta}(z)\right)}\\
&b(z)=\frac{1}{2 \sigma^{2}} \frac{1}{P_{g+\epsilon}\left(g_{\theta}(z)\right)+P_{r+\epsilon}\left(g_{\theta}(z)\right)} \frac{P_{r+\epsilon}\left(g_{\theta}(z)\right)}{P_{g+\epsilon}\left(g_{\theta}(z)\right)}
\end{aligned}
$$同时$a,b$为正函数，由于$b=a \frac{P_{r+\epsilon}}{P_{g+\epsilon}}$，可以得知当且仅当$P_{r+\epsilon}>P_{g+\epsilon}$时$b>a$；当且仅当$P_{r+\epsilon} < P_{g+\epsilon}$时$b < a$；同时得出$$
\begin{aligned}
&\mathbb{E}_{z \sim p(z)}\left[\nabla_{\theta} \log \left(1-D^{*}\left(g_{\theta}(z)\right)\right)\right]\\
&\begin{array}{l}
{=\mathbb{E}_{z \sim p(z)}\left[2 \sigma^{2} a(z) \nabla_{\theta}\left[-P_{r+\epsilon}\left(g_{\theta}(z)\right)\right]-2 \sigma^{2} b(z)\right] \nabla_{\theta}\left[-P_{g+\epsilon}\left(g_{\theta}(z)\right)\right]} \\
{=\mathbb{E}_{z \sim p(z)}\left[2 \sigma^{2} a(z) \int_{\mathcal{M}}-\nabla_{\theta} \frac{1}{Z} e^{\frac{-\| g_{\theta}(z)-y | l_{2}}{2 \sigma^{2}}} \mathrm{d} \mathbb{P}_{r}(y)-2 \sigma^{2} b(z) \int_{\mathcal{P}}-\nabla_{\theta} \frac{1}{Z} e^{\frac{-\| g_{\theta}(z)-y | l_{2}^{2}}{2 \sigma^{2}}} \mathrm{d} \mathbb{P}_{g}(y)\right]} \\
{=\mathbb{E}_{z \sim p(z)}\left[a(z) \int_{\mathcal{M}} \frac{1}{Z} e^{\frac{-\| g_{\theta}(z)-y | 2}{2 \sigma^{2}}} \nabla_{\theta}\left\|g_{\theta}(z)-y\right\|^{2} \mathrm{d} \mathbb{P}_{r}(y)\right.}
\end{array}\\
&\begin{array}{l}
{\left.-b(z) \int_{\mathcal{P}} \frac{1}{Z} e^{\frac{-\left\|g_{\theta}(z)-y\right\|_{2}^{2}}{2 \sigma^{2}}} \nabla_{\theta}\left\|g_{\theta}(z)-y\right\|^{2} \mathrm{d} \mathbb{P}_{g}(y)\right]} \\
{=\mathbb{E}_{z \sim p(z)}\left[a(z) \int_{\mathcal{M}} P_{\epsilon}\left(g_{\theta}(z)-y\right) \nabla_{\theta}\left\|g_{\theta}(z)-y\right\|^{2} \mathrm{d} \mathbb{P}_{r}(y)\right.} \\
{\left.-b(z) \int_{\mathcal{P}} P_{\epsilon}\left(g_{\theta}(z)-y\right) \nabla_{\theta}\left\|g_{\theta}(z)-y\right\|^{2} \mathrm{d} \mathbb{P}_{g}(y)\right]}
\end{array}
\end{aligned}
$$得证

该定理证明，我们将样本$g_{\theta}$沿着数据流形朝着临界点移动，并对其概率和与样本之间的距离进行加权。 此外，第二项使我们的点远离高概率样本，再次由样本流形和到这些样本的距离加权。 这在本质上与对比散度相似，在对比散度中，我们降低了样本的自由能并增加了数据点的自由能。 当我们拥有比$\mathbb{P}_r$高的可能性时，从$\mathbb{P}_g$得出的样本更清楚地看到了该术语的重要性。 在这种情况下，我们将使b> a，并且第二项将具有降低这种太可能样本的概率的强度。 最后，如果x周围有一个区域与$\mathbb{P}_g$相比具有与$\mathbb{P}_r$相同的概率，则两个项之间的梯度贡献将抵消，因此当$\mathbb{P}_r$与$\mathbb{P}_g$相似时，可以稳定梯度。

完全采用上式的梯度步骤存在一个重要问题，那就是在这种情况下，$D$将忽略恰好位于$g_\mathcal(Z)$中的误差，因为这是一组度量0。但是，g将仅在该空间上优化其成本。这将使鉴别器极易受到对抗性例子的影响，并且将使生成器的成本降低而鉴别器的成本却不高，并且糟糕的样本将变得毫无意义。当我们意识到上式的期望内的项将为正标量乘以$\nabla_{x} \log \left(1-D^{*}(x)\right) \nabla_{\theta} g_{\theta}(z)$时，这是很容易看出的，这是朝向精确值的方向导数Goodfellow等人的对抗词。 （2014b）。因此，在发生器中的噪声样本中反向传播也很重要。这将产生关键的好处：生成器的反向支持期限将通过歧视者会关注的一组积极措施上的样本进行。正式化这个概念，通过发生器的实际梯度现在将与$\nabla_{\theta} J S D\left(\mathbb{P}_{r+\epsilon} \| \mathbb{P}_{q+\epsilon}\right)$成比例，这将使两个噪声分布匹配。当我们对噪声进行退火时，这也会使Pr和Pg匹配。为了完整起见，我们显示了在这种情况下获得的平滑渐变。证明与定理3.2的证明相同，因此我们留给读者。

- **推理3.2**:令$\epsilon, \epsilon^{\prime} \sim \mathcal{N}\left(0, \sigma^{2} I\right)$ 和 $\tilde{g}_{\theta}(z)=g_{\theta}(z)+\epsilon^{\prime}$那么
$$
\begin{array}{l}
{\mathbb{E}_{z \sim p(z), \epsilon^{\prime}}\left[\nabla_{\theta} \log \left(1-D^{*}\left(\tilde{g}_{\theta}(z)\right)\right)\right]} \\
{\quad=\mathbb{E}_{z \sim p(z), \epsilon^{\prime}}\left[a(z) \int_{\mathcal{M}} P_{\epsilon}\left(\tilde{g}_{\theta}(z)-y\right) \nabla_{\theta}\left\|\tilde{g}_{\theta}(z)-y\right\|^{2} \mathrm{d} \mathbb{P}_{r}(y)\right.} \\
{\left.\quad-b(z) \int_{\mathcal{P}} P_{\epsilon}\left(\tilde{g}_{\theta}(z)-y\right) \nabla_{\theta}\left\|\tilde{g}_{\theta}(z)-y\right\|^{2} \mathrm{d} \mathbb{P}_{g}(y)\right]} \\
{\quad=2 \nabla_{\theta} J S D\left(\mathbb{P}_{r+\epsilon} \| \mathbb{P}_{g+\epsilon}\right)}
\end{array}
$$

与定理3.2相同，a和b将具有相同的属性。 主要区别在于，我们会将所有嘈杂的样本移向数据流形，可以将其视为将一小部分样本移向数据流形。 这将保护区分器免受措施0的对抗示例

一个有趣的观察是，如果我们有两个分布在$\mathbb{P}_{r}$和$\mathbb{P}_{g}$的支撑集在流形上是封闭的，则噪声项将使嘈杂的分布$\mathbb{P}_{r+\epsilon}$和$\mathbb{P}_{g+\epsilon}$几乎重叠，并且它们之间的JSD将很小。 这与无噪声变型$\mathbb{P}_{r}$和$\mathbb{P}_{g}$形成了鲜明的对比，在无噪声变型中，所有歧管都最大化，而与歧管的紧密程度无关。 我们可以争辩说，使用带噪声的变体的JSD来度量原始分布之间的相似性，但这将取决于噪声量，而不是$\mathbb{P}_{r}$和$\mathbb{P}_{g}$的固有度量。 幸运的是，还有其他选择。

- **定义3.1**:$\mathcal{X}$上$P$和$Q$的两个分布的Wasserstein度量$W(P;Q)$的定义。$$
W(P, Q)=\inf _{\gamma \in \Gamma} \int_{\mathcal{X} \times \mathcal{X}}\|x-y\|_{2} d \gamma(x, y)
$$其中$\gamma $是$\mathcal{X} \times \mathcal{X}$上具有边际$P$ 和 $Q$的和


当我们减小噪声方差时，它平稳地变为0。
- **引理4**:如果$\epsilon$为一均值为$0$的随机向量，那么
$$
W\left(\mathbb{P}_{X}, \mathbb{P}_{X+\epsilon}\right) \leq V^{\frac{1}{2}}
$$
其中 $V=\mathbb{E}\left[\|\epsilon\|_{2}^{2}\right]$ 是 $\epsilon$的方差
**证明**:令$x \sim \mathbb{P}_{X}$和$y=x+\epsilon$ 其中 $\epsilon$和$x$无关。 $\gamma$ 为$(x, y)$的节点，具有边际$\mathbb{P}_{X}$ 和 $\mathbb{P}_{X+\epsilon}$ 因此
$$
\begin{aligned}
W\left(\mathbb{P}_{X}, \mathbb{P}_{X+\epsilon}\right) & \leq \int\|x-y\|_{2} d \gamma(x, y) \\
&=\mathbb{E}_{x \sim \mathbb{P}_{X}} \mathbb{E}_{y \sim x+\epsilon}\left[\|x-y\|_{2}\right] \\
&=\mathbb{E}_{x \sim \mathbb{P}_{X}} \mathbb{E}_{y \sim x+\epsilon}\left[\|\epsilon\|_{2}\right] \\
&=\mathbb{E}_{x \sim \mathbb{P}_{X}} \mathbb{E}_{\epsilon}\left[\|\epsilon\|_{2}\right] \\
&=\mathbb{E}_{\epsilon}\left[\|\epsilon\|_{2}\right] \\
& \leq \mathbb{E}_{\epsilon}\left[\|\epsilon\|_{2}^{2}\right]^{\frac{1}{2}}=V^{\frac{1}{2}}
\end{aligned}
$$其中最后由于Jensen导致不平等

现在我们来谈谈我们的主要结果之一。 我们感兴趣的是研究Pr和Pg之间的距离而没有任何噪声，即使它们的支撑位于不同的歧管上，因为（例如）这些歧管越近，样本数据歧管上的实际点也越近。 此外，我们最终想要一种评估生成模型的方法，而不论它们是连续的（如在VAE中）还是不连续的（如在GAN中），这是一个目前尚未完全解决的问题。 下一个定理将Pr和Pg的Wasserstein距离（无任何噪声或修改）与Pr +和Pg +的发散度以及噪声的方差相关。 由于Pr +和Pg +是连续分布，因此这种偏差是一个明智的估计，甚至可以尝试将其最小化，因为根据这些分布训练的鉴别器将近似它们之间的JSD，并根据Corolary 3.2提供平滑的梯度。


- **定理3.3**:令 $\mathbb{P}_{r}$ and $\mathbb{P}_{g}$ be any two distributions, and $\epsilon$ a random vector with mean 0 and variance $V .$ If $\mathbb{P}_{r+\epsilon}$ and $\mathbb{P}_{g+\epsilon}$ have support contained on a ball of diameter $C,$ then 
$$
W\left(\mathbb{P}_{r} | \mathbb{P}_{g}\right) \leq 2 V^{\frac{1}{2}}+2 C \sqrt{J S D\left(\mathbb{P}_{r+\epsilon} \| \mathbb{P}_{g+\epsilon}\right)}
$$
**证明**:
$$
\begin{aligned}
W\left(\mathbb{P}_{r}, \mathbb{P}_{g}\right) & \leq W\left(\mathbb{P}_{r}, \mathbb{P}_{r+\epsilon}\right)+W\left(\mathbb{P}_{r+\epsilon}, \mathbb{P}_{g+\epsilon}\right)+W\left(\mathbb{P}_{g+\epsilon}, \mathbb{P}_{g}\right) \\
& \leq 2 V^{\frac{1}{2}}+W\left(\mathbb{P}_{r+\epsilon}, \mathbb{P}_{g+\epsilon}\right) \\
& \leq 2 V^{\frac{1}{2}}+C \delta\left(\mathbb{P}_{r+\epsilon}, \mathbb{P}_{g+\epsilon}\right) \\
& \leq 2 V^{\frac{1}{2}}+C\left(\delta\left(\mathbb{P}_{r+\epsilon}, \mathbb{P}_{m}\right)+\delta\left(\mathbb{P}_{g+\epsilon}, \mathbb{P}_{m}\right)\right) \\
& \leq 2 V^{\frac{1}{2}}+C(\sqrt{\frac{1}{2} K L\left(\mathbb{P}_{r+\epsilon} \| \mathbb{P}_{m}\right)}+\sqrt{\frac{1}{2} K L\left(\mathbb{P}_{g+\epsilon} \| \mathbb{P}_{m}\right)}) \\
& \leq 2 V^{\frac{1}{2}}+2 C \sqrt{J S D\left(\mathbb{P}_{r+\epsilon} \| \mathbb{P}_{g+\epsilon}\right)}
\end{aligned}
$$

定理3.3告诉我们一个有趣的idea。即上述式子中的两项是可以控制的。第一项可以通过噪音退火的方式来减少，第二项可以通过一个GAN（基于噪音输入来训练判别器）来最小化，因为他会近似于两个连续分布的JSD。该方法的一个优点是我们不再需要担心训练的选择方案。因为噪音，我们可以训练判别器直到最优而且没任何问题，并通过推理3.2得到平滑的可解释梯度。所有这一切仍然是在最小化$\mathbb{P}_r$和$\mathbb{P}_g$之间的距离，这两个分布也是我们最终关心的两个无噪声分布。

>**$\downarrow$结论白话文$\downarrow$**

**原始GAN问题的问题根源**
- 等价优化的距离衡量（KL散度、JS散度）不合理
- 生成器随机初始化后的生成分布很难与真实分布有不可忽略的重叠。

本文其实已经针对第二点提出了一个解决方案，就是**对生成样本和真实样本加噪声**
- 直观上说，使得原本的两个低维流形“弥散”到整个高维空间，强行让它们产生不可忽略的重叠。而一旦存在重叠，JS散度就能真正发挥作用，此时如果两个分布越靠近，它们“弥散”出来的部分重叠得越多，JS散度也会越小而不会一直是一个常数，于是（在第一种原始GAN形式下）梯度消失的问题就解决了。

在训练过程中，我们可以对所加的噪声进行退火（annealing），慢慢减小其方差，到后面两个低维流形“本体”都已经有重叠时，就算把噪声完全拿掉，JS散度也能照样发挥作用，继续产生有意义的梯度把两个低维流形拉近，直到它们接近完全重合。以上是对原文的直观解释。
在这个解决方案下我们可以放心地把判别器训练到接近最优，不必担心梯度消失的问题。而当判别器最优时，对公式26取反可得判别器的最小loss为
$$ \begin{aligned} \min L_D(P_{r+\epsilon}, P_{g+\epsilon}) &= - \mathbb{E}_{x\sim P_{r+\epsilon}}\,\,[\log D^*(x)] - \mathbb{E}_{x\sim P_{g+\epsilon}}\,\,[\log(1-D^*(x))] \\ &= 2\log 2 - 2JS(P_{r+\epsilon} || P_{g+\epsilon}) \end{aligned}$$
其中$P_{r+\epsilon}$和$P_{g+\epsilon}$分别是加噪后的真实分布与生成分布。反过来说，从最优判别器的loss可以反推出当前两个加噪分布的JS散度。两个加噪分布的JS散度可以在某种程度上代表两个原本分布的距离，也就是说可以通过最优判别器的loss反映训练进程！……真的有这样的好事吗？
并没有，因为加噪JS散度的具体数值受到噪声的方差影响，随着噪声的退火，前后的数值就没法比较了，所以它不能成为$P_r$和$P_g$距离的本质性衡量。
加噪方案是针对原始GAN问题的第二点根源提出的，解决了训练不稳定的问题，不需要小心平衡判别器训练的火候，可以放心地把判别器训练到接近最优，但是仍然没能够提供一个衡量训练进程的数值指标。但是WGAN从第一点根源出发，用Wasserstein距离代替JS散度，同时完成了稳定训练和进程指标的问题！
> **$\uparrow$结论白话文$\uparrow$**


WGAN前作分析了Ian Goodfellow提出的原始GAN两种形式各自的问题，第一种形式等价在最优判别器下等价于最小化生成分布与真实分布之间的JS散度，由于随机生成分布很难与真实分布有不可忽略的重叠以及JS散度的突变特性，使得生成器面临梯度消失的问题；第二种形式在最优判别器下等价于既要最小化生成分布与真实分布直接的KL散度，又要最大化其JS散度，相互矛盾，导致梯度不稳定，而且KL散度的不对称性使得生成器宁可丧失多样性也不愿丧失准确性，导致collapse mode现象。
WGAN前作针对分布重叠问题提出了一个过渡解决方案，通过对生成样本和真实样本加噪声使得两个分布产生重叠，理论上可以解决训练不稳定的问题，可以放心训练判别器到接近最优，但是未能提供一个指示训练进程的可靠指标，也未做实验验证。

---

# 附录
## <span id="Appendix_1">附录1——两者分布不重合JS散度为log2</span>
**题目**：对于真实数据分布$P_r$和生成数据分布$P_g$，如果满足上述无法全维度重合的情况的话，则$J S D\left(P_{r} \| P_{g}\right)=\log 2$
**证明**：
KL散度定义：$D_{K L}(P \| Q)=\int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} d x$
JS散度定义：$D_{JS}(P||Q)={\frac{1}{2}} KL(P||M) + {\frac{1}{2}} KL(Q||M) \quad \quad M = {\frac{1}{2}}(P+Q)$
$$
\begin{aligned}
&\therefore 
    \begin{aligned} 
    J S D(P \| Q)&=\frac{1}{2} \int_{-\infty}^{\infty} p(x) \log \left(\frac{p(x)}{\frac{p(x)+q(x)}{2}}\right)+\frac{1}{2} \int_{-\infty}^{\infty} q(x) \log \left(\frac{q(x)}{\frac{p(x)+q(x)}{2}}\right)\\
    &=\frac{1}{2} \int_{-\infty}^{\infty} p(x) \log \left(\frac{2 p(x)}{p(x)+q(x)}\right)+\frac{1}{2} \int_{-\infty}^{\infty} q(x) \log \left(\frac{2 q(x)}{p(x)+q(x)}\right)
    \end{aligned}\\
&\because \int_{-\infty}^{\infty} p(x)=\int_{-\infty}^{\infty} q(x)=1\\
&\therefore J S D(P \| Q)=\frac{1}{2} \sum p(x) \log \left(\frac{p(x)}{p(x)+q(x)}\right)+\frac{1}{2} \sum q(x) \log \left(\frac{q(x)}{p(x)+q(x)}\right)+\log 2
\end{aligned}
$$
因为对于任意一个$x$只有四种可能：
- $P_1(x) = 0$且$P_2(x) = 0$对计算JS散度无贡献
- $P_1(x) \neq 0$且$P_2(x) \neq 0$由于重叠部分可忽略所以贡献也为0
- $P_1(x) = 0$且$P_2(x) \neq 0$第一项因为$0 \times y=0$,第二项因为$\log (1)=0$
- $P_1(x) \neq 0$且$P_2(x) = 0$与上述情况类似
$$\therefore \forall x \in R, \text { 都有 } J S D(P \| Q)=\log 2$$

---

[Paper---Towards Principled Methods for Training Generative Adversarial Networks](http://arxiv.org/abs/1701.04862)
[博客园---Generative Adversarial Nets[Pre-WGAN]](https://www.cnblogs.com/shouhuxianjian/p/10268624.html)
[CSDN---Wasserstein GANs 三部曲（一）：Towards Principled Methods for Training Generative Adversarial Networks的理解](https://blog.csdn.net/xiaohouzi1992/article/details/80839921)
[知乎---令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)


[知乎---WGAN和GAN直观区别和优劣](https://zhuanlan.zhihu.com/p/31394374)
[CSDN---GAN（Generative Adversarial Network）的学习历程](https://blog.csdn.net/Jasminexjf/article/details/82621223)