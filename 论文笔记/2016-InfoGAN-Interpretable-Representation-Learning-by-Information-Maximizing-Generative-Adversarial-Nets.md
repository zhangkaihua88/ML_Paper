<!-- toc -->
[toc]

---

# 1. InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
>arXiv:1606.03657 [cs.LG]
>tensorflow2代码：https://github.com/zhangkaihua88/ML_Paper

---

## 1.1. 摘要
**InfoGAN**
- 对生成对抗网络的信息理论扩展
- 完全无监督的方式学习特征分离表示
- 能最大化潜在变量的一小部分与观察(生成)结果之间的互信息

**本文**
- 有效优化的互信息目标的下界
- InfoGAN可以学习与现有监督方法学习得到了具有竞争力的可解释性表征。

---

## 1.2. 引言
**无监督学习**
- 一般可以被描述为从大量存在的未标记数据中提取数值的问题。
- 流行框架是**表征学习**
    - 目标是使用未标记的数据来学习一种表示，以从重要语义特征中找到易于解释的要素。
- 特征分离的表示对于需要知道数据的显著属性的自然任务可能是有用的
- 无监督学习算法必须对下游分类任务有正确的效果（在不直接接触下游任务的情况下）。
- 很大一部分是由生成模型驱动的。
    - 动力源于对生成模型能力的相信，或为观察数据“创造”某种形式的理解，并希望良好的生成模型能自动学习一个有区分度的表示，即便通过随便一个不好的表示也容易构造完美的生成模型。最突出的生成模型是变分自动编码器（VAE）和生成对抗网络（GAN）。

**本文**
- 生成对抗网络目标进行了简单的修改，鼓励其学习可解释和有意义的表示。
- 通过最大化GAN噪声变量的固定小子集与观测值之间的互信息来实现，这一点相对比较直观。
- 增加互信息成本的生成模型是学习特征表示的有效途径。

--- 

## 1.3. 相关工作

现在存在大量关于无监督表示学习的工作。
- 早期的方法是基于堆叠的（通常是去噪）自动编码器或受限玻尔兹曼机
- 阶梯网络，它在MNIST数据集的半监督变学习上取得了惊人的成果。

此外，先前的研究试图使用监督数据来学习分离的特征表示。
- 使用监督学习训练表示的一部分来匹配所提供的标签
- 弱监督方法来消除对明确标记变量的需要

---



## 1.4. 补充知识
### 1.4.1. 信息量
$ I(x) = -\log {p(x)} = \log { \frac { 1}{ p (x) }  } $
一个事件发生的概率越大，这件事情发生所包含的信息量就越小，比如说一个高富帅追求一个白富美，追到手了没有什么稀奇的，因为这是一件概率很高的事情，但是如果一个矮穷矬追求一个白富美，追到手了，这种事情发生的概率很低，其中一定有其他的原因：比如这个矮穷矬救过白富美的命或者这个矮穷矬器大活好不黏人，所以概率低的事情发生所包含的信息量大；两个相互独立的事情同时发生的信息量是两者单独发生的信息量之和。
### 1.4.2. 信息熵
信息量的均值
$$H(x) = - \sum _{ x } p(x)log p(x) $$

### 1.4.3. 互信息
在信息论中，X和Y之间的互信息 $I(X;Y)$测量从随机变量Y的知识中学习的关于另一个随机变量X的“信息量”。
    - 互信息可以表示为两个熵项的差值：
$$I(X;Y)=H(X)-H(X|Y)=H(Y)-H(Y|X)$$
这个定义有一个直观的解释： $I(X,Y)$是观察到 $Y$ 时， $X$的不确定性的减少量。如果 $X$ 和 $Y$ 是独立的，那么 $I(X;Y)=0$ ，因为一个变量与另一个变量毫无关系；相反，如果$X$和$Y$通过确定性可逆函数相关，则获得最大互信息。
![20200209214308.png](https://image.zkhweb.top/20200209214308.png)

---

## 1.5. InfoGAN

### 1.5.1. 知识回顾——生成对抗网络
生成器G，判别器D，相互对抗使目标函数，达到最优。
$$ \min _{G}\max _{ D } V(D,G)={ \mathbb{E} }_{ x ～ { p }_  { data } (x) }[logD(x)] + { \mathbb{E} }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z)))]$$

但是**无约束、不可控、噪声信号z很难解释等问题。**

### 1.5.2. 符号定义
$x$ $\rightarrow$ 真实数据
$y$ $\rightarrow$ 标签（辅助信息）
$z$ $\rightarrow$ 噪音（生成器的输入数据）
$c$ $\rightarrow$ 潜在代码(latent code)
$p_c$ $\rightarrow$ 潜在代码的分布，可以为连续分布，也可以为离散分布
$p_x$ $\rightarrow$ 真实数据的分布
$p_{z}(z)$ $\rightarrow$ 原始噪音数据的分布
$p_g$ $\rightarrow$ 经过生成器后数据的分布
$G()$ $\rightarrow$ 生成映射函数（可微），结构为多层感知机，参数$\theta_{g}$
$D()$ $\rightarrow$ 判别映射函数（可微），结构为多层感知机，参数$\theta_{d}$
$Q(c|x)$ $\rightarrow$ 辅助分布用于逼近后验概率$P(c|x)$
$G(z,c;\theta_{g})$ $\rightarrow$ 将噪音$z$和潜在代码映射到新的数据空间
$D(x ; \theta_{d})$ $\rightarrow$ $x$来自真实数据而不是生成数据的概率（真=1，假=0）

### 1.5.3. 直观感受
![20200209213237.png](https://image.zkhweb.top/20200209213237.png)
当Q与D共享参数
![20200209213913.png](https://image.zkhweb.top/20200209213913.png)

### 1.5.4. 目标函数
$$
\begin{aligned}
\min _{G} \max _{D} V_{I}(D, G)
&=V(D, G)-\lambda I(c ; G(z, c)) \\
&= { \mathbb{E} }_{ x ～ { p }_  { data } (x) }[logD(x)] + { \mathbb{E} }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z)))]-\lambda I(c ; G(z, c))
\end{aligned}
$$
和原始GAN相近，只是G中加入了潜码$c$（和$G(z,c)$有关），可以调节$c$来改变生成样式(加入了互信息的惩罚)

当用分布$Q(c|x)$逼近后验概率$P(c|x)$时，目标函数变为
$$\min_{G,Q}\max_D V_{InfoGAN}(D,G,Q)=V(D,G)-\lambda L_I(G,Q)$$
**Note**:此处的$V(D, G)$不局限于原始GAN的代价函数，可以使用其他的代价函数，以获得更好的训练

### 1.5.5. 对抗
判别器$D$的目标
1. 要尽可能把真的样本判断为真，对应最大化第一项：${ E }_{ x ～ { p }_  { data } (x) }[logD(x)]$
2. 把假的样本判断为假，对应最大化第二项：${ E }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z,c)))] $
- 总之，也就是说**判别器$D$要最大化目标函数**；

生成器$G$的目标
1. 要尽可能的让$D$将生成的假样本判断为真，对应最小化第二项：${ E }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z)))] $
2. 同时加入潜码$c$是为了使生成$G(z,c)$和$c$有很强的关联：$I(c ; G(z, c))$最大化，即$-\lambda I(c ; G(z, c))$($\lambda$为整数)最小

- 总之，也就是说**生成器$G$要最小化目标函数**；

总的来说，这是一个**信息正则化的MinMax Game**；
当用分布$Q(c|x)$逼近后验概率$P(c|x)$时，InfoGAN**具有互信息和超参数λ的变分正则化的MinMax Game**

### 1.5.6. 目标函数由来
为了让让网络学习到了可解释的特征表示
将提出将**输入噪声向量分成为两部分**：
- $z$:被视为不可压缩的噪声源；
- $c$[^1]:将其称为潜在代码(latent code)，其目的在于数据分布的显著结构化的语义特征。

[^1]:在数学上，$c_1,c_2,...,c_L$​ 表示结构化潜在变量的集合。在其最简单的形式中，可以假设一个因式分布，由 $P\left(c_{1}, c_{2}, \ldots, c_{L}\right)=\prod_{i=1}^{L} P\left(c_{i}\right)$ 给出。为了便于表示，使用潜在代码 $c$ 来表示所有潜在变量 $c_i$​ 的联合。

但是在标准GAN中，通过找到满足 $P_G(x|c)= P_G(x)$ 的解，生成器可以自由地忽略附加潜在代码$c$ 。

提出了一种**基于信息论的正则化方法**：潜在码 $c$ 和生成分布 $G(z,c)$ 之间应该有很高的互信息。因此 $I(c;G(z,c))$ 应该很高。

给定任何 $x \sim P_{G}(x)$ ，希望 $P_G(c|x)$ 具有小的熵。换句话说，**潜在码 $c$ 中的信息不应该在生成过程中丢失**。在聚类的背景下，之前已经有过类似的互信息启发目标函数。因此，本文建议通过以下信息正则化的minimax游戏来解决问题：
$$
\min _{G} \max _{D} V_{I}(D, G)=V(D, G)-\lambda I(c ; G(z, c))
$$

### 1.5.7. 变分互信息最大化
**引理1**对于随机变量$X$，$Y$和函数$f(x,y)$在适当的正则条件下$\mathbb{E}_{x \sim X, y \sim Y | x}[f(x, y)]=\mathbb{E}_{x \sim X, y \sim Y\left|x, x^{\prime} \sim X\right| y\left[f\left(x^{\prime}, y\right)\right]}$

**证明**：
$$
\begin{aligned}
\mathbb{E}_{x \sim X, y \sim Y | x}[f(x, y)] &=\int_{x} P(x) \int_{y} P(y | x) f(x, y) d y d x \\
&=\int_{x} \int_{y} P(x, y) f(x, y) d y d x \\
&=\int_{x} \int_{y} P(x, y) f(x, y) \int_{x^{\prime}} P\left(x^{\prime} | y\right) d x^{\prime} d y d x \\
&=\int_{x} P(x) \int_{y} P(y | x) \int_{x^{\prime}} P\left(x^{\prime} | y\right) f\left(x^{\prime}, y\right) d x^{\prime} d y d x \\
&=\mathbb{E}_{x \sim X, y \sim Y\left|x, x^{\prime} \sim X\right| y}\left[f\left(x^{\prime}, y\right)\right]
\end{aligned}
$$

**变分信息最大化**
互信息项 $I(c;G(z,c))$ 难以直接最大化，因为它先得到后验概率 $P(c|x)$ 。
**通过定义辅助分布 $Q(c|x)$ 来逼近 $P(c|x)$** ，可以得到它的下界：

$$
\begin{aligned}
I(c ; G(z, c)) &=H(c)-H(c | G(z, c)) \\
&=\mathbb{E}_{c \sim P(c),x \sim G(z, c)}[\log P(c |x)]+H(c)\\
根据引理1&=\mathbb{E}_{x \sim G(z, c)} \left[\mathbb{E}_{c^{\prime} \sim P(c | x)} \left[\log P(c^{\prime}|x) \right] \right]+H(c)\\
&=\mathbb{E}_{x \sim G(z, c)}\left[\mathbb{E}_{c^{\prime} \sim P(c | x)}\left[\log P\left(c^{\prime} | x\right)\right]\right]+H(c) \\
&=\mathbb{E}_{x \sim G(z, c)}\left[\mathbb{E}_{c^{\prime} \sim P(c | x)} \left[ \frac{P(c^{\prime} | x) Q(c^{\prime} | x)}{Q(c^{\prime} | x)} \right] \right] \\
&=\mathbb{E}_{x \sim G(z, c)}\left[  \mathbb{E}_{c^{\prime} \sim P(c | x)} \left[ \log \frac{P(c^{\prime} | x)}{Q(c^{\prime} | x)}\right]+ \mathbb{E}_{c^{\prime} \sim P(c | x)}\left[\log Q\left(c^{\prime} | x\right)\right]\right]+H(c)\\
&=\mathbb{E}_{x \sim G(z, c)}\left[\underbrace{D_{K L}(P(\cdot | x) \| Q(\cdot | x))}_{\geq 0}+\mathbb{E}_{c^{\prime} \sim P(c | x)}\left[\log Q\left(c^{\prime} | x\right)\right]\right]+H(c) \\
& \geq \mathbb{E}_{x \sim G(z, c)}[{\left.D_{c^{\prime} \sim P(c | x)}\left[\log Q\left(c^{\prime} | x\right)\right]\right]}+H(c)
\end{aligned}
$$



潜在编码 $H(c)$ 的熵也可以优化，因为对于常见分布，它具有简单的分析形式。然而，在本文中，通过修复潜在编码分布来选择简化的表示，并**将 $H(c)$ 视为常量**。



**引理2** 对随机变量 $X$,$Y$ 和函数 $f(x,y)$ ，在合适的规则条件下： $\mathbb{E}_{x \sim X, y \sim Y | x}[f(x, y)]=\mathbb{E}_{x \sim X, y \sim Y|x, x^{\prime} \sim X| y}[f\left(x^{\prime}, y\right)]$

**证明**：
通过使用引理1可以定义互信息 $I(c;G(z,c))$ 变分的下界 $L_I(G,Q)$:
$$
\begin{aligned}
L_{I}(G, Q) &=E_{c \sim P(c), x \sim G(z, c)}[\log Q(c | x)]+H(c) \\
&=E_{x \sim G(z, c)}\left[\mathbb{E}_{c^{\prime} \sim P(c | x)}\left[\log Q\left(c^{\prime} | x\right)\right]\right]+H(c) \\
& \leq I(c ; G(z, c))
\end{aligned}
$$

注意到 $L_I(G,Q)$ 很容易用**蒙特卡罗模拟近似**。 特别是，可以对于 w.r.t.$Q$ 和 w.r.t.$G$ 使用重参数化技巧将 $L_I$​ 最大化。 因此， $L_I(G,Q)$ 可以添加到GAN的目标而不改变GAN的训练过程，称之为**信息最大化生成对抗网络(InfoGAN)**。

变分互信息最大化表明，当辅助分布 $Q$ 接近真实的后验分布 $\mathbb E_x[D_{KL}P(·|x)||Q(·|x))]→0$ 时，下限变紧了。另外，当变分下界达到离散潜码的最大值 $L_I(G,Q)= H(c)$ 时，边界变紧并且达到最大互信息。

---

## 1.6. 实现
**整体**
- 将辅助分布 $Q$ 参数化为神经网络。 
    - 在大多数实验中， $Q$ 和 $D$ **共享所有卷积层**
    - 存在一个最终完全连接的层以输出条件分布 $Q(c|x)$ 的参数(softmax)
    
这意味着InfoGAN仅向GAN添加了**可忽略的计算成本**
在实验中， $L_I(G,Q)$ 总是比正常的GAN目标更快收敛

**分类的潜在代码$c_i$​**
- 使用非线性softmax来表示 $Q(c_i|x)$
    - 对于连续潜在代码 $c_j$​ ，根据真正的后验概率 $P(c_j|x)$ ，有更多的选项。可以简单地将 $Q(c_j|x)$ 作为因子化的高斯来处理就足够了。
- **c为离散变量**
$Q(c|x)$可以表示为一个神经网络$Q(X)$的输出
$$
\begin{aligned}
&\because L_{I}(G, Q)=c \cdot \log Q(G(z, c))+H(c)\\
&\therefore \max L_{I}(G, Q) \text{等价于} \max c \cdot \log Q(G(z, c))
\end{aligned}
$$
其中$\cdot$表示内积，c是一个选择计算哪个$\log$的参数，$H(c)$可以消去
**$L_I(G,Q)$本质上是$x$与$G(z,x)$之间的KL散度**
$$\begin{aligned}
&\because \begin{aligned}
D_{K L}(c \| Q(G(z, c))) &=-c \cdot \log \frac{Q(G(z, c))}{c} \\
&=-c \cdot \log Q(G(z, c))+c \cdot \log c \\
&=-c \cdot \log Q(G(z, c))+H(c) \\
&=-L_{I}(G, Q)+2 H(c)
\end{aligned}\\
&\therefore \max L_{I}(G, Q) \text{等价于} \min D_{K L}(c \| Q(G(z, c)))
\end{aligned}
$$
$\min D_{K L}(c \| Q(G(z, c)))$意味着减少$c$与$Q(G(z, c)$

- **c为连续变量**
设 $Q(x)$ 输出的参数潜码 $c$ 的均值 $\mu$，标准差 $\sigma$ 分别为 $Q(x)_{\mu}$ 和$Q(x)_{\sigma}$，那么对于参数潜码 c
$$
\begin{aligned}
&L_{I}(G, Q)=\mathbb{E}_{x \sim G(z, c)}\left[\mathbb{E}_{c^{\prime} \sim P(c | x)}\left[\log Q\left(c^{\prime} | x\right)\right]\right]+H(c) \\
&\because p(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}}(\text{c符合正态分布;均值 $\mu$,标准差 $\sigma$})\\
&\therefore \log p(c) = -\frac{(c-\mu)^{2}}{2 \sigma^{2}}-\log (\sigma \sqrt{2 \pi})\\
&\therefore L_{I}(G, Q)=-\frac{\left(c-Q(x)_{\mu}\right)^{2}}{2 Q(x)_{\sigma}^{2}}-\log \left(Q(x)_{\sigma} \sqrt{2 \pi}\right)+H(c)\\
&\therefore \max L_{I}(G, Q) \text{等价于} \min \left(\frac{\left(c-Q(x)_{\mu}\right)^{2}}{2 Q(x)_{\sigma}^{2}}+\log \left(Q(x)_{\sigma} \sqrt{2 \pi}\right)\right)
\end{aligned}
$$
不考虑$Q(x)_{\sigma}$的影响则
$$
\max L_{I}(G, Q) \text{等价于} \min \left(c-Q(x)_{\mu}\right)^{2}
$$
$\min \left(c-Q(x)_{\mu}\right)^{2}$意味这减小$c$与$Q(x)_{\mu}$的差

- **总之$L_{I}$的优化过程，实质上是以G为编码器(Encoder)，Q为解码器(Decoder)，生成图像作为要编码的码(code)，训练一个自编码器(Autoencoder)**

**超参数 $\lambda$**
- 对于*离散*的潜在代码，它很容易调整，简单地设置为1就足够。
- 当潜在编码包含*连续*变量时，较小的 $\lambda$ 通常用于确保包含差分熵的 $L_I(G,Q)$ 的规模与GAN目标的规模相同。

---

## 1.7. 实验
**目标**
- 调查是否可以有效地最大化互信息
- 评估InfoGAN是否可以通过利用生成器一次仅改变一个潜在因子来学习有区分度的可解释的表示，以评估改变这样的因素是否导致生成的图像中只有一种类型的语义变化。 

### 1.7.1. 互信息的最大化
为了评估潜在编码 $c$ 和生成的图像 $G(z,c)$ 之间的互信息是否可以通过提出的方法有效地最大化，作者在MNIST数据集上训练InfoGAN，对潜在编码 $c$ 进行统一的分类分布 $c\sim Cat(K=10,p=0.1)$ 。在 图1 中，下限 $L_I(G,Q)$ 被快速最大化为 $H(c)\approx 2.30$ ，这意味着下限 (4) 快速紧贴到到最大的互信息。

作为基准，当没有明确促使生成图像与潜在编码最大化的互信息时，作者还训练具有辅助分布 $Q$ 的常规GAN。由于作者使用神经网络对 $Q$ 进行参数化，假设 $Q$ 可以合理地近似真实的后验概率 $P(c|x)$ ，因此在常规GAN中潜在编码和生成图像之间几乎没有互信息。作者注意到，使用不同的神经网络架构，即使在实验中没有观察到这种情况，潜在代码和生成的图像之间可能存在更高的相互信息。这种比较是为了证明在常规GAN中，不能保证生成器能够利用潜在编码。

### 1.7.2. 有区分度的表示
**离散的c捕捉离散变换，连续的c捕捉连续变换**
**MNIST**
- $c_1\sim Cat(K=10,p=0.1)$ 控制数字
- $c_2 \sim Unif(-1,1)$​ 控制旋转数字
- $c_3 \sim Unif(-1,1)$​ 控制宽度

但$c_2,c_3$​还调整其他细节，如厚度或笔触样式
$c$处于$-2 \sim 2$之间仍有用，表明InfoGAN学习的潜在表示可泛化

**Faces**
$c_i\sim Unif(-1,1),1\leq i\leq 5$ 
表示为方位角（姿势），仰角和照明等潜在因素看做连续潜在变量。

**CelebA**
$c_1,c_2,c_3,c_4\sim (K=20,p=0.05)$和一个连续码 $c_5\sim Unif(-1,1)$ 
代表旋转的连续编码。使用单个连续代码连续插入不同宽度的类似椅子类型。

**SVHN**
利用四个10维分类变量和两个均匀连续变量作为潜在代码。

---

## 1.8. 结论
- InfoGAN完全没有监督
- 在具有挑战性的数据集上学习可解释和有区分度的表示。
- InfoGAN在GAN之上仅增加了可忽略不计的计算成本，并且易于训练。 
- 使用互信息表示的核心思想可以应用于其他方法，

---

## 1.9. 附录——解释为“sleep-sleep”算法
InfoGAN可以看作是Helmholtz机（wake-sleep算法）：
- $P_{G}(x | c)$是生成分布
- $Q(c|x)$识别分布

提出wake-sleep算法，通过执行wake阶段和sleep阶段更新Helmholtz机
- wake阶段：通过优化有关生成的变分下界$\log P_G(x)$来更新
$$\max _{G} \mathbb{E}_{x \sim \operatorname{Data}, c \sim Q(c | x)}\left[\log P_{G}(x | c)\right]$$
- sleep阶段：通过在当前生成分布中生成样本，而不是从实际数据分布中提取样本来更新辅助分布

因此，当优化代理损失函数[^2]$L_I$（关于$Q$），更新步骤正是wake-sleep算法中的sleep过程。
InfoGAN与Wake-sleep不同之处在于优化$L_I$（关于$Q$）生成网络G在潜在代码$P(c)$整个先前分布中使用了潜在代码$c$。由于InfoGAN在sleep过程中更新了生成器，所以可以解释为sleep-sleep算法。

这种解释突出了InfoGAN与以前生成建模技术的区别：
明确鼓励生成器以潜在代码传达信息，并建议将相同原理应用于其他生成模型。


[^2]:代理损失函数:当原本的loss function不便计算的时候，我们就会考虑使用surrogate loss function。

---

## 1.10. 参考资料
[Paper---InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)
[CSDN---深度学习（四十八）InfoGAN学习笔记](https://blog.csdn.net/hjimce/article/details/55657325)
[CSDN---InfoGAN论文笔记+源码解析](https://blog.csdn.net/wspba/article/details/54808833)
[知乎---InfoGAN解读](https://zhuanlan.zhihu.com/p/58261928)
[Github---DequanZhu/GANs-collections-tf2.0_keras-eager_mode](https://github.com/DequanZhu/GANs-collections-tf2.0_keras-eager_mode)
[Github---openai/InfoGAN](https://github.com/openai/InfoGAN)
[个人Blog---InfoGAN: using the variational bound on mutual information (twice)](https://www.inference.vc/infogan-variational-bound-on-mutual-information-twice/)
[机器之心---InfoGAN：一种无监督生成方法 | 经典论文复现](https://www.jiqizhixin.com/articles/2018-10-29-21)
[程序员大本营---【GAN ZOO翻译系列】InfoGAN： Interpretable Representation Learning by Information Maximizing GAN](http://www.pianshen.com/article/448358405/)