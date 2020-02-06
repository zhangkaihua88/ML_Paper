<!-- toc -->
[toc]
# 1. Generative Adversarial Nets
>arXiv:1406.2661 [stat.ML]


---

## 1.1. 摘要
通过对抗过程估计生成模型的框架，同时训练两个模型：
**生成模型G**用来获取数据分布
**判别模型D**估计样本来自训练数据而不是G的概率
G的训练目标是为了最大化D产生错误的概率
在任意函数G和D的空间中存在唯一的解，其中G恢复训练数据分布，并且D处处都等于$\frac{1}{2}$。

---

## 1.2. 介绍
最成功的的模型之一就是**判别式模型**
通常它们将高维丰富的感知器输入映射到类标签上。
主要是**基于反向传播和丢弃算法**来实现的，特别是具有特别良好梯度的分段线性单元。


对抗网络
1. 生成模型通过将随机噪声传输到多层感知机来生成样本的特例
2. 判别模型通过多层感知机去判别一个样本是来自模型分布还是数据分布
3. 生成模型和判别模型互相对抗

可以仅使用非常成熟的反向传播和丢弃算法训练两个模型，生成模型在生成样本时只使用前向传播算法。并且不需要近似推理和马尔可夫链作为前提。

---

## 1.3. 对抗网络
对抗模型框架是最直接的应用是多层感知机
### 1.3.1. 符号定义
$x$ $\rightarrow$ 真实数据
$z$ $\rightarrow$ 噪音（生成器的输入数据）
$p_x$ $\rightarrow$ 真实数据的分布
$p_{z}(z)$ $\rightarrow$ 原始噪音数据的分布
$p_g$ $\rightarrow$ 经过生成器后数据的分布
$G()$ $\rightarrow$ 生成映射函数（可微），结构为多层感知机，参数$\theta_{g}$
$D()$ $\rightarrow$ 判别映射函数（可微），结构为多层感知机，参数$\theta_{d}$
$G(z;\theta_{g})$ $\rightarrow$ 将噪音$z$映射到新的数据空间
$D(x ; \theta_{d})$ $\rightarrow$ $x$来自真实数据而不是生成数据的概率（真=1，假=0）

### 1.3.2. 极大似然估计
对于真实数据$x$和生成数据$G(z)$，经过判别器判别后的，$D$认为$x$是真样本的概率为$D(x)$，$D$认为$G(z)$是假样本的概率为$1-D(G(z))$，那么对于$D$有$log$似然函数为：
$$L=log[D(x)*(1-D(G(z)))] \tag{1}$$

### 1.3.3. 目标函数
$$ \min _{G}\max _{ D } V(D,G)={ \mathbb{E} }_{ x ～ { p }_  { data } (x) }[logD(x)] + { \mathbb{E} }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z)))] \tag{2}$$
$D(x)$和$D(G(z))$分别表示$x$和$G(z)$经过判别器$D$的判别后，$D$认为输入样本是真样本的概率，则$1-D(G(z))$表示$D$将假样本判断为假的概率；那么，真实的概率分布与$D$判断出来的情况列表如下：

| $D$ | $D$将真样本$x$判断为真的概率:$D(x)$ | $D$将假样本$G(z)$判断为假的概率:$1-D(G(z))$ |
| :-: | :-: | :-: |
| 真实情况 | 真样本$x$为真的概率:1 | 假样本$G(z)$为假的概率:1 |
| 用交叉熵作为目标函数 | $1*log[D(x)]对应第一项$ | $1*log[1-D(G(z))]$对应第二项 |
**Note:$D$输出的是概率，那么$D$的输出层的激活函数必须是$sigmoid$**

### 1.3.4. 对抗
判别器$D$的目标
1. 要尽可能把真的样本判断为真，对应最大化第一项：${ E }_{ x ～ { p }_  { data } (x) }[logD(x)]$
2. 把假的样本判断为假，对应最大化第二项：${ E }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z)))] $
- 总之，也就是说**判别器$D$要最大化目标函数**；

生成器$G$的目标
1. 要尽可能的让$D$将生成的假样本判断为真，对应最小化第二项：${ E }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z)))] $

- 总之，也就是说**生成器$G$要最小化目标函数**；

总的来说，这是一个**MinMax Game**；
**Note:实际训练当中，训练$G$的时候$D$的参数是固定的，$G$并不干扰$D$对真实数据的判断，$G$需要$D$的正确引导，$G$只是不断提升自己生成数据的能力。**

### 1.3.5. Loss Function
$D$的损失函数（最小化）：
$$Loss_D = -[1*logD(x) + 1*log(1-D(G(z)))] \tag{3}$$
$G$的损失函数（最小化）：
$$Loss_G = 0*logD(x) + 1*log(1-D(G(z)))=log(1-D(G(z))) \tag{4}$$

### 1.3.6. 具体算法过程
![20190816181118.png](https://image.zkhweb.top/20190816181118.png)

Note：
1. 生成对抗网络的minibatch随机梯度下降训练 
2. 先更新$D$，再更新$G$，只有$D$有了正确的判断能力，$G$才能按照$D$的指示来更新;
3. 可以设置一个超参数k来协调$D$、$G$两者之间更新的次数比例，在实验中k=1，使消耗最小;
4. 在训练$G$的时候$D$的参数要固定，在训练$D$的时候$G$的参数要固定;

---

## 1.4. 改进

### 1.4.1. G替代版的Loss Function

由于$G(z)$是从噪声中生成的样本，所以在最开始$G$生成的样本非常假，很容易被$D$抓出来，也就是说$D(G(z))$非常小,那么$Loss_G = log(1-D(G(z)))$就非常接近0，在反向传播的时候就不能够传播足够的梯度给$G$来更新参数，所以我们从Heuristic的角度来理解：我们本身是要最小化$D$抓出来假样本的概率，现在我们可以换成最大化$D$抓不出来的概率（$\log D(G(z))$），也就是将$G$的损失函数换成：
$$Loss_G=-logD(G(z)) $$
由于$D$是按照：
$$Loss_G = log(1-D(G(z)))$$
训练的，那么如果损失函数更换，这两项不是等价的，所以$D$给出的值就能够提供足够的梯度。

**Note:
$Loss_G =log(1-D(G(z)))$对应的GAN叫做MMGAN
$Loss_G=-logD(G(z)) $对应的GAN叫做NSGAN
改进后的仍然存在些许问题，见与[定理1：全局最优的Note3](#1)**

从函数图像上，可以直观的看出，两种损失函数的梯度变化趋势：
![损失函数图像](https://image.zkhweb.top/损失函数图像)

---

## 1.5. 补充知识
### 1.5.1. 信息量
$ I(x) = -\log {p(x)} = \log { \frac { 1}{ p (x) }  } $
一个事件发生的概率越大，这件事情发生所包含的信息量就越小，比如说一个高富帅追求一个白富美，追到手了没有什么稀奇的，因为这是一件概率很高的事情，但是如果一个矮穷矬追求一个白富美，追到手了，这种事情发生的概率很低，其中一定有其他的原因：比如这个矮穷矬救过白富美的命或者这个矮穷矬器大活好不黏人，所以概率低的事情发生所包含的信息量大；两个相互独立的事情同时发生的信息量是两者单独发生的信息量之和。
### 1.5.2. 信息熵
信息量的均值
$$H(x) = - \sum _{ x } p(x)log p(x) $$
### 1.5.3. 交叉熵
$$H(P, Q) = - \sum _{ x } p(x)log q(x) $$
用估计编码$q(x)$近似真实编码$p(x)$需要的平均编码长度

### 1.5.4. KL散度(Kullback–Leibler散度，相对熵)
统计中的一个概念，时衡量两种概率分布的相似程度，其越小，表示两种概率分布越接近。（当P(x)和Q(x)的相似度越高，KL散度越小）
对于离散的概率分布定义如下：
$$D_{KL}(P||Q)=- \sum _{ x } p(x)log q(x) + \sum _{ x } p(x)log p(x) =H(P, Q)-H(P)$$
对于连续的概率分布定义如下：
$$
D_{K L}(P \| Q)=\int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} d x
$$
想要将一个随机高斯噪声$z$通过一个生成网络G得到一个和

- 性质：
    1. 不对称性
    尽管KL散度从直观上是个度量或距离函数，但它并不是一个真正的度量或者距离，因为它不具有对称性，即$D(P||Q) \not= D(Q||P)$。
    2. 非负性
    相对熵的值是非负值，即$D(P||Q)>0$。

### 1.5.5. JS散度(Jensen-Shannon散度)

$$D_{JS}(P||Q)={\frac{1}{2}} KL(P||M) + {\frac{1}{2}} KL(Q||M) \quad \quad M = {\frac{1}{2}}(P+Q)$$

- 不同于KL主要又两方面：
    1. 值域范围
    JS散度的值域范围是[0,1]，相同则是0，相反为1。相较于KL，对相似度的判别更确切了。
    2. 对称性
    即 JS(P||Q)=JS(Q||P)，从数学表达式中就可以看出。



---

## 1.6. 理论结果

### 1.6.1. 最优判别器D：$D^{*}(x) =\frac{P_{\text {data}}(x)}{P_{\text {data}}(x)+P_{G}(x)}$
对于给定生成器G，最大化$V(D,G)$而得出最优判别器D。原论文中价值函数可写为在$x$上的积分，即将数学期望展开为积分形式：

$$
\begin{aligned}
\max _{ D } V(D,G)&={ E }_{ x ～ { p }_  { data } (x) }[logD(x)] + { E }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z)))]\\
&=\int_{x} p_{d a t a}(x) \log D(x) \mathrm{d} x+\int_{z} p(z) \log (1-D(G(z))) \mathrm{d} z\\
&=\int_{x} p_{d a t a}(x) \log D(x)+p_{G}(x) \log (1-D(x)) \mathrm{d} x
\end{aligned}
$$
取函数，求偏导数
（对于任意的$(a, b) \in \mathbb{R}^{2} \backslash\{0,0\}$,函数$y \rightarrow a \log (y)+b \log (1-y)$在$[0,1]$中的$\frac{a}{a+b}$处达到最大值）
$$
\begin{aligned}
f(D) &=a \log (D)+b \log (1-D) \\
\frac{d f(D)}{d D}&= a \times \frac{1}{D}+b \times \frac{1}{1-D} \times(-1)=0 \\
a \times \frac{1}{D^{*}} &= b \times \frac{1}{1-D^{*}} \\
\Leftrightarrow a \times & (1-D^{*}) =b \times D^{*} & \\
\text{得到最优判别器}&{ D }^{ * }(x)：\\
D^{*}(x) &=\frac{P_{\text {data}}(x)}{P_{\text {data}}(x)+P_{G}(x)}
\end{aligned}
$$

### 1.6.2. 最优生成器：$p_{g}=p_{\text {data }}$
我们知道对于$G$来说，最好的$G$是让：
$${ P }_{ r }(x) = { P }_{ g }(x)$$
此时，有：
$${ D }^{ * }(x)=1/2$$
也就是说最好的生成器使最好的判别器无法判别出来样本是生成样本还是真实样本。

### 1.6.3. 定理1：全局最优
**定理1：当且仅当$p_{g}=p_{\text {data }}$时，$C(G)$达到全局最小。此时，$C(G)$的值为$−log4$。**
注意到，判别器$D$的训练目标可以看作为条件概率$P(Y=y | x)$的最大似然估计，当$y=1$时，x来自于$p_{\text {data }}$；当$y=0$时，$x$来自$p_{g}$。公式1中的极小化极大问题可以变形为： 
$$
\begin{aligned} C(G) &=\max _{D} V(G, D) \\ &=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}\left[\log \left(1-D_{G}^{*}(G(\boldsymbol{z}))\right)\right] \\ &=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \left(1-D_{G}^{*}(\boldsymbol{x})\right)\right] \\ &=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log \frac{p_{\text {data }}(\boldsymbol{x})}{P_{\text {data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \frac{p_{g}(\boldsymbol{x})}{p_{\text {data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right] \\ &=\int_{x} p_{\text {data}}(x) \log \left(\frac{p_{\text {data}}(x)}{p_{\text {data}}(x)+p_{g}(x)}\right)+p_{g}(x) \log \left(\frac{p_{g}(x)}{p_{\text {data}}(x)+p_{g}(x)}\right) d x\\&=\int_{x} p_{d a t a}(x) \log \left(\frac{p_{d a t a}(x)}{\frac{p_{d a t a}(x)+p_{g}(x)}{2}}\right)+p_{g}(x) \log \left(\frac{p_{g}(x)}{\frac{p_{d a t a}(x)+p_{g}(x)}{2}}\right) d x-\log (4)\\
&=\underbrace{K L\left(p_{\text {data}}(x) \| \frac{p_{\text {data}}(x)+p_{g}(x)}{2}\right)}_{\geq 0}+\underbrace{K L\left(p_{g}(x) \| \frac{p_{\text {data}}(x)+p_{g}(x)}{2}\right)}_{\geq 0}-\log (4)\\&=2\underbrace{\cdot JSD(p_{data}\|p_{g})}_{\geq 0}-log(4)\\\min _{G} C(G)&=0+0-\log (4)=-\log (4)
\end{aligned}$$
当且仅当$p_{\text {data}}(x)=\frac{p_{\text {data}}(x)+p_{g}(x)}{2}$即$p_{g}=p_{\text {data }}$时成立，此时$C(G)$达到全局最小，$C(G)$的值为$−log4$。
**Note1**
$KL$散度：$KL({ P }_{ 1 }||{ P }_{ 2 })={ P }_{ 1 }\log { \frac { { P }_{ 1 } }{ { P }_{ 2 } }  } $
$JS$散度：$ JS({ P }_{ 1 }||{ P }_{ 2 })=\frac { 1 }{ 2 } KL({ P }_{ 1 }||\frac { { P }_{ 1 }+{ P }_{ 2 } }{ 2 } )+\frac { 1 }{ 2 } KL({ P }_{ 2 }||\frac { { P }_{ 1 }+{ P }_{ 2 } }{ 2 } ) $
**Note2（MMGAN）**$Loss_G =log(1-D(G(z)))$
当判别器$D$最优的时候，生成器$G$是在减小真实分布与生成分布之间的$JS$散度
**<span id="1">Note3（NSGAN）</span>**$Loss_G=-logD(G(z)) $ 
$$\begin{aligned}
KL({ P }_{ g }(x)||{ P }_{ r }(x))
&={ P }_{ g }(x)*\log { \frac { { P }_{ g }(x) }{ { P }_{ r }(x) }  } \\ 
&={ P }_{ g }(x)*\log { \frac { { P }_{ g }(x)/({ P }_{ r }(x)+{ P }_{ g }(x)) }{ { P }_{ r }(x)/({ P }_{ r }(x)+{ P }_{ g }(x)) }  } \\ 
&={ P }_{ g }(x)*\log  \frac { 1-D^{ * }(x) }{ D^{ * }(x) } \\ 
&={ P }_{ g }(x)log[1-D^{ * }(x)]-{ P }_{ g }(x)logD^{ * }(x)\\
-{P}_{g}(x)*logD^*(x)&=KL({ P }_{ g }(x)||{ P }_{ r }(x))-{ P }_{ g }(x)log[1-D^{ * }(x)]\\
Loss_{ G }&=KL({ P }_{ g }(x)||{ P }_{ r }(x))-{ P }_{ g }(x)log[1-D^{ * }(x)]\\
\because {P}_{r}(x)*log[D^*(x)] &+ {P}_{g}(x)*log[1-D^*(x)]=2JS({ P }_{ r }||{ P }_{ g })-2log2\\
\therefore Loss_{ G }=KL({ P }_{ g }(&x)||{ P }_{ r }(x))-2JS({ P }_{ r }||{ P }_{ g })+{P}_{r}(x)*log[D^*(x)]+2log2[1-D^{ * }(x)]
\end{aligned}$$
**从上面的式子可以看出KL散度和JS散度同时存在且方向相反，而JS散度和KL散度都是衡量两个分布距离的度量，且是单调性同步的函数，这样的话就会导致梯度的方向不稳定，一会儿上升一会儿下降，所以这个替代版的损失函数也不是一个好的选择。**

### 1.6.4. 算法的收敛性
**命题：如果$G$和$D$有足够的性能，对于算法1中的每一步，给定$G$时，判别器能够达到它的最优，并且通过更新$p_g$来提高这个判别准则。 
$$\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \left(1-D_{G}^{*}(\boldsymbol{x})\right)\right]$$
则$p_g$收敛为$p_{data}$。**

<!-- 证明.如上述准则，考虑$V(G, D)=U\left(p_{g}, D\right)$为关于$p_{g}$的函数。注意到$U\left(p_{g}, D\right)$为$p_g$的凸函数。该凸函数上确界的一次导数包括达到最大值处的该函数的导数。换句话说，如果$f(x)=\sup _{\alpha \in \mathcal{A}} f_{\alpha}(x)$且对于每一个$α$，$f_α(x)$ 是关于$x$的凸函数，那么如果$\beta=\operatorname{argsup}_{\alpha \in \mathcal{A}} f_{\alpha}(x)$，则$\partial f_{\beta}(x) \in \partial f$。这等价于给定对应的$G$和最优的$D$，计算$p_g$的梯度更新。如定理1所证明，$\sup _{D} U\left(p_{g}, D\right)$是关于$p_g$的凸函数且有唯一的全局最优解，因此，当$p_g$的更新足够小时，$p_g$收敛到$p_x$，证毕。 -->

**Note**
优化$θ_g$而不是$p_g$本身

---

## 1.7. 优势和劣势
### 1.7.1. 优势
- 根据实际的结果，它们看上去可以比其它模型产生了更好的样本（图像更锐利、清晰）。
- 生成对抗式网络框架能训练任何一种生成器网络（理论上-实践中，用 REINFORCE 来训练带有离散输出的生成网络非常困难）。大部分其他的框架需要该生成器网络有一些特定的函数形式，比如输出层是高斯的。重要的是所有其他的框架需要生成器网络遍布非零质量（non-zero mass）。生成对抗式网络能学习可以仅在与数据接近的细流形（thin manifold）上生成点。
- 不需要设计遵循任何种类的因式分解的模型，任何生成器网络和任何鉴别器都会有用。
- 无需利用马尔科夫链反复采样，无需在学习过程中进行推断（Inference），回避了近似计算棘手的概率的难题。

### 1.7.2. 劣势
- 解决不收敛（non-convergence）的问题。 
目前面临的基本问题是：所有的理论都认为 GAN 应该在纳什均衡（Nash equilibrium）上有卓越的表现，但梯度下降只有在凸函数的情况下才能保证实现纳什均衡。当博弈双方都由神经网络表示时，在没有实际达到均衡的情况下，让它们永远保持对自己策略的调整是可能的【OpenAI Ian Goodfellow的Quora】。

- 难以训练：崩溃问题（collapse problem） 
GAN模型被定义为极小极大问题，没有损失函数，在训练过程中很难区分是否正在取得进展。GAN的学习过程可能发生崩溃问题（collapse problem），生成器开始退化，总是生成同样的样本点，无法继续学习。当生成模型崩溃时，判别模型也会对相似的样本点指向相似的方向，训练无法继续。[Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)

- 无需预先建模，模型过于自由不可控。 
与其他生成式模型相比，GAN这种竞争的方式不再要求一个假设的数据分布，即不需要formulate p(x)，而是使用一种分布直接进行采样sampling，从而真正达到理论上可以完全逼近真实数据，这也是GAN最大的优势。然而，这种不需要预先建模的方法缺点是太过自由了，对于较大的图片，较多的 pixel的情形，基于简单 GAN 的方式就不太可控了(超高维)。在GAN[Goodfellow Ian, Pouget-Abadie J] 中，每次学习参数的更新过程，被设为D更新k回，G才更新1回，也是出于类似的考虑。

---

## 1.8. 结论和未来研究方向
该框架允许许多直接的扩展：

- 条件生成模型$p(x | c)$可以通过将$c$作为$G$和$D$的输入来获得。
- 给定$x$，可以通过训练一个任意的模型来学习近似推理，以预测$z$。这和wake-sleep算法训练出的推理网络类似，但是它具有一个优势，就是在生成器训练完成后，这个推理网络可以针对固定的生成器进行训练。
- 能够用来近似模型所有的条件概率$p\left(\boldsymbol{x}_{S} | \boldsymbol{x}_{\beta}\right)$，其中$S$通过训练共享参数的条件模型簇的关于$x$索引的一个子集。本质上，可以使用生成对抗网络来随机拓展MP-DBM。
- 半监督学习：当标签数据有限时，判别网络或推理网络的特征不会提高分类器效果。
- 效率改善：为协调$G$和$D$设计更好的方法，或训练期间确定更好的分布来采样$z$，能够极大的加速训练。

---

## 1.9. 参考资料
[Paper---Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
[知乎---GAN入门理解及公式推导](https://zhuanlan.zhihu.com/p/28853704)
[CSDN---GAN论文阅读——原始GAN（基本概念及理论推导）](https://blog.csdn.net/stalbo/article/details/79283399)
[CSDN---KL散度、JS散度以及交叉熵对比](https://blog.csdn.net/FrankieHello/article/details/80614422)
[CSDN---Generative Adversarial Nets论文笔记+代码解析](https://blog.csdn.net/wspba/article/details/54582391)
[CSDN---Generative Adversarial Nets（译）](https://blog.csdn.net/wspba/article/details/54577236)
[Github---andyhujinzhao/Generative_Adversarial_Nets](https://github.com/andyhujinzhao/Generative_Adversarial_Nets)

---