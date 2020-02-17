<!-- toc -->
[toc]

---


# 1. Conditional Generative Adversarial Nets
>arXiv:1411.1784 [cs.LG]
>tensorflow2代码：https://github.com/zhangkaihua88/ML_Paper

---

## 1.1. 摘要
在GAN的基础上引入标签y，同时使用在生成器和判别器中.
可以应用于[多模态模型](https://www.jiqizhixin.com/graph/technologies/4468592f-93e9-4575-be91-fd64c0c6afe0)中。

---

## 1.2. 引言
生成对抗网络
- 规避了棘手的概率计算
- 不需要使用马尔科夫链，仅使用反向传播算法去获得梯度
- 训练时不需要推断，可以轻松的将各种因素和相互作用纳入模型

但**无条件生成模型无法控制生成的数据**，给模型加入附加信息可以知道数据的生成

---

## 1.3. 条件对抗网络

### 1.3.1. 符号定义
$x$ $\rightarrow$ 真实数据
$y$ $\rightarrow$ 标签（辅助信息）
$z$ $\rightarrow$ 噪音（生成器的输入数据）
$p_x$ $\rightarrow$ 真实数据的分布
$p_{z}(z)$ $\rightarrow$ 原始噪音数据的分布
$p_g$ $\rightarrow$ 经过生成器后数据的分布
$G()$ $\rightarrow$ 生成映射函数（可微），结构为多层感知机，参数$\theta_{g}$
$D()$ $\rightarrow$ 判别映射函数（可微），结构为多层感知机，参数$\theta_{d}$
$G(z;\theta_{g})$ $\rightarrow$ 将噪音$z$映射到新的数据空间
$D(x ; \theta_{d})$ $\rightarrow$ $x$来自真实数据而不是生成数据的概率（真=1，假=0）

### 1.3.2. 知识回顾——生成对抗网络
生成器G，判别器D，相互对抗使目标函数，达到最优。
$$ \min _{G}\max _{ D } V(D,G)={ \mathbb{E} }_{ x ～ { p }_  { data } (x) }[logD(x)] + { \mathbb{E} }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z)))]$$

### 1.3.3. 直观感受
![20200205211839.png](https://image.zkhweb.top/20200205211839.png)

### 1.3.4. 目标函数
$$ \min _{G}\max _{ D } V(D,G)={ \mathbb{E} }_{ x ～ { p }_  { data } (x) }[logD(x|y)] + { \mathbb{E} }_{ z ～ { p }_{ z }(z) }[log(1-D(G(z|y)))] $$
和原始GAN相近，只是G，D在y的条件下生成或判别

### 1.3.5. 对抗
生成器G通过z，y联合生成图片
判别器D在y的条件下判别G(z)
主要是在y条件下的MinMax Game

--- 

## 1.4. 实验
### 1.4.1. 单模式——MNIST

### 1.4.2. 多模式——MIR Flickr

---

## 1.5. 参考资料
[Paper---Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
[知乎---《Conditional Generative Adversarial Nets》阅读笔记](https://zhuanlan.zhihu.com/p/23648795)
[CSDN---Conditional Generative Adversarial Nets论文翻译](https://blog.csdn.net/Chaolei3/article/details/78870858)
[CSDN---Conditional Generative Adversarial Nets论文笔记](https://blog.csdn.net/wspba/article/details/54666907)
