<!-- toc -->
[toc]

---

# 1. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
>arXiv:1511.06434 [cs]
>tensorflow2代码：https://github.com/zhangkaihua88/ML_Paper

---

# 2. 总结
- 要解决什么问题
    - 结合CNN和GAN，提出了具体的实现细节和技巧
    - 对CNN结果进行可视化，帮助理解CNN过程
- 用什么方法解决
    - 通过CNN构建GAN中的生成器和判别器
    - 在CNN具体实现中，提出了一些改进方案，提高稳定性
- 还存在什么问题
    - 稳定性差——GAN通病

---

# 3. 引言
- 提出并评估了一系列卷积GAN体系结构拓扑上的约束条件，这些约束条件使得它们在大多数情况下可以稳定地训练。我们将这种架构称为Deep Convolutional GANs（DCGAN）
- 使用图像分类任务上训练出来的判别器和其他的非监督算法做了比较
- 对GAN学习到的特征做出了可视化，并经验性的证明了特殊的特征表征了特殊的对象
- 针对生成器，我们提出了一个很有趣的算法向量，这个向量能很简单的在语义层面上操作生成样例的质量

---

# 4. 相关工作
- 无监督的表征学习
    - 一个经典的非监督表征学习手段是做出数据聚类，之后利用聚类结果来改善分类结果。
    - 训练自编码器（卷积式的自编码器）都能将图像作成紧编码，并且尽可能的通过解码器还原图像
        - 分离编码中向量的意义和位置
        - 分析编码的梯度结构

- 生成自然图像
    - 参数化领域
        是在图像数据库下做匹配，经常对成批的图像做匹配，它在纹理合成，超分辨率重建和in-paiting中用的较多。
    - 非参数化领域

- CNN内部的可视化
使用反卷积，过滤最大激活，可以逼近网络中每一个卷积滤波器的结果

---

# 5. DCGAN构建方法(CNN)
- 全卷积网络
    使用**逐步卷积替代确定性的空间池化函数**,允许网络学习自身上采样(upsampling)或下采样(downsampling)方式（生成器G/判别器D）。在网络中，所有的pooling层使用步幅卷积(判别网络)和微步幅度卷积(生成网络)进行替换。
- 在卷积特征之上消除全连接层
    例如全局平均池化，全局平均pooling增强了模型稳定性，但减缓了收敛速度
- 批量归一化(Batch Normalization)
    将每个单元的输入都标准化为0均值与单位方差
    - 改进了训练问题
    - 缓解了深层网络中的梯度溢出问题
    
    但实际上，这种方法在深层的生成器中被证明是不适用的，它会**导致生成器反复震荡生成单点数据**。但是，将所有层都进行BN，会导致样本震荡和模型不稳定，所以，**不要在生成器的输出层和判别器的输入层上使用BN**。
- 激活函数
    生成器：除了最终输出层使用`Tanh`，其他都使用`Relu`
    判别器：都是用`leaky relu`(leaky rectified activation)

**稳定DCGAN的架构指导：**
- 判别器中，使用带步长的卷基层来替换所有pooling层，生成器中使用小步长卷积来代替pooling层。 
- 在生成器和判别器中使用BN。 
- 去除深度架构中的全连接隐藏层。
- 生成器中，除去最后一层使用Tanh之外，每一层都使用ReLU来激活。 
- 判别器中，每一层都使用LeakReLU来激活。

---

# 6. 对抗式训练的细节
- 基本对原始数据都不进行数据增强，只是将像素值变换到[-1,1]之间（与生成器最终输出层的tanh对应）
- 使用Adam作为优化器，初始学习率0.0002，beta_1=0.5
- leaky relu=0.2

判别器的网络构造
![20200212214025.png](https://image.zkhweb.top/20200212214025.png)
**常用验证unsupervised representation learning algorithms** 的方法是：
选择某个监督学习数据集，使用训练好的模型输入数据提取特征，使用线性模型用于监督数据集任务，查看性能。

---

# 7. 参考资料
[Paper---Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)
[CSDN---论文翻译——无监督DCGAN做表征学习](https://blog.csdn.net/xiening0618/article/details/79417734)
[CSDN---DCGAN论文译本](https://blog.csdn.net/qq_40667584/article/details/79690043)
[CSDN---DCGAN论文笔记+源码解析](https://blog.csdn.net/wspba/article/details/54730871)
[CSDN---GAN论文阅读——DCGAN](https://blog.csdn.net/stalbo/article/details/79359095)
[知乎---精读深度学习论文(26) DCGAN](https://zhuanlan.zhihu.com/p/40126869)
[个人Blog---Transposed Convolution, Fractionally Strided Convolution or Deconvolution](https://buptldy.github.io/2016/10/29/2016-10-29-deconv/)