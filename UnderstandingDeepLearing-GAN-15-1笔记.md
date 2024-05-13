# C15 GAN 对抗生成网络
A generative adversarial network or GAN（生成对抗网络）is **an unsupervised model（无监督模型）** that aims to generate new samples that are indistinguishable from a set of training examples. (其目的是生成与一组训练示例无异的新样本。)
**基本思想**：主生成器网络(generator network)通过将随机噪声映射到输出数据空间来创建样本。如果第二个鉴别器网络(discriminator network)不能区分生成的样本和真实的样本，那么样本是合理的。如果这个网络能分辨出区别，这就提供了一个训练信号，以提高样本的质量。
**GAN思想简单但训练很困难**：由于学习算法不稳定，尽管GANs可以学习生成真实的样本，但不能生成所有可能的样例。
**GAN在图像领域取得最大成功**，因此本章例子主要关注合成图像。
## 15.1  Discrimination as a signal
目的是从一组真实训练数据${x_i}$的相同分布中生成新样本${x^*_j}$。
新样本${x^*_j}$生成步骤：
> (i)从简单的基分布，例如标准正态分布选择一个潜在变量${z_j}$；
(ii)通过一个参数为${\theta}$的网络${x^*=g[z_j,\theta]}$传递此数据。

上述网络称为**生成器**，目标是学习出参数${\theta}$，使得新样本${x^*_j}$与真实数据${x_i}$“相似”。
“相似”怎么定义？引出第二个网络${f[\cdot,\phi]}$(参数为${\phi}$)，称为**判别器**。判别器的目标是对输入的样本进行分类，判别其是真的还是生成的数据。若判别器辨别不出来是真是假，说明生成了合理的样本，成功了。**判别器提供了改进生成过程的信号**
图15.1说明了这个策略。
![图15.1 GAN mechanism](/picture/udl-15-1.png#pic_center)
### 15.1.1 GAN loss function
设判别器为${f[x,\phi]}$，即样本x作为输入，参数为${\phi}$，返回该样本是真实的可能性。其本质上是一个**二分类问题**，因此采用**二分类交叉熵损失函数**(最小化)。
>二分类交叉熵损失函数原式表达为：
 ![式15.2](/picture/udl-式15-2.png#pic_center)

这里设真实的例子x有标签y = 1，生成的样本${x^∗}$有标签y = 0，这样上面式子变为：
![式15.3](/picture/udl-式15-3.png#pic_center)

而对于训练生成器${x^*_j=g[z_j,\theta]}$中的参数${\theta}$，应使其最大化，因为希望生成的样本被错误地分类。
![式15.4](/picture/udl-式15-4.png#pic_center)
### 15.1.2 Training GANs
根据上面式子，判别器参数φ使损失函数最小，生成器参数θ使损失函数最大化，可以看出这是一个minmax游戏。因此解决方案是一个**纳什均衡**。
为了训练GAN，将式15.4分为两个损失函数：
![式15.5](/picture/udl-式15-5.png#pic_center)
这样都是最小化了。这样可以在两个损失函数上进行梯度下降。
![图15.2 GAN loss functions](/picture/udl-图15-2.png#pic_center)


### 15.1.3 Deep convolutional GAN (DCGAN)
深度卷积GAN或DCGAN是早期专门用于图像的GAN生成架构（图15.3）。
![图15.3 DCGAN architecture](/picture/udl-图15-3.jpg#pic_center)
训练结束后，判别器被丢弃不用，只用训练好的生成器来创建新的样本(例子如图15.4所示)。
![图15.4 来自DCGAN模型的合成图像](/picture/udl-图15-4.png#pic_center)

### 15.1.4 Difficulty training GANs
**GAN的理论相当简单，但是出了名的难以训练。** 因为要训练GANs,就要有：
> (i)使用分支卷积来进行上采样和下采样;
(ii)除了最后一层和第一层外，分别在生成器和判别器中使用批规范(BatchNorm);
(iii)在判别器中使用LeakyReLU激活函数;
(iv)使用Adam优化器，但其动量系数比通常**更低**。

深度学习模型对于上述每个步骤都要robust...就很困难。
一个常见的错误就是生成器提供了可信的样本，但这些样本只代表了数据的一个子集(例如，对于面孔，生成器可能永远不会产生有胡子的面孔。)，这种情况称为**mode dropping**(模式下降)。在此之下一种极端现象就是生成器完全或大多忽略潜在变量z，并将所有样本压缩为一个或几个点，这种现象称为**mode collapse**(模式崩溃)，如图15.5所示。
![图15.5 模式崩溃](/picture/udl-图15-5.png#pic_center)


非概率模型
关注Prx
扩散模型容易些
# 2 VAEs 变分自编码器
概率生成模型
V：变分。
AE：自编码 x->z->x
- 编码x->z
- 解码z->x'/x
VAE是辅助学习Pr(x)的神经架构
最后的Pr(x)可以描述为：**非线性潜在变量模型**。没有变分，也没有自编码器。
## 2.1 潜在变量模型
$\alpha$
设变量为高斯分布
## 2.2 非线性潜在变量模型
救命
## 2.3 训练
学数学
最大化似然对数方法


