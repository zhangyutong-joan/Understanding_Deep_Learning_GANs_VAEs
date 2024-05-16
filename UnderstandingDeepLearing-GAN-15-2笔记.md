# C15 GAN 对抗生成网络
## 15.2 Improving stability
为什么GANs很难训练->原因在于其**损失函数**。
### 15.2.1 Analysis of GAN loss function
![15-2-1公式们](/picture/udl-15-2-1公式们.jpg#pic_center)
> 在式15.9中，对于第一项，若${Pr(x^*)}$越高、${(Pr(x^*)+Pr(x))/2}$越高，第一项就越小。换句话说，它惩罚的是**有生成样本${x^∗}$但没有真实样本x**的区域。它强调质量(quality)。
>对于第二项，若${Pr(x)}$越高、${(Pr(x^*)+Pr(x))/2}$越高，第二项就越小。换句话说，它惩罚**有真实样本但没有生成样本**的区域。它强调覆盖(coverage).

从公式15.6可以看出，第二项并不依赖于生成器，因此，它并不关心覆盖范围(coverage)。这导致了模式下降(mode dropping,只生成了真实数据的子集)。

### 15.2.2 Vanishing gradients (梯度消失问题)
15.2.1节说明了**当鉴别器是最优的时，损失函数使生成的样本和真实样本之间的距离的度量最小化。**
但使用上式定义的损失函数存在一个潜在问题：当真实数据的概率分布与生成数据的概率分布**完全不相交**时，怎么改变生成器都不能降低损失函数。也就是说，**当判别器能够完美地分离生成的和真实的样本，生成器生成的数据无论怎么微调，判别器的分类得分都不会因此而有所提高**，因为判别器已经能够完美区分真假数据（图15.6）。
![图15.6 Problem with GAN loss function.](/picture/udl-图15-6.png#pic_center)

而真的存在这种情况(图15.7)。简而言之，在判别器和生成的质量之间应该有一个很好的平衡；如果判别器变得太好了，生成器的训练更新就会减弱。
![图15.7 Vanishing gradients in the generator of a DCGAN.](/picture/udl-图15-7.png#pic_center)

### 15.2.3 Wasserstein distance (Wasserstein距离)
> 根据前面所说，
> (i)GAN损失函数可以用概率分布之间的“距离”来定义；
> (ii)当生成的样本太容易与实际样本区分时，这个“距离”的梯度将变为0。
> **如何解决梯度消失的问题？**

最容易想到的方法就是选择另一个具有更好属性的距离度量：**Wasserstein距离**。即使两个分布是不相交的，Wasserstein距离也是well-defined的，当两个分布彼此靠近时也会平滑地减小。

### 15.2.4 Wasserstein distance for discrete distributions(离散分布下的Wasserstein距离)
### 15.2.5 Wasserstein distance for continuous distributions(连续分布下的Wasserstein距离)
概率论跳过啊啊
### 15.2.6 Wasserstein GAN loss function
对于判别器${f[x,\phi]}$，通过优化参数${\phi}$来最大化f[x]，近似积分：
![式15.16](/picture/udl-式15-16.png#pic_center)

其中，必须约束判别器${f[x_i,\phi]}$在每个位置x上的绝对梯度范数小于1：
![式15.17](/picture/udl-式15-17.png#pic_center)

为了实现这一点，一种方法是将鉴别器的权重缩小到一个小的范围内（例如，[−0.01,0.01]）。
另一种选择是**the gradient penalty Wasserstein GAN or WGAN-GP**，它增加了一个正则化项，随着梯度范数偏离统一而增加。




