# support-tensor-machine
Try to realize the algorithm "support tensor machine"

If you have any problem ， you can ask me via my e-mile: 694789266@qq.com

## 总结一些网友的提问

### 网友A对HOSTM.m的疑问
在HOSTM.m文件中，网友A反应改程序有问题，大意为 “程序主要的问题是使用了三个没有被定义的函数，具体的为原代码中的第53，65，73行（我已在附件提供您的原代码以及注释）能否说明这几个函数的作用以及实现思路”
我对应的回答：
1.	53行的frob函数其含义应该是求一个张量（V）的F范数，具体的计算为：V中所有元素的平方和的算术平方根，其意义应该是用来计算新旧张量的距离差距，我这里用了F范数进行度量，以便得到迭代的停止条件。
2.	65行的tmprod函数应该是张量的k-mode积，假设张量V是三维的，当张量V与一个向量a进行方向k的k-mode积时，能够起到降维的作用，令W=k-mode（V,a），则W将变为二维的，也即矩阵，再次将W与某个向量b进行某个方向的k-mode积，能够降到1维，令M = k-mode（W,a）,此时由于W是矩阵，a是向量，K-mode积将退化成矩阵与向量的乘法，即M=k-mode(W,a)=W · a. 这里编程的思路应该是将X个训练集的张量数据（假设其维度维n），通过n-1次k-mode积降成1维的向量，得到X个向量数据，然后用传统的SVM的方法，计算这样得到的SVM的系数，其表现为某个向量。从而进行更新k-mode系数。程序的目的其实就是求k-mode系数，它的含义相当于分类为题  f(x) = u’·x的系数u。
3.	73行的outprob，其含义应该是张量的外积，这里不同于线性代数，或者解析几何里的张量积，他的作用更像于通过某种运算，将几个低维的向量，结合成高维的张量，其作用效果应该类似于 (1,2,3)T · (1,2,3) = F, 即通过向量（1,2,3）得到了矩阵F的这种过程。这里的张量外积，应该是通过n-1个向量，组合成一个n-1维张量，然后计算其范数，进而为求该次循环中的目标k-mode系数进行某种标准化，请查看73-75行，outprob的相关变量的流动方向。我现在想想，觉得这条或许并不重要，或许我们直接计算出目标k-mode系数的所有其余k-mode的平方和的算术平方根应该能够达到同样的效果。

以及我的总结：高阶支持张量积的算法的编程思路完全类似于STM（支持张量积）的编程思路，区别在于高阶的张量为了降维使用了k-mode积这个映射，而STM只要使用矩阵与向量的一般乘法即可完成降维。

### 网友B反应代码不全
我对应的回答：其实你能在这个项目上找到的所有代码同样是我拥有的所有代码，包括HOSTM.m即matlab版本的高阶支持张量积和python版本的二阶支持张量积，但是你可根据编程需要，自己更改相关的代码，达到想要的效果。额外说明：当初我读研时，只想着完成算法以便完成实验，顺便玩一下github，没有想的那么系统，把所有实验项目相关的东西都传到网上。几年过后，原先计算机也更换了，很多东西都遗失了。因此很遗憾，我本人也只能凭借记忆说明一些代码的作用。

## 声明：
我本人非常乐意您使用我的代码进行您的实验研究，如果这对您的实验有所帮助，将是我这些代码的价值与意义。
同样希望，当您成功改进代码时，也可联系我，我愿意在本项目处放上链接，链接到您的代码，一起把算法变得更好。
如果您成功使用了我的代码进行实验，也希望您在引用中进行说明，一个脚注或者网页文献的引用。这对算法的传播将起到巨大的作用。
最后，如果代码仍有问题，也请联系我，我将在力所能及的范围内帮助您。
