# Keras-GraphSAGE
基于Keras搭建一个GraphSAGE，用cora数据集和citeseer数据集对GraphSAGE进行训练，完成模型的保存和加载对节点分类测试。


环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />

注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：将数据集文件解压至此<br />
2. /save_models：保存训练好的模型权重文件<br />

GraphSAGE概述<br />
图神经网络(Graph Neural Network, GNN)是指神经网络在图上应用的模型的统称，根据采用的技术不同和分类方法的不同，
又可以分为下图中的不同种类，例如从传播的方式来看，图神经网络可以分为图卷积神经网络（GCN），图注意力网络（GAT），Graph LSTM等等<br />
GraphSAGE 是 2017 年提出的一种图神经网络算法，解决了 GCN 网络的局限性: GCN 训练时需要用到整个图的邻接矩阵，依赖于具体的图结构，一般只能用在直推式学习 Transductive Learning。GraphSAGE 使用多层聚合函数，每一层聚合函数会将节点及其邻居的信息聚合在一起得到下一层的特征向量，GraphSAGE 采用了节点的邻域信息，不依赖于全局的图结构。<br /><br />

GraphSAGE 包含采样和聚合 (Sample and aggregate)，首先使用节点之间连接信息，对邻居进行采样，
然后通过多层聚合函数不断地将相邻节点的信息融合在一起。用融合后的信息预测节点标签。<br /><br />

GraphSAGE 提供了四种聚合节点的函数：<br />
1.Mean aggregator: 对节点 v 进行聚合时，对节点 v 和邻域的特征向量求均值。<br />
2.GCN aggregator: 采用了类似 GCN 卷积的方式进行聚合。<br />
3.LSTM aggregator: 使用了 LSTM 进行聚合，但是因为节点之间没有明显的顺序关系，因此会打乱之后放入 LSTM。<br />
4.Pooling aggregator: 先把所有邻居节点的特征向量传入一个全连接层，然后使用 max-pooling 聚合。<br /><br />

数据集：<br />
cora：包含2708篇科学出版物网络，共有5429条边，总共7种类别。<br />
数据集中的每个出版物都由一个 0/1 值的词向量描述，表示字典中相应词的缺失/存在。 该词典由 1433 个独特的词组成。<br />
链接：https://pan.baidu.com/s/1u7v3oJcTvnFWAhHdSLHwtA?pwd=52dl 提取码：52dl<br />
citeseer：包含3312个节点，4723条边构成的引文网络。标签共6个类别。数据集的特征维度是3703维。<br />
链接：https://pan.baidu.com/s/11n2AQCVSV6OevSkUhYWcNg?pwd=52dl 提取码：52dl<br /><br />
