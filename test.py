# coding=gbk

from model import GraphSAGE
from utils import *


#---------------------------------设置参数-------------------------------------
dataset='cora'                              # 数据集：'cora'/ 'citeseer'/ 'pubmed'

if dataset == 'cora':
    total_sample_num = 2703                 # 总数据量
    feature_dim = 1433                      # 特征维度
    label_dim = 7                           # 标签维度
if dataset == 'citeseer':
    total_sample_num = 3312 
    feature_dim = 3703 
    label_dim = 6
if dataset == 'pubmed':
    ...

dim_rounds = [128, label_dim]               # GraphSAGE各轮聚合器的输出维度
aggr_method_rounds=['gcn', 'mean']          # GraphSAGE各轮聚合器的聚合方法
activation_rounds=['relu', 'softmax']       # GraphSAGE各轮聚合器的激活函数
use_bias_rounds=[True, True]                # GraphSAGE各轮聚合器是否用偏置
sample_nums = [4, 3]                        # GraphSAGE各阶邻居采样数
optimizer='adam'                            # GraphSAGE优化函数
loss='categorical_crossentropy'             # GraphSAGE损失函数
metrics=['acc']                             # GraphSAGE评价指标

test_size = [0.8, 1]                        # 测试数据量，整个数据集最后20%部分
batch_size = 64                             # 测试批次大小               
#-----------------------------------------------------------------------------


#---------------------------------设置路径-------------------------------------
if dataset == 'cora':
    data_path = "D:/科研/python代码/炼丹手册/GraphSAGE/datasets/cora"
    load_path = "D:/科研/python代码/炼丹手册/GraphSAGE/save_models/gsage_cora.h5"
if dataset == 'citeseer':
    data_path = "D:/科研/python代码/炼丹手册/GraphSAGE/datasets/citeseer"
    load_path = "D:/科研/python代码/炼丹手册/GraphSAGE/save_models/gsage_citeseer.h5"
if dataset == 'pubmed':
    data_path = "D:/科研/python代码/炼丹手册/GraphSAGE/datasets/pubmed"
    load_path = "D:/科研/python代码/炼丹手册/GraphSAGE/save_models/gsage_pubmed.h5"
#-----------------------------------------------------------------------------


#--------------------------------加载数据集-------------------------------------
if dataset == 'cora':
    A,X,Y = load_cora(data_path)
if dataset == 'citeseer':
    A,X,Y = load_citeseer(data_path)
if dataset == 'pubmed':
    ...
test_index = [i for i in range(int(len(X)*test_size[0]), int(len(X)*test_size[1]))]
test_data_gen = GraphSAGE_DataGenerator(
    x_set=X,
    y_set=Y,
    adj_matrix = A,
    batch_size=batch_size,
    sample_nums=sample_nums,
    indexes=test_index,
    shuffle=False
    )
#-----------------------------------------------------------------------------


#---------------------------------搭建模型-------------------------------------
sage = GraphSAGE(
    dim_rounds = dim_rounds,
    aggr_method_rounds = aggr_method_rounds,
    activation_rounds = activation_rounds,
    use_bias_rounds = use_bias_rounds,
    )

samples = [1]+sample_nums
input_shape = [(None, np.prod(samples[:i+1]), feature_dim) for i in range(len(samples))]

sage.build(input_shape)
sage.summary()
sage.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metrics
    )
#-----------------------------------------------------------------------------


#--------------------------------读取和测试-------------------------------------
sage.load_weights(load_path)
sage.evaluate(
    x = test_data_gen
    )
#-----------------------------------------------------------------------------