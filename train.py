# coding=gbk

from model import GraphSAGE
from utils import *


#---------------------------------设置参数-------------------------------------
dataset='pubmed'                            # 数据集：'cora'/ 'citeseer'/ 'pubmed'

if dataset == 'cora':
    total_sample_num = 2703                 # 总数据量
    feature_dim = 1433                      # 特征维度
    label_dim = 7                           # 标签维度
if dataset == 'citeseer':
    total_sample_num = 3312 
    feature_dim = 3703 
    label_dim = 6
if dataset == 'pubmed':
    total_sample_num = 19717 
    feature_dim = 500 
    label_dim = 3

dim_rounds = [128, label_dim]               # GraphSAGE各轮聚合器的输出维度
aggr_method_rounds=['gcn', 'mean']          # GraphSAGE各轮聚合器的聚合方法
activation_rounds=['relu', 'softmax']       # GraphSAGE各轮聚合器的激活函数
use_bias_rounds=[True, True]                # GraphSAGE各轮聚合器是否用偏置
sample_nums = [6, 3]                        # GraphSAGE各阶邻居采样数
optimizer='adam'                            # GraphSAGE优化函数
loss='categorical_crossentropy'             # GraphSAGE损失函数
metrics=['acc']                             # GraphSAGE评价指标

train_size = [0, 0.5]                       # 训练数据量，整个数据集的前50%部分
valid_size = [0.5, 0.8]                     # 验证数据量，整个数据集的50%-80%部分
batch_size = 64                             # 训练批次大小
epochs = 5                                  # 训练轮数                                                              
steps_per_epoch = total_sample_num*(        # 每轮训练步数，不超过训练样本量//batch_size
    train_size[1]-train_size[0])//batch_size                      
#-----------------------------------------------------------------------------


#---------------------------------设置路径-------------------------------------
if dataset == 'cora':
    data_path = "D:/科研/python代码/炼丹手册/GraphSAGE/datasets/cora"
    save_path = "D:/科研/python代码/炼丹手册/GraphSAGE/save_models/gsage_cora.h5"
if dataset == 'citeseer':
    data_path = "D:/科研/python代码/炼丹手册/GraphSAGE/datasets/citeseer"
    save_path = "D:/科研/python代码/炼丹手册/GraphSAGE/save_models/gsage_citeseer.h5"
if dataset == 'pubmed':
    data_path = "D:/科研/python代码/炼丹手册/GraphSAGE/datasets/pubmed"
    save_path = "D:/科研/python代码/炼丹手册/GraphSAGE/save_models/gsage_pubmed.h5"
#-----------------------------------------------------------------------------


#--------------------------------加载数据集-------------------------------------
if dataset == 'cora':
    A,X,Y = load_cora(data_path)
if dataset == 'citeseer':
    A,X,Y = load_citeseer(data_path)
if dataset == 'pubmed':
    A,X,Y = load_pubmed(data_path)

train_index = [i for i in range(int(len(X)*train_size[0]), int(len(X)*train_size[1]))]
valid_index = [i for i in range(int(len(X)*valid_size[0]), int(len(X)*valid_size[1]))]

train_data_gen = GraphSAGE_DataGenerator(
    x_set=X,
    y_set=Y,
    adj_matrix = A,
    batch_size=batch_size,
    sample_nums=sample_nums,
    indexes=train_index
    )
valid_data_gen = GraphSAGE_DataGenerator(
    x_set=X,
    y_set=Y,
    adj_matrix = A,
    batch_size=batch_size,
    sample_nums=sample_nums,
    indexes=valid_index,
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


#--------------------------------训练和保存-------------------------------------
sage.fit(
    x = train_data_gen,
    epochs = epochs,
    steps_per_epoch = steps_per_epoch,
    validation_data = valid_data_gen
    )
sage.save_weights(save_path)
#-----------------------------------------------------------------------------