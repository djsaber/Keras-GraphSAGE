# coding=gbk

from model import GraphSAGE
from utils import *


#---------------------------------���ò���-------------------------------------
dataset='pubmed'                            # ���ݼ���'cora'/ 'citeseer'/ 'pubmed'

if dataset == 'cora':
    total_sample_num = 2703                 # ��������
    feature_dim = 1433                      # ����ά��
    label_dim = 7                           # ��ǩά��
if dataset == 'citeseer':
    total_sample_num = 3312 
    feature_dim = 3703 
    label_dim = 6
if dataset == 'pubmed':
    total_sample_num = 19717 
    feature_dim = 500 
    label_dim = 3

dim_rounds = [128, label_dim]               # GraphSAGE���־ۺ��������ά��
aggr_method_rounds=['gcn', 'mean']          # GraphSAGE���־ۺ����ľۺϷ���
activation_rounds=['relu', 'softmax']       # GraphSAGE���־ۺ����ļ����
use_bias_rounds=[True, True]                # GraphSAGE���־ۺ����Ƿ���ƫ��
sample_nums = [6, 3]                        # GraphSAGE�����ھӲ�����
optimizer='adam'                            # GraphSAGE�Ż�����
loss='categorical_crossentropy'             # GraphSAGE��ʧ����
metrics=['acc']                             # GraphSAGE����ָ��

train_size = [0, 0.5]                       # ѵ�����������������ݼ���ǰ50%����
valid_size = [0.5, 0.8]                     # ��֤���������������ݼ���50%-80%����
batch_size = 64                             # ѵ�����δ�С
epochs = 5                                  # ѵ������                                                              
steps_per_epoch = total_sample_num*(        # ÿ��ѵ��������������ѵ��������//batch_size
    train_size[1]-train_size[0])//batch_size                      
#-----------------------------------------------------------------------------


#---------------------------------����·��-------------------------------------
if dataset == 'cora':
    data_path = "D:/����/python����/�����ֲ�/GraphSAGE/datasets/cora"
    save_path = "D:/����/python����/�����ֲ�/GraphSAGE/save_models/gsage_cora.h5"
if dataset == 'citeseer':
    data_path = "D:/����/python����/�����ֲ�/GraphSAGE/datasets/citeseer"
    save_path = "D:/����/python����/�����ֲ�/GraphSAGE/save_models/gsage_citeseer.h5"
if dataset == 'pubmed':
    data_path = "D:/����/python����/�����ֲ�/GraphSAGE/datasets/pubmed"
    save_path = "D:/����/python����/�����ֲ�/GraphSAGE/save_models/gsage_pubmed.h5"
#-----------------------------------------------------------------------------


#--------------------------------�������ݼ�-------------------------------------
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


#---------------------------------�ģ��-------------------------------------
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


#--------------------------------ѵ���ͱ���-------------------------------------
sage.fit(
    x = train_data_gen,
    epochs = epochs,
    steps_per_epoch = steps_per_epoch,
    validation_data = valid_data_gen
    )
sage.save_weights(save_path)
#-----------------------------------------------------------------------------