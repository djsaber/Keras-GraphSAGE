# coding=gbk

from model import GraphSAGE
from utils import *


#---------------------------------���ò���-------------------------------------
dataset='cora'                              # ���ݼ���'cora'/ 'citeseer'/ 'pubmed'

if dataset == 'cora':
    total_sample_num = 2703                 # ��������
    feature_dim = 1433                      # ����ά��
    label_dim = 7                           # ��ǩά��
if dataset == 'citeseer':
    total_sample_num = 3312 
    feature_dim = 3703 
    label_dim = 6
if dataset == 'pubmed':
    ...

dim_rounds = [128, label_dim]               # GraphSAGE���־ۺ��������ά��
aggr_method_rounds=['gcn', 'mean']          # GraphSAGE���־ۺ����ľۺϷ���
activation_rounds=['relu', 'softmax']       # GraphSAGE���־ۺ����ļ����
use_bias_rounds=[True, True]                # GraphSAGE���־ۺ����Ƿ���ƫ��
sample_nums = [4, 3]                        # GraphSAGE�����ھӲ�����
optimizer='adam'                            # GraphSAGE�Ż�����
loss='categorical_crossentropy'             # GraphSAGE��ʧ����
metrics=['acc']                             # GraphSAGE����ָ��

test_size = [0.8, 1]                        # �������������������ݼ����20%����
batch_size = 64                             # �������δ�С               
#-----------------------------------------------------------------------------


#---------------------------------����·��-------------------------------------
if dataset == 'cora':
    data_path = "D:/����/python����/�����ֲ�/GraphSAGE/datasets/cora"
    load_path = "D:/����/python����/�����ֲ�/GraphSAGE/save_models/gsage_cora.h5"
if dataset == 'citeseer':
    data_path = "D:/����/python����/�����ֲ�/GraphSAGE/datasets/citeseer"
    load_path = "D:/����/python����/�����ֲ�/GraphSAGE/save_models/gsage_citeseer.h5"
if dataset == 'pubmed':
    data_path = "D:/����/python����/�����ֲ�/GraphSAGE/datasets/pubmed"
    load_path = "D:/����/python����/�����ֲ�/GraphSAGE/save_models/gsage_pubmed.h5"
#-----------------------------------------------------------------------------


#--------------------------------�������ݼ�-------------------------------------
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


#--------------------------------��ȡ�Ͳ���-------------------------------------
sage.load_weights(load_path)
sage.evaluate(
    x = test_data_gen
    )
#-----------------------------------------------------------------------------