# coding=gbk

import numpy as np
import pandas as pd
import networkx as nx
from keras.utils import Sequence
from tqdm import tqdm


def sampling(src_nodes, sample_num, neighbor_table, seed=1):
    '''����Դ�ڵ����ָ���������ھӽڵ㣬ע��ʹ�õ����зŻصĲ�����
    ĳ���ڵ���ھӽڵ��������ڲ�������ʱ��������������ظ��Ľڵ�
    
    ����:
        - src_nodes {list, ndarray} -- Դ�ڵ��б�
        - sample_num {int} -- ��Ҫ�����Ľڵ���
        - neighbor_table {dict} -- �ڵ㵽���ھӽڵ��ӳ���
        - seed���������
    ���:
        np.ndarray -- ����������ɵ��б�
    '''
    results = []
    np.random.seed(seed)
    for sid in src_nodes:
        # �ӽڵ���ھ��н����зŻصؽ��в���
        res = np.random.choice(neighbor_table[sid], size=(sample_num, ))
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table, seed):
    '''����Դ�ڵ���ж�ײ���
    
    ����:
        - src_nodes {list, np.ndarray} -- Դ�ڵ�id
        - sample_nums {list of int} -- ÿһ����Ҫ�����ĸ���
        - neighbor_table {dict} -- �ڵ㵽���ھӽڵ��ӳ��
        - seed���������
    ���:
        [list of ndarray] -- ÿһ�ײ����Ľ��
    '''
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table, seed)
        sampling_result.append(hopk_result)
    return sampling_result


def load_cora(path):
    '''��ȡcora���ݼ�
    ������
        - path�����ݼ�·��
    ���أ�
        - A���ڽӾ���
        - X����������
        - Y��ѵ��ʱ�ı�ǩ
    '''
    raw_data = pd.read_csv(path+'/cora.content', sep='\t', header=None)
    raw_data_cites = pd.read_csv(path+'/cora.cites', sep='\t', header=None)
    node_num = raw_data.shape[0]
    node_id = list(raw_data.index)
    paper_id = list(raw_data[0])
    c = zip(paper_id, node_id)
    map_ = dict(c)
    A = np.zeros((node_num,node_num), dtype='float32')
    for paper_id_i, paper_id_j in zip(raw_data_cites[0], raw_data_cites[1]):
        x = map_[paper_id_i]
        y = map_[paper_id_j]
        A[x][y] = A[y][x] = 1
    X = raw_data.iloc[:,1:-1].to_numpy(dtype='float32')
    Y = pd.get_dummies(raw_data[1434]).to_numpy(dtype='float32')

    return A, X, Y


def load_citeseer(path):
    '''��ȡciteseer���ݼ�
    ������
        - path�����ݼ�·��
    ���أ�
        - A���ڽӾ���
        - X����������
        - Y��ѵ��ʱ�ı�ǩ
    '''
    raw_data = pd.read_csv(path+'/citeseer.content', sep='\t', header=None)
    raw_data_cites = pd.read_csv(path+'/citeseer.cites', sep='\t', header=None)
    node_num = raw_data.shape[0]
    node_id = list(raw_data.index)
    paper_id = list(raw_data[0])
    paper_id = [str(a_paper_id) for a_paper_id in paper_id]
    map_dict = dict(zip(paper_id, node_id))
    A = np.eye(node_num, dtype='float32')
    for paper_id_i, paper_id_j in zip(raw_data_cites[0], raw_data_cites[1]):
        try:
            x = map_dict[paper_id_i]
            y = map_dict[paper_id_j]
            if x != y:
                A[x][y] = A[y][x] = 1
        except:
            print(f'{paper_id_i} or {paper_id_j} is not in map_dict.keys()!')
    X = raw_data.iloc[:,1:-1].to_numpy(dtype='float32')
    Y = pd.get_dummies(raw_data[3704]).to_numpy(dtype='float32')

    return A, X, Y


def load_pubmed(path):
    '''��ȡciteseer���ݼ�
    ������
        - path�����ݼ�·��
    ���أ�
        - A���ڽӾ���
        - X����������
        - Y��ѵ��ʱ�ı�ǩ
    '''
    paper2idx = {}
    word2idx = {}
    with open(path+'/Pubmed-Diabetes.DIRECTED.cites.tab', "r") as f:
        lines = f.readlines()[2:]
        edges = set()
        for line in tqdm(lines, desc='load nodes and edges'):
            paper1 = line[line.index(':')+1:line.rindex(':')-8]
            paper2 = line[line.rindex(':')+1:-1]
            paper2idx.setdefault(paper1, len(paper2idx))
            paper2idx.setdefault(paper2, len(paper2idx))
            edges.add((paper2idx[paper1], paper2idx[paper2]))
    Graph = nx.Graph()
    Graph.add_edges_from(edges)
    A = nx.adjacency_matrix(Graph, nodelist=list(paper2idx.values())).todense()
    A = A.astype('float32')
    A += np.eye(A.shape[0])
    with open(path+'/Pubmed-Diabetes.NODE.paper.tab', "r") as f:
        lines = f.readlines()[2:]
        X = np.zeros((A.shape[0], 500), np.float32)
        Y = np.zeros((A.shape[0], 3), np.float32)
        for line in tqdm(lines, desc='load features and labels'):
            inf = line.split('\t')[:-1]
            node = paper2idx[inf[0]]
            label = int(inf[1][-1])-1
            feature = {w.split('=')[0]:float(w.split('=')[1]) for w in inf[2:]}
            Y[node][label] = 1
            for w,v in feature.items():
                word2idx.setdefault(w, len(word2idx))
                idx = word2idx[w]
                X[node][idx] = v

    return A, X, Y


class GraphSAGE_DataGenerator(Sequence):
    '''���GraphSAGEģ�͵���������������
    ������
        - x_set����������
        - y_set����ǩ����
        - adj_matrix���ڽӾ���
        - batch_size�����ߴ�
        - sample_nums�������ھӵĲ�����
        - indexes=None����������������
        - shuffle=True���Ƿ����
        - seed=1��shuffle���������
    '''
    def __init__(
        self, 
        x_set, 
        y_set, 
        adj_matrix, 
        batch_size, 
        sample_nums, 
        indexes=None,
        shuffle=True, 
        seed=1
        ):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.sample_nums = sample_nums
        self.indexes = np.array(indexes) if indexes else np.arange(len(self.x))
        assert len(self.indexes) >= self.batch_size
        self.shuffle = shuffle
        self.seed = seed
    
        g = nx.from_numpy_array(adj_matrix)
        self.neighbor_table = {n:list(g.neighbors(n)) for n in g.nodes}
              
    def __len__(self):
        return int((len(self.indexes) / self.batch_size))
    
    def __getitem__(self, idx):
        sample_idx_list = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]
        sample_result = multihop_sampling(
            sample_idx_list, 
            self.sample_nums, 
            self.neighbor_table, 
            self.seed
            )
        batch_x = []
        batch_y = np.expand_dims(self.y[sample_idx_list], axis=1)
        for arr in sample_result:
            node_features = self.x[arr]
            node_features = np.array(np.split(node_features, self.batch_size))
            batch_x.append(node_features)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.seed(self.seed)
            np.random.shuffle(self.indexes)