# coding=gbk

from keras.layers import Layer, Input
from keras.models import Model
import keras.activations as activations
import keras.backend as K


class Aggregator(Layer):
    '''聚合函数
    参数：
        - outout_dim：输出维度
        - aggr_method：聚合方法，'mean'：平均聚合，'gcn'：类GCN聚合，'pooling'：max pooling聚合
        - activation：激活函数
        - use_bias：是否使用偏置
    输入：
        - [
        目标节点特征, (n, dim)，目标节点数为n
        邻居节点特征, (n*k, dim)，每个目标节点的邻节点数为k
        ]
    输出：
        - 更新后的目标节点特征， (n, outout_dim)
    '''
    def __init__(
        self, 
        output_dim,
        aggr_method,
        activation,
        use_bias=True,
        **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.aggr_method = aggr_method
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        if self.aggr_method in ['mean', 'pooling']:
            self.w = self.add_weight(
                name = 'w',
                shape=(2*input_dim, self.output_dim),
                initializer = 'glorot_uniform',
                )
            if self.aggr_method == 'pooling':
                self.w_pool = self.add_weight(
                    name = 'w_pool',
                    shape=(input_dim, input_dim),
                    initializer = 'glorot_uniform',
                    )
                self.bias_pool = self.add_weight(
                    name = 'bias_pool',
                    shape=(input_dim, ),
                    initializer = 'zero',
                    )
        if self.aggr_method in ['gcn']:
            self.w = self.add_weight(
                name = 'w',
                shape=(input_dim, self.output_dim),
                initializer = 'glorot_uniform',
                )
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape=(self.output_dim,),
                initializer = 'zero',
                )   

    def call(self, inputs):
        src_features = inputs[0]
        neighbor_features = inputs[1]

        batch = K.shape(neighbor_features)[0]
        num = K.shape(src_features)[1]
        k = K.shape(neighbor_features)[1] // num
        input_dim = K.shape(src_features)[-1]
        neighbor_target_shape = (batch, num, k, input_dim)

        if self.aggr_method == 'mean':
            # (batch, num, k, input_dim)
            neighbor_features = K.reshape(neighbor_features, neighbor_target_shape)
            # (batch, num, input_dim)
            neighbor_features = K.mean(neighbor_features, axis = 2)
            # (batch, num, 2*input_dim)
            src_features = K.concatenate([src_features, neighbor_features])
            # (batch, num, output_dim)
            src_features = K.dot(src_features, self.w)

        elif self.aggr_method == 'gcn':
            # (batch, num, 1, input_dim)
            src_features = K.expand_dims(src_features, axis=2)
            # (batch, num, k, input_dim)
            neighbor_features = K.reshape(neighbor_features, neighbor_target_shape)
            #(batch, num, k+1, input_dim)
            src_features = K.concatenate([src_features, neighbor_features], axis = 2)
            #(batch, num, input_dim)
            src_features = K.mean(src_features, axis = 2)
            # (batch, num, output_dim)
            src_features = K.dot(src_features, self.w)

        elif self.aggr_method == 'pooling':
            # (batch, num*k, input_dim)
            neighbor_features = K.dot(neighbor_features, self.w_pool)
            # (batch, num*k, input_dim)
            neighbor_features = K.bias_add(neighbor_features, self.bias_pool)
            # (batch, num*k, input_dim)
            neighbor_features = self.activation(neighbor_features)
            # (batch, num, k, input_dim)
            neighbor_features = K.reshape(neighbor_features, neighbor_target_shape)
            # (batch, num, input_dim)
            neighbor_features = K.max(neighbor_features, axis=2)
            # (batch, num, 2*input_dim)
            src_features = K.concatenate([src_features, neighbor_features])
            # (batch, num, output_dim)
            src_features = K.dot(src_features, self.w)

        if self.use_bias:
            src_features = K.bias_add(src_features, self.bias)

        return self.activation(src_features)


class GraphSAGE(Model):
    '''GraphSAGE模型
    通过多轮聚合，更新目标节点的embedding
    例如输入[目标节点, 1阶邻居, 2阶邻居, 3阶邻居]，聚合3轮时：
        第1轮聚合：聚合1阶邻居更新目标节点，聚合2阶邻居更新1阶邻居，聚合3阶邻居更新2阶邻居
        第2轮聚合：聚合1阶邻居更新目标节点，聚合2阶邻居更新1阶邻居
        第3轮聚合：聚合1阶邻居更新目标节点
    参数：
        - dim_rounds：各轮聚合的维度，[dim_1, dim_2, ... dim_out]
        - aggr_method_rounds：各轮聚合的聚合方法，['pooling', 'pooling', ... 'mean']
        - activation_rounds：各轮聚合的激活函数，['relu', 'relu', ... 'softmax']
        - use_bias_rounds：各轮聚合是否使用偏置，[True, True, ..., True]
    输入：
        - [
        目标节点特征      (batch, 1, dim)，
        1阶邻居特征       (batch, n1, dim), 
        2阶邻居特征       (batch, n1*n2, dim), 
        ..., 
        K阶邻居节点特征    (batch, n1*...*nk-1*nk, dim)
        ]
    输出：
        - 目标节点特征     (batch, 1, hidden_dim)
    '''
    def __init__(
        self,
        dim_rounds, 
        aggr_method_rounds, 
        activation_rounds, 
        use_bias_rounds, 
        **kwargs):
        super().__init__(**kwargs)
        self.dim_rounds=dim_rounds
        self.aggr_method_rounds=aggr_method_rounds
        self.activation_rounds=activation_rounds
        self.use_bias_rounds=use_bias_rounds
        self.rounds = len(dim_rounds)

    def build(self, input_shape):
        self.aggregators = [
            Aggregator(
                output_dim=self.dim_rounds[i],
                aggr_method=self.aggr_method_rounds[i],
                activation=self.activation_rounds[i],
                use_bias=self.use_bias_rounds[i]
                ) for i in range(self.rounds)
            ]
        self.call([Input(shape[1:]) for shape in input_shape])
        super().build(input_shape)

    def call(self, inputs):
        hidden = inputs
        for r in range(self.rounds):
            next_hidden = []
            aggregator = self.aggregators[r]
            for hop in range(self.rounds-r):
                src_nodes = hidden[hop]
                neighbor_nodes = hidden[hop+1]
                aggr_nodes = aggregator([src_nodes, neighbor_nodes])
                next_hidden.append(aggr_nodes)
            hidden = next_hidden
        return hidden[0]