# coding=gbk

from keras.layers import Layer, Input
from keras.models import Model
import keras.activations as activations
import keras.backend as K


class Aggregator(Layer):
    '''�ۺϺ���
    ������
        - outout_dim�����ά��
        - aggr_method���ۺϷ�����'mean'��ƽ���ۺϣ�'gcn'����GCN�ۺϣ�'pooling'��max pooling�ۺ�
        - activation�������
        - use_bias���Ƿ�ʹ��ƫ��
    ���룺
        - [
        Ŀ��ڵ�����, (n, dim)��Ŀ��ڵ���Ϊn
        �ھӽڵ�����, (n*k, dim)��ÿ��Ŀ��ڵ���ڽڵ���Ϊk
        ]
    �����
        - ���º��Ŀ��ڵ������� (n, outout_dim)
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
    '''GraphSAGEģ��
    ͨ�����־ۺϣ�����Ŀ��ڵ��embedding
    ��������[Ŀ��ڵ�, 1���ھ�, 2���ھ�, 3���ھ�]���ۺ�3��ʱ��
        ��1�־ۺϣ��ۺ�1���ھӸ���Ŀ��ڵ㣬�ۺ�2���ھӸ���1���ھӣ��ۺ�3���ھӸ���2���ھ�
        ��2�־ۺϣ��ۺ�1���ھӸ���Ŀ��ڵ㣬�ۺ�2���ھӸ���1���ھ�
        ��3�־ۺϣ��ۺ�1���ھӸ���Ŀ��ڵ�
    ������
        - dim_rounds�����־ۺϵ�ά�ȣ�[dim_1, dim_2, ... dim_out]
        - aggr_method_rounds�����־ۺϵľۺϷ�����['pooling', 'pooling', ... 'mean']
        - activation_rounds�����־ۺϵļ������['relu', 'relu', ... 'softmax']
        - use_bias_rounds�����־ۺ��Ƿ�ʹ��ƫ�ã�[True, True, ..., True]
    ���룺
        - [
        Ŀ��ڵ�����      (batch, 1, dim)��
        1���ھ�����       (batch, n1, dim), 
        2���ھ�����       (batch, n1*n2, dim), 
        ..., 
        K���ھӽڵ�����    (batch, n1*...*nk-1*nk, dim)
        ]
    �����
        - Ŀ��ڵ�����     (batch, 1, hidden_dim)
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