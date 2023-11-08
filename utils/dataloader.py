from functools import partial
from itertools import chain

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix


def generate_data(datas):
    x = datas[0]
    y = datas[1]
    for sample, label in zip(x, y):
        yield sample, label  # 生成几个，流式处理接口就放几个，这里并没有按照 输出定义 输出tuple，而是单个的


def process(x, y):  # 这里仅是数据集中的一个元素 (x, y) 流式处理时并不带有batch的维度
    # features = row[0]
    features = x
    labels = y
    items, alias_inputs = tf.unique(features)
    # 注意 alias_inputs 并不一致
    vector_length = tf.shape(features)[0]
    n_nodes = tf.shape(items)[0]
    adj = tf.zeros([n_nodes, n_nodes], dtype=tf.int32)  # TODO: 待会看看需不需要+1 注意shape 留意后续处理
    # A.先算出 value 和 index 然后 创建 稀疏矩阵 转化成 密集矩阵
    # B.如何优化循环，不用python代码
    for i in range(vector_length - 1):
        u = tf.where(condition=items == features[i])[0][0]
        # adj[u][u] = 1
        adj = tf.tensor_scatter_nd_update(tensor=adj, indices=[[u, u]], updates=[1])  # depth = 2
        v = tf.where(condition=items == features[i + 1])[0][0]
        if u == v or adj[u][v] == 4:
            continue
        # adj[v][v] = 1
        adj = tf.tensor_scatter_nd_update(tensor=adj, indices=[[v, v]], updates=[1])
        if adj[v][u] == 2:
            # adj[u][v] = 4
            # adj[v][u] = 4
            adj = tf.tensor_scatter_nd_update(tensor=adj, indices=[[u, v],
                                                                   [v, u]], updates=[4, 4])
        else:
            # adj[u][v] = 2
            # adj[v][u] = 3
            adj = tf.tensor_scatter_nd_update(tensor=adj, indices=[[u, v],
                                                                   [v, u]], updates=[2, 3])
    mask = tf.fill(tf.shape(features), 1.0)
    adj = tf.cast(adj, tf.float32)
    x = (alias_inputs, adj, items, mask, features)
    label = labels - 1
    return x, label


def process_data(x, y):
    pass


def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i] - 1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    return matrix


# def data_masks_new(all_sessions, n_node):
#     indices = []
#     values = []
#     dense_shape = (len(all_sessions), n_node)
#     for session in all_sessions:
#         for item in session:



def compute_max_len(raw_data):
    x = raw_data[0]
    # 找出最长的序列长度
    len_list = [len(d) for d in x]
    max_len = np.max(len_list)
    return max_len


def compute_item_num(sequence):
    seq_in_1D = list(chain.from_iterable(sequence))
    items_num = len(np.unique(seq_in_1D))
    return items_num


def compute_max_node(sequence):
    seq_in_1D = list(chain.from_iterable(sequence))
    max_node = np.max(seq_in_1D)
    return max_node


class DataLoader:
    def __init__(self, raw_data, train_mode=True):
        self.max_len = compute_max_len(raw_data)  # 最长序列
        self.data = raw_data
        self.data = data_masks(self.data[0], 40727)
        # self.data = self.reverse_data()  # 反转输入序列
        self.train_mode = train_mode
        # self.max_n_node =

    def dataloader(self):
        dataset = tf.data.Dataset.from_generator(generator=partial(generate_data, self.data),
                                                 output_signature=(tf.TensorSpec(shape=None,
                                                                                 dtype=tf.int32),
                                                                   tf.TensorSpec(shape=(),
                                                                                 dtype=tf.int32)))  # (x, label)
        # for data in dataset.batch(1):
        #     print(data)
        #     break
        # dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)  # 见鬼了，什么奇葩问题
        if self.train_mode:
            pass
            # TODO： 训练时打开shuffle，调试时避免减损性能
            dataset = dataset.shuffle(buffer_size=len(self.data[0]) - (len(self.data[0]) % 100))
        dataset = dataset.padded_batch(batch_size=100,
                                       padded_shapes=(
                                           ([self.max_len],
                                            [self.max_len, self.max_len],
                                            [self.max_len],
                                            [self.max_len],
                                            [self.max_len]
                                            ),
                                           []),
                                       drop_remainder=True
                                       )
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def reverse_data(self):
        x = self.data[0]
        x = [list(reversed(seq)) for seq in x]
        y = self.data[1]
        new_data = (x, y)
        return new_data


if __name__ == '__main__':
    # path_dataset = '../dataset/tmall'
    # train_data = pickle.load(open(f'{path_dataset}/train.txt', 'rb'))
    # a = [903, 907, 906, 905, 904, 903, 902]
    # b = 903
    # a = tf.constant(a)
    # b = tf.constant(b)
    seq = [[1], [3], [5], [7], [9], [11]]
    n_node = len(seq)
    data_masks(seq, n_node)
    # [903, 907, 906, 905, 904, 903, 902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
