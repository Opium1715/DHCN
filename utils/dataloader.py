from functools import partial
from itertools import chain

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops as csr
from tensorflow import raw_ops


def generate_data(datas):
    x = datas[0]
    y = datas[1]
    for sample, label in zip(x, y):
        yield sample, label  # 生成几个，流式处理接口就放几个，这里并没有按照 输出定义 输出tuple，而是单个的


def process_data(x, y):
    session_len = tf.shape(x)[0]
    items = x
    reversed_items = tf.reverse(tensor=x, axis=0)
    mask = tf.ones(shape=(session_len,))

    sample = (session_len, items, reversed_items, mask)
    labels = y - 1

    return sample, labels


def data_masks_old(all_sessions, n_node):
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


def data_masks_new(all_sessions, n_node):
    session_length = len(all_sessions)
    indices = []
    dense_shape = (session_length, n_node)  # M x N
    for session_id, session in zip(range(session_length), all_sessions):
        unique_item, idx = tf.unique(session)
        length = tf.shape(unique_item)[0]
        for uid in range(length):
            indices.append([session_id, tf.gather(unique_item, indices=uid)])  # [HyperEdge_id, node_id]
    values = tf.ones(shape=(len(indices),), dtype=tf.int64)
    H_T = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)  # 注意这里是HyperGraph矩阵的转置
    H_T_Matrix = csr.sparse_tensor_to_csr_sparse_matrix(indices=indices, values=values,
                                                        dense_shape=dense_shape)
    # BH_T
    BH_T_Matrix = csr.sparse_matrix_transpose(csr.matmul(a=H_T_Matrix,
                                                         b=csr.CSRSparseMatrix(
                                                             1.0 / tf.sparse.reshape(tf.sparse.reduce_sum(H_T, axis=1),
                                                                                     shape=(1, -1))),
                                                         transpose_a=True
                                                         ), type=tf.float32)  # 如果转置失败，直接去乘法转置
    # H
    H = tf.sparse.transpose(H_T)
    # DH
    DH_T_Matrix = csr.matmul(a=H_T_Matrix,
                             b=csr.CSRSparseMatrix(1.0 / tf.sparse.reshape(tf.sparse.reduce_sum(H, axis=1),
                                                                           shape=(1, -1))))
    DHBH_T_Matrix = csr.matmul(a=DH_T_Matrix, b=BH_T_Matrix, transpose_a=True)
    DHBH_T = csr.csr_sparse_matrix_to_sparse_tensor(DHBH_T_Matrix, type=tf.float32)
    DHBH_T = tf.SparseTensor(DHBH_T)
    return DHBH_T


@tf.function
def get_overlap_weight(sessions):
    batch_size = tf.shape(sessions)[0]
    weight = tf.zeros(shape=(batch_size, batch_size))
    for i in range(batch_size):
        seq_a = tf.gather(sessions, i)
        seq_a, _ = tf.unique(seq_a)
        for j in range(i + 1, batch_size):
            seq_b, _ = tf.unique(tf.gather(sessions, j))
            intersection = tf.sets.intersection(seq_a, seq_b)
            union = tf.sets.union(seq_a, seq_b)
            weight = tf.tensor_scatter_nd_update(tensor=weight, indices=[[i, j],
                                                                         [j, i]],
                                                 updates=tf.shape(intersection)[0] / tf.shape(union)[0])
    weight = weight + tf.eye(batch_size)
    degree = tf.reduce_sum(weight, axis=1)
    degree = tf.linalg.diag(diagonal=1.0 / degree)

    return weight, degree


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
    def __init__(self, raw_data, n_node, train_mode=True):
        self.max_len = compute_max_len(raw_data)  # 最长序列
        self.data = raw_data
        self.n_node = n_node
        self.adj = data_masks_new(self.data[0], n_node=self.n_node)
        # self.data = self.reverse_data()  # 反转输入序列
        self.train_mode = train_mode

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
            # dataset = dataset.shuffle(buffer_size=len(self.data[0]) - (len(self.data[0]) % 100))
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
    get_slice()
