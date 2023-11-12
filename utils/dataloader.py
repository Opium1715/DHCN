from functools import partial
from itertools import chain

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops as csr
from tqdm import tqdm
from tensorflow import raw_ops


def generate_data(datas):
    x = datas[0]
    y = datas[1]
    for sample, label in zip(x, y):
        yield sample, label  # 生成几个，流式处理接口就放几个，这里并没有按照 输出定义 输出tuple，而是单个的


def process_data(x, y):
    session_len = tf.cast(tf.shape(x)[0], tf.float32)
    items = x
    reversed_items = tf.reverse(tensor=x, axis=[0])
    mask = tf.ones(shape=(session_len,))

    sample = (session_len, items, reversed_items, mask)
    labels = y - 1

    return sample, labels


def data_masks_old(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in tqdm(range(len(all_sessions)), desc="处理稀疏矩阵坐标", total=len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i] - 1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    H_T = matrix
    BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
    BH_T = BH_T.T
    H = H_T.T
    DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    DH = DH.T
    DHBH_T = np.dot(DH, BH_T)
    adj = DHBH_T.tocoo()
    values = adj.data
    indices = tf.cast(tf.stack([adj.row, adj.col], axis=1), tf.int64)
    dense_shape = adj.shape
    adj = tf.SparseTensor(indices, values, dense_shape)

    return adj


def data_masks_new(all_sessions, n_node):
    session_length = len(all_sessions)
    indices = []
    dense_shape = (session_length, n_node)  # M x N
    for session_id, session in tqdm(zip(range(session_length), all_sessions), desc='处理稀疏矩阵坐标：',
                                    total=session_length):
        unique_item = np.unique(session)
        length = len(unique_item)
        for uid in range(length):
            indices.append([session_id, unique_item[uid]])  # [HyperEdge_id, node_id]
    values = tf.ones(shape=(len(indices),))
    H_T = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)  # 注意这里是HyperGraph矩阵的转置
    H_T_Matrix = csr.SparseTensorToCSRSparseMatrix(indices=H_T.indices, values=H_T.values, dense_shape=H_T.dense_shape)
    BH_T_b = tf.sparse.reshape(tf.sparse.reduce_sum(H_T, axis=1, output_is_sparse=True),
                               shape=(1, -1))
    b_values = tf.divide(1.0, BH_T_b.values)
    b_dim = tf.shape(b_values)[0]
    b_diag_indices = tf.stack([tf.range(b_dim, dtype=tf.int64), tf.range(b_dim, dtype=tf.int64)], axis=1)  # 对角矩阵坐标
    BH_T_b = tf.SparseTensor(indices=b_diag_indices, values=b_values, dense_shape=(b_dim, b_dim))
    # BH_T
    # # Row vector times matrix. other is a row.
    #             elif other.shape[0] == 1 and self.shape[1] == other.shape[1]:
    #                 other = self._dia_container(
    #                     (other.toarray().ravel(), [0]),
    #                     shape=(other.shape[1], other.shape[1])
    #                 )
    #                 return self._mul_sparse_matrix(other)
    BH_Matrix = csr.SparseMatrixSparseMatMul(a=H_T_Matrix,
                                             b=csr.SparseTensorToCSRSparseMatrix(indices=BH_T_b.indices,
                                                                                 values=BH_T_b.values,
                                                                                 dense_shape=BH_T_b.dense_shape),
                                             transpose_a=True, type=tf.float32)  # 如果转置失败，直接去乘法转置
    # H
    H = tf.sparse.transpose(H_T)
    # 注意此处在维度加处理之后，由于求和结果中含有0，导致reshape时向量形状改变
    DH_T_b = tf.sparse.reduce_sum(H, axis=1, output_is_sparse=True)
    b_values = tf.divide(1.0, DH_T_b.values)
    b_dim = tf.shape(H)[0]
    # b_diag_indices = tf.stack([tf.range(b_dim, dtype=tf.int64), tf.range(b_dim, dtype=tf.int64)], axis=1)  # 对角矩阵坐标
    b_diag_indices = tf.stack([tf.squeeze(DH_T_b.indices, -1), tf.squeeze(DH_T_b.indices, -1)], axis=1)
    DH_T_b = tf.SparseTensor(indices=b_diag_indices, values=b_values, dense_shape=(b_dim, b_dim))
    # DH
    DH_T_Matrix = csr.SparseMatrixSparseMatMul(a=H_T_Matrix,
                                               b=csr.SparseTensorToCSRSparseMatrix(indices=DH_T_b.indices,
                                                                                   values=DH_T_b.values,
                                                                                   dense_shape=DH_T_b.dense_shape),
                                               type=tf.float32)
    DHBH_T_Matrix = csr.SparseMatrixSparseMatMul(a=DH_T_Matrix, b=BH_Matrix, transpose_a=True, transpose_b=True,
                                                 # 挂在了这一步，不知道什么原因
                                                 type=tf.float32)
    DHBH_T = csr.CSRSparseMatrixToSparseTensor(sparse_matrix=DHBH_T_Matrix, type=tf.float32)
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


def process_batch(x, y):
    sessions = x[1]
    batch_size = tf.shape(sessions)[0]
    weight = tf.zeros(shape=(batch_size, batch_size))
    for i in range(batch_size):
        seq_a = tf.gather(sessions, i)
        seq_a, _ = tf.unique(seq_a)
        for j in range(i + 1, batch_size):
            seq_b, _ = tf.unique(tf.gather(sessions, j))
            intersection = tf.sets.intersection(tf.expand_dims(seq_a, 0), tf.expand_dims(seq_b, 0))
            union = tf.sets.union(tf.expand_dims(seq_a, 0), tf.expand_dims(seq_b, 0))
            updates = tf.tile(
                tf.expand_dims(tf.cast(x=(tf.shape(intersection.values)[0] - 1) / (tf.shape(union.values)[0] - 1)
                                       , dtype=tf.float32), axis=0), multiples=[2])
            weight = tf.tensor_scatter_nd_update(tensor=weight, indices=[[0, 1],
                                                                         [1, 0]], updates=updates)
    weight = weight + tf.eye(batch_size)
    degree = tf.reduce_sum(weight, axis=1)
    degree = tf.linalg.diag(diagonal=1.0 / degree)

    sample = x + (weight, degree, )
    label = y

    return sample, label


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
        # self.data = self.reverse_data()  # 反转输入序列
        self.train_mode = train_mode

    def get_adj(self):
        # return data_masks_new(self.data[0], n_node=self.n_node)
        return data_masks_old(self.data[0], n_node=self.n_node)

    def dataloader(self):
        dataset = tf.data.Dataset.from_generator(generator=partial(generate_data, self.data),
                                                 output_signature=(tf.TensorSpec(shape=None,
                                                                                 dtype=tf.int32),
                                                                   tf.TensorSpec(shape=(),
                                                                                 dtype=tf.int32)))  # (x, label)
        dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        if self.train_mode:
            pass
            # TODO： 训练时打开shuffle，调试时避免减损性能
            # dataset = dataset.shuffle(buffer_size=len(self.data[0]) - (len(self.data[0]) % 100))
        dataset = dataset.padded_batch(batch_size=100,  # (session_len, items, reversed_items, mask)
                                       padded_shapes=(
                                           ([],
                                            [self.max_len],
                                            [self.max_len],
                                            [self.max_len],
                                            ),
                                           []),
                                       drop_remainder=True
                                       )
        dataset = dataset.map(process_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


if __name__ == '__main__':
    process_data([1, 2, 3, 4], 5)
