import math

import tensorflow as tf
from tensorflow import keras


class HyperConv(keras.layers.Layer):
    def __init__(self, layers, emb_size=100):
        super().__init__()
        self.emb_size = emb_size
        self.layers = layers

    def call(self, inputs, *args, **kwargs):  # inputs = (adj, item_emb)
        adj = inputs[0]
        item_embeddings = inputs[1]
        whole = tf.range(40727)
        # try:
        #     tf.debugging.check_numerics(item_embeddings, f'error value Nan{item_embeddings}')
        # except tf.errors.InvalidArgumentError as e:
        #     print(f"Invalid values in tensor: {e}")
        item_embeddings_0 = item_embeddings(whole)
        final = [item_embeddings_0]
        # conv
        for i in range(self.layers):
            item_embeddings_layer = tf.sparse.sparse_dense_matmul(adj, final[-1])  # 大力出奇迹
            final.append(item_embeddings_layer)
        item_embeddings_Conv = tf.add_n(final) / (self.layers + 1)  # 做法过于草率，后期试试注意力机制

        return item_embeddings_Conv


class LineConv(keras.layers.Layer):
    def __init__(self, layers, emb_size=100):
        super().__init__()
        self.layers = layers
        self.emb_size = emb_size

    def call(self, inputs, *args, **kwargs):
        item_embeddings = inputs[0]
        D = inputs[1]
        A = inputs[2]
        session_item = inputs[3]
        session_len = inputs[4]
        mask = inputs[5]
        mask = tf.cast(tf.squeeze(mask, -1), tf.int32)
        # zeros = tf.zeros(shape=(1, self.emb_size))
        # item_embeddings = item_embeddings(tf.range(40727))
        # item_embeddings = tf.concat([zeros, item_embeddings], axis=0)  # 顶下去了
        # seq_h = tf.gather(item_embeddings, session_item)
        seq_h = item_embeddings((session_item - 1) * mask)  # 为什么要下顶，这样也可以
        mask = tf.cast(tf.expand_dims(mask, -1), tf.float32)
        # for i in tf.range(tf.shape(session_item)[0]):
        #     seq_h.append(item_embeddings(tf.gather(session_item, indices=i)))
        session_emb_lgcn = tf.divide(tf.reduce_sum(seq_h * mask, axis=1), session_len)
        session_emb_lgcn_0 = session_emb_lgcn
        session = [session_emb_lgcn_0]
        DA = tf.matmul(D, A)
        # conv
        for i in range(self.layers):
            session_emb_lgcn_layer = tf.matmul(DA, session[-1])
            session.append(session_emb_lgcn_layer)
        session_emb_lgcn_Conv = tf.add_n(session) / (self.layers + 1)
        return session_emb_lgcn_Conv


class SelfSuperviseLayer(keras.layers.Layer):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def call(self, inputs, *args, **kwargs):  # (sess_emb_hgnn, sess_emb_lgcn)
        session_emb_hgcn = inputs[0]
        session_emb_lgcn = inputs[1]
        shuffle_row_emb_hgcn = tf.gather(params=session_emb_hgcn,
                                         indices=tf.random.shuffle(tf.range(tf.shape(session_emb_hgcn)[0]),
                                                                   seed=self.seed))
        shuffle_row_clo_emb_hgcn = tf.gather(params=shuffle_row_emb_hgcn,
                                             indices=tf.random.shuffle(tf.range(tf.shape(session_emb_hgcn)[1]),
                                                                       seed=self.seed),
                                             axis=1)

        pos = tf.reduce_sum(tf.matmul(session_emb_hgcn, session_emb_lgcn), 1)
        neg1 = tf.reduce_sum(tf.matmul(session_emb_lgcn, shuffle_row_clo_emb_hgcn), 1)
        ones = tf.ones(shape=(tf.shape(neg1)[0]))

        con_loss = tf.reduce_sum(-tf.math.log(1e-8 + tf.sigmoid(pos)) - tf.math.log(1e-8 + (ones - tf.sigmoid(neg1))))

        return con_loss


class DHCN(keras.models.Model):
    def __init__(self, adj, n_node, layers, beta, emb_size=100):
        super().__init__()
        self.emb_size = emb_size
        self.adj = adj
        self.n_node = n_node
        self.layer_num = layers
        self.beta = beta
        self.stdv = 1.0 / math.sqrt(self.emb_size)

        initializer = tf.random_uniform_initializer(minval=-self.stdv, maxval=self.stdv)
        self.embedding = keras.layers.Embedding(self.n_node, self.emb_size, embeddings_initializer=initializer)
        self.pos_embedding = keras.layers.Embedding(200, self.emb_size, embeddings_initializer=initializer)
        self.HyperGraph = HyperConv(self.layer_num, emb_size=self.emb_size)
        self.LineGraph = LineConv(self.layer_num, emb_size=self.emb_size)
        self.SSL = SelfSuperviseLayer()
        self.w_1 = keras.layers.Dense(units=self.emb_size,  # tanh(dot(w1*input)+bias) (3)
                                      activation='tanh',
                                      use_bias=True,
                                      kernel_initializer=initializer,
                                      bias_initializer=initializer)
        self.f = self.add_weight(shape=(self.emb_size, 1),
                                 dtype=tf.float32,
                                 initializer=initializer,
                                 trainable=True,
                                 name='f')
        self.glu1 = keras.layers.Dense(units=self.emb_size,
                                       use_bias=True,
                                       kernel_initializer=initializer,
                                       bias_initializer=initializer)
        self.glu2 = keras.layers.Dense(units=self.emb_size,
                                       use_bias=False,
                                       kernel_initializer=initializer)

    def call(self, inputs, training=None, mask=None):  # (session_item, session_len, reversed_sess_item, mask)
        session_item = inputs[1]
        session_len = inputs[0]
        D = inputs[5]
        A = inputs[4]
        reversed_sess_item = inputs[2]
        mask = inputs[3]

        batch_size = tf.shape(mask)[0]
        # HyperGraph
        item_embedding_hg = self.HyperGraph((self.adj, self.embedding))
        # Tmall 作者认为不使用反向位置编码对于该数据集更好
        zeros = tf.zeros(shape=(1, self.emb_size))
        item_embedding = tf.concat([zeros, item_embedding_hg], axis=0)  # 下顶
        # seq_h = tf.zeros(shape=(batch_size, tf.shape(reversed_sess_item)[1], self.emb_size))
        # 获取反向item的向量
        # seq_h = [self.embedding(tf.gather(reversed_sess_item, index)) for index in range(batch_size)]
        # seq_h = [tf.gather(params=item_embedding,
        #                    indices=tf.gather(reversed_sess_item, index)) for index in range(batch_size)]
        seq_h = tf.gather(item_embedding, indices=reversed_sess_item)
        seq_h = tf.reshape(tf.concat(seq_h, axis=0), shape=(batch_size, tf.shape(reversed_sess_item)[1], self.emb_size))
        Xs = tf.divide(tf.reduce_sum(seq_h, axis=1), session_len)  # 这里不需要mask吗？ 前面的下顶使得第0行是0, 那么每次取0都是0，聪明
        mask = tf.expand_dims(mask, -1)
        len_seq = tf.shape(seq_h)[1]

        Xs = tf.tile(tf.expand_dims(Xs, -2), multiples=(1, len_seq, 1))
        # nh = tf.nn.tanh(seq_h) 为什么直接连偏置都没有了，权重都没有了？
        # Xt = self.w_1(seq_h)
        Xt = tf.tanh(seq_h)
        Xt = tf.nn.sigmoid(self.glu1(Xt) + self.glu2(Xs))

        beta = tf.matmul(Xt, self.f)
        beta = beta * mask

        theta_h = tf.reduce_sum(beta * seq_h, axis=1)

        # LineGraph
        session_emb_lg = self.LineGraph((self.embedding, D, A, session_item, session_len, mask))
        # self-supervise
        con_loss = self.SSL((theta_h, session_emb_lg))
        beta_con_loss = self.beta * con_loss
        # predict # 这里和之前的不一样，用了新提取出来的item_emb, 之前都是用本轮原始的item_emb
        # scores = tf.matmul(theta_h, item_embedding_hg, transpose_b=True)  # 原本的
        scores = tf.matmul(theta_h, self.embedding(tf.range(40727)), transpose_b=True)
        # output = [scores, beta_con_loss]
        # self.add_loss(beta_con_loss)

        return scores
