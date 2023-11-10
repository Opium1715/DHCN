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
        final = [item_embeddings]
        # conv
        for i in range(self.layers):
            item_embeddings = tf.sparse.sparse_dense_matmul(adj, item_embeddings)  # 大力出奇迹
            final.append(item_embeddings)
        item_embeddings = tf.add(final) / (self.layers + 1)  # 做法过于草率，后期试试注意力机制
        return item_embeddings


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
        zeros = tf.zeros(shape=(1, self.emb_size))
        item_embeddings = tf.concat([zeros, item_embeddings], axis=0)  # 顶下去了
        seq_h = item_embeddings(session_item)
        # for i in tf.range(tf.shape(session_item)[0]):
        #     seq_h.append(item_embeddings(tf.gather(session_item, indices=i)))
        session_emb_lgcn = tf.divide(tf.reduce_sum(seq_h, axis=1), session_len)
        session = [session_emb_lgcn]
        DA = tf.matmul(D, A)
        # conv
        for i in range(self.layers):
            session_emb_lgcn = tf.matmul(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        session_emb_lgcn = tf.add_n(session) / (self.layers + 1)
        return session_emb_lgcn


class SelfSuperviseLayer(keras.layers.Layer):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
    def call(self, inputs, *args, **kwargs):  # (sess_emb_hgnn, sess_emb_lgcn)
        session_emb_hgcn = inputs[0]
        session_emb_lgcn = inputs[1]
        shuffle_row_emb_hgcn = tf.gather(params=session_emb_hgcn,
                                     indices=tf.random.shuffle(tf.range(tf.shape(session_emb_hgcn)[0]), seed=self.seed))
        shuffle_row_clo_emb_hgcn = tf.gather(params=shuffle_row_emb_hgcn,
                                             indices=tf.random.shuffle(tf.range(tf.shape(session_emb_hgcn)[1]), seed=self.seed))

        pos = tf.reduce_sum(tf.matmul(session_emb_hgcn, session_emb_lgcn), 1)
        neg1 = tf.reduce_sum(tf.matmul(session_emb_lgcn, shuffle_row_clo_emb_hgcn), 1)
        ones = tf.ones(shape=(tf.shape(neg1)[0]))
        con_loss = tf.reduce_sum(-tf.math.log(1e-8 + tf.sigmoid(pos))-tf.math.log(1e-8 + (ones - tf.sigmoid(neg1))))

        return con_loss



class DHCN(keras.models.Model):
    def __init__(self, adj, n_node, layers, beta, emb_size=100):
        super().__init__()
        self.emb_size = emb_size
        self.adj = adj
        self.n_node = n_node
        self.layers = layers
        self.beta = beta
        self.stdv = 1.0 / tf.math.sqrt(self.emb_size)

        initializer = tf.random_uniform_initializer(minval=-self.stdv, maxval=self.stdv)
        self.embedding = keras.layers.Embedding(self.n_node, self.emb_size, embeddings_initializer=initializer)
        self.pos_embedding = keras.layers.Embedding(200, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, emb_size=self.emb_size)
        self.LineGraph = LineConv(self.layers, emb_size=self.emb_size)
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

    def get_overlap_weight(self, sessions):
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

    def call(self, inputs, training=None, mask=None):  # (session_item, session_len, D, A, reversed_sess_item, mask)
        session_item = inputs[0]
        session_len = inputs[1]
        D = inputs[2]
        A = inputs[3]
        reversed_sess_item = inputs[4]
        mask = inputs[5]

        batch_size = tf.shape(mask)[0]
        # HyperGraph
        item_embedding_hg = self.HyperGraph((self.adj, self.embedding))
        # Tmall 作者认为不使用反向位置编码对于该数据集更好
        zeros = tf.zeros(shape=(1, self.emb_size))
        item_embedding = tf.concat([zeros, item_embedding_hg], axis=0)  # 下顶
        # seq_h = tf.zeros(shape=(batch_size, tf.shape(reversed_sess_item)[1], self.emb_size))
        # 获取反向item的向量
        # seq_h = [self.embedding(tf.gather(reversed_sess_item, index)) for index in range(batch_size)]
        seq_h = [tf.gather(params=item_embedding,
                           indices=tf.gather(reversed_sess_item, index)) for index in range(batch_size)]
        seq_h = tf.reshape(tf.concat(seq_h, axis=0), shape=(batch_size, tf.shape(reversed_sess_item)[1], self.emb_size))
        Xs = tf.divide(tf.reduce_sum(seq_h, axis=1), session_len)  # 这里不需要mask吗？ 前面的下顶使得第0行是0
        mask = tf.expand_dims(mask, -1)
        len_seq = tf.shape(seq_h)[1]

        Xs = tf.tile(tf.expand_dims(Xs, -2), multiples=(1, len_seq, 1))
        # nh = tf.nn.tanh(seq_h) 为什么直接连偏置都没有了，权重都没有了？
        Xt = self.w_1(seq_h)
        Xt = tf.nn.sigmoid(self.glu1(Xt) + self.glu2(Xs))

        beta = tf.matmul(Xt, self.f)
        beta = beta * mask
        theta_h = tf.reduce_sum(beta * seq_h, axis=1)  # 注意，这里与论文中的式子不符合

        # LineGraph
        session_emb_lg = self.LineGraph(self.embedding, D, A, session_item, session_len)
        con_loss =



