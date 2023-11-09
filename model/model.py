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
        for i in range(self.layers):
            item_embeddings = tf.sparse.sparse_dense_matmul(adj, item_embeddings)
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
        item_embeddings = tf.concat([zeros, item_embeddings], axis=0)
        seq_h = item_embeddings(session_item)
        # for i in tf.range(tf.shape(session_item)[0]):
        #     seq_h.append(item_embeddings(tf.gather(session_item, indices=i)))
        session_emb_lgcn = tf.divide(tf.reduce_sum(seq_h, axis=1), session_len)
        session = [session_emb_lgcn]
        DA = tf.matmul(D, A)
        for i in range(self.layers):
            session_emb_lgcn = tf.matmul(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        session_emb_lgcn = tf.add_n(session) / (self.layers + 1)
        return session_emb_lgcn


class DHCN(keras.models.Model):
    def __init__(self, adj, n_node, layers, beta, emb_size=100):
        super().__init__()

