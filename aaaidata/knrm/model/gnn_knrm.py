"""
GNN-KNRM 实现 (TensorFlow 2.x)
说明：
    - 使用 GCN 作为图表示学习模块，
    - 使用 KNRM 模块进行匹配建模，
    - 二者结合用于 Query-Document 排序。
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class GCNLayer(tf.keras.layers.Layer):
    """
    图卷积层（Graph Convolution Layer）
    输入：X（节点特征矩阵），A_hat（预处理后的邻接矩阵）
    输出：新的节点表示 H
    """
    def __init__(self, output_dim, activation=tf.nn.relu):
        super(GCNLayer, self).__init__()
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.w = self.add_weight(shape=(input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        x, a_hat = inputs  # x: (N, F), a_hat: (N, N)
        xw = tf.matmul(x, self.w)  # (N, output_dim)
        out = tf.matmul(a_hat, xw)  # 图卷积核心操作
        return self.activation(out)


class KNRM(tf.keras.Model):
    """
    核心匹配模块 KNRM：将 query 和 doc 各自嵌入后计算匹配特征，
    通过高斯核进行 soft-TF 匹配，最后输出匹配得分。
    """
    def __init__(self, embedding_matrix, n_kernels=11):
        super(KNRM, self).__init__()
        self.emb = tf.Variable(initial_value=embedding_matrix,
                               trainable=False,
                               dtype=tf.float32)
        self.n_kernels = n_kernels
        self.kernel_mu = self._kernel_mus()
        self.kernel_sigma = self._kernel_sigmas()
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def _kernel_mus(self):
        l = 2.0
        mu = [1.0]  # exact match
        if self.n_kernels == 1:
            return np.array(mu, dtype=np.float32)
        bin_size = 2.0 / (self.n_kernels - 1)
        mu.append(1 - bin_size / 2)
        for i in range(1, self.n_kernels - 1):
            mu.append(mu[i] - bin_size)
        return np.array(mu[::-1], dtype=np.float32)

    def _kernel_sigmas(self):
        if self.n_kernels == 1:
            return np.array([0.001], dtype=np.float32)
        return np.array([0.001] + [0.1] * (self.n_kernels - 1), dtype=np.float32)

    def call(self, q, d):
        q_embed = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.emb, q), -1)
        d_embed = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.emb, d), -1)

        sim = tf.matmul(q_embed, d_embed, transpose_b=True)  # (B, q_len, d_len)

        KM = []
        for mu, sigma in zip(self.kernel_mu, self.kernel_sigma):
            tmp = tf.exp(- (sim - mu) ** 2 / (2 * (sigma ** 2)))
            pooling_sum = tf.reduce_sum(tmp, axis=2)  # sum over document terms
            log_pooling = tf.math.log(tf.maximum(pooling_sum, 1e-10)) * 0.01
            KM.append(tf.reduce_sum(log_pooling, axis=1))  # sum over query terms
        phi = tf.stack(KM, axis=1)
        return self.dense(phi)


class GNN_KNRM_Model(tf.keras.Model):
    """
    整体模型：GCN 编码图结构 → KNRM 做排序匹配
    用于 Query-Expert / Query-Doc 排序任务
    """
    def __init__(self, vocab_size, emb_dim, embedding_matrix, gcn_units=64, n_kernels=11):
        super(GNN_KNRM_Model, self).__init__()
        self.gcn1 = GCNLayer(gcn_units)
        self.gcn2 = GCNLayer(gcn_units)
        self.knrm = KNRM(embedding_matrix, n_kernels)

        self.node_embeddings = tf.Variable(tf.random.uniform([vocab_size, emb_dim], -0.1, 0.1), trainable=True)

    def call(self, inputs):
        query_ids, doc_ids, adj = inputs  # adj: normalized A_hat
        # 用图卷积更新 node 表示
        h = self.gcn1([self.node_embeddings, adj])
        h = self.gcn2([h, adj])

        # 用更新后的 node embeddings 替代 word embeddings
        q_embed = tf.nn.embedding_lookup(h, query_ids)
        d_embed = tf.nn.embedding_lookup(h, doc_ids)

        # 送入 KNRM 模块做匹配
        return self.knrm(query_ids, doc_ids)
