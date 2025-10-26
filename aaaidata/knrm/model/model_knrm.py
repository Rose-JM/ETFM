import sys
#sys.path.append(r'/home/dyx2/team2box/team2box/aaaidata')
import os
import time
sys.path.append('/home/dyx2/team2box')
sys.path.append('/home/dyx2/team2box/team2box/aaaidata/knrm/model')
import tensorflow as tf
import numpy as np
from ..data import DataGenerator
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
    Unicode,
)
import time
import argparse
from traitlets.config.loader import PyFileConfigLoader
from traitlets.config import Configurable
from team2box.aaaidata.knrm.model import BaseNN
from tensorflow.keras import Model
from tqdm import tqdm
"""
通过将 query 和 document 的词向量相似度映射到多个高斯核中（不同相似度段），提取匹配特征（soft-TF），并用这些特征训练一个排序模型
在 Team2Box 框架中，将其用于专家匹配问题，将 query 视为问题文本，将 document 表示为专家对应的回答内容，最后输出排序得分
"""
class Knrm(tf.keras.Model):
    neg_sample = 1
    # lamb = Float(0.5, help="guassian_sigma = lamb * bin_size").tag(config=True)
    # emb_in = Unicode('None', help="initial embedding. Terms should be hashed to ids.").tag(config=True)
    # learning_rate = Float(0.001, help="learning rate, default is 0.001").tag(config=True)
    # epsilon = Float(0.00001, help="Epsilon for Adam").tag(config=True)
    lamb = 0.5
    emb_in = 'None'
    learning_rate = 0.001
    epsilon = 0.00001

    def __init__(self,qmaxlen, dmaxlen, vocabsize, config,**kwargs):
        super().__init__(**kwargs)
        #super(Knrm, self).__init__(**kwargs)  # 不传 config 给 keras.Model
        #self.config = config
        self.base = BaseNN(config=config)  # 不继承 BaseNN，而是组合

        self.data_generator = self.base.data_generator
        self.val_data_generator = self.base.val_data_generator
        self.test_data_generator = self.base.test_data_generator

        self.max_q_len = qmaxlen
        self.max_d_len = dmaxlen
        self.vocabulary_size = vocabsize
        self.embedding_size = self.base.embedding_size
        self.batch_size = self.base.batch_size
        self.n_bins = self.base.n_bins

        self.max_epochs = config.get("max_epochs", 100)
        self.eval_frequency = config.get("eval_frequency", 10000)
        self.checkpoint_steps = config.get("checkpoint_steps", 10000)

        self.lamb = config.get("lamb", 0.5) if hasattr(config, "get") else 0.5
        self.emb_in = config.get("emb_in", "None") if hasattr(config, "get") else "None"
        self.learning_rate = config.get("learning_rate", 0.001) if hasattr(config, "get") else 0.001
        self.epsilon = config.get("epsilon", 1e-5) if hasattr(config, "get") else 1e-5
        #self.lamb = self.base.lamb
        #self.emb_in = self.base.emb_in
        #self.learning_rate = self.base.learning_rate
        #self.epsilon = self.base.epsilon

        # 将传入的参数 qmaxlen, dmaxlen 和 vocabsize 赋值
        self.mus = BaseNN.kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = BaseNN.kernel_sigmas(self.n_bins, self.lamb, use_exact=True)
        print("kernel sigma values: ", self.sigmas)

        print("trying to load initial embeddings from: ", self.emb_in)
        #初始化词嵌入矩阵
        if self.emb_in != 'None':
            self.emb = self.load_word2vec(self.emb_in)
            self.embeddings = tf.Variable(
                tf.constant(self.emb, dtype='float32',
                            shape=[self.vocabulary_size + 1, self.embedding_size]))
            print("Initialized embeddings with {0}".format(self.emb_in))

        else:
            self.embeddings = tf.Variable(tf.random.uniform([self.vocabulary_size + 1, self.embedding_size],
                                                            minval=-1.0, maxval=1.0),trainable=True)

        self.W1 = BaseNN.weight_variable([self.n_bins, 1])
        self.b1 = tf.Variable(tf.zeros([1]))

    def load_word2vec(self, emb_file_path):
        #加载 Word2Vec 词向量
        emb = np.random.uniform(low=-1, high=1, size=(self.vocabulary_size + 1, self.embedding_size))
        nlines = 0
        with open(emb_file_path) as f:
            for line in f:
                nlines += 1
                if nlines == 1:
                    continue
                items = line.split()
                tid = int(items[0])
                if tid > self.vocabulary_size:
                    print (tid)
                    continue
                vec = np.array([float(t) for t in items[1:]])
                emb[tid, :] = vec
                if nlines % 20000 == 0:
                    print ("load {0} vectors...".format(nlines))
        return emb

    #@tf.function
    @staticmethod
    def weight_variable(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial)


    def model(self, inputs_q, inputs_d, mask, q_weights, mu, sigma):
        """
        The pointwise model graph
        :param inputs_q: input queries. [nbatch, qlen, emb_dim]
        :param inputs_d: input documents. [nbatch, dlen, emb_dim]
        :param mask: a binary mask. [nbatch, qlen, dlen]
        :param q_weights: query term weigths. Set to binary in the paper.
        :param mu: kernel mu values.
        :param sigma: kernel sigma values.
        :return: return the predicted score for each <query, document> in the batch
        点式模型图：输入查询和文档 ID，通过嵌入查找、归一化、cos相似度计算、Gaussian 核计算后，
        汇聚生成匹配特征，再经过一层全连接得到最终匹配分数。
        """
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        q_embed = tf.nn.embedding_lookup(self.embeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(self.embeddings, inputs_d, name='demb')

        # 归一化
        # L2 normalize
        norm_q = tf.norm(q_embed, axis=2, keepdims=True)
        norm_d = tf.norm(d_embed, axis=2, keepdims=True)
        normalized_q_embed = q_embed / (norm_q + 1e-10)
        normalized_d_embed = d_embed / (norm_d + 1e-10)
        # 计算相似度矩阵
        sim = tf.matmul(normalized_q_embed, normalized_d_embed, transpose_b=True)

        # reshape similarity: [batch, qlen, dlen, 1]
        rs_sim = tf.expand_dims(sim, axis=-1)

        # 计算高斯核
        # mu = tf.reshape(mu, [1, 1, 1, self.n_bins])
        # sigma = tf.reshape(sigma, [1, 1, 1, self.n_bins])
        # tmp = tf.exp(-tf.square(rs_sim - mu) / (2 * tf.square(sigma)))
        mu = tf.reshape(mu, [1, 1, 1, self.n_bins])
        sigma = tf.reshape(sigma, [1, 1, 1, self.n_bins])
        tmp = tf.exp(-tf.square(rs_sim - mu) / (2 * tf.square(sigma)))

        # 掩码
        #tmp = tmp * tf.expand_dims(mask, axis=-1)
        print('tmp shape:', tmp.shape)
        print('mask shape:', mask.shape)
        # 修正mask
        mask = mask[:, :tf.shape(sim)[1], :tf.shape(sim)[2]]
        if len(mask.shape) == 3:
            mask = tf.expand_dims(mask, axis=-1)

        tmp = tmp * mask

        # 计算 soft-TF 特征
        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, axis=2)
        kde = tf.math.log(tf.maximum(kde, 1e-10)) * 0.01
        # [batch_size, qlen, n_bins]

        # weighted sum over query words
        #q_weights = tf.expand_dims(q_weights, axis=-1)  # [batch, qlen, 1]
        q_weights = q_weights[:, :tf.shape(kde)[1], :]
        # Support query-term weigting if set to continous values (e.g. IDF).
        # 聚合查询词
        aggregated_kde = tf.reduce_sum(kde * q_weights, axis=1)

        # Final features and scoring
        feats_tmp = aggregated_kde  # [batch, n_bins]
        print("batch feature shape:", feats_tmp.shape)
        # 连接特征
        feats_flat = tf.reshape(feats_tmp, [-1, self.n_bins])
        print("flat feature shape:", feats_flat.shape)

        o = tf.tanh(tf.matmul(feats_flat, self.W1) + self.b1)

        # 统计参数数量
        total_params = np.sum([np.prod(v.shape) for v in self.trainable_variables])
        print("Total trainable params:", total_params)
        return (sim, feats_flat), o

    def pad_mask(self, mask, batch_size):
        # mask shape: (batch_size, real_q_len, real_d_len)
        padded_mask = np.zeros((batch_size, self.max_q_len, self.max_d_len), dtype=np.float32)
        for i in range(mask.shape[0]):
            q_len = mask.shape[1]
            d_len = mask.shape[2]
            padded_mask[i, :q_len, :d_len] = mask[i, :q_len, :d_len]
        return padded_mask

    def train(self, train_pair_file_path, val_pair_file_path, train_size, checkpoint_dir, load_model=False):

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)

        # 加载模型
        if load_model:
            checkpoint_path = os.path.join(checkpoint_dir, 'data.ckpt')
            if os.path.exists(checkpoint_path + '.index'):
                self.load_weights(checkpoint_path)
                print("Model loaded from", checkpoint_path)
            else:
                print("Checkpoint not found, training from scratch.")

            # Loop through training steps.
        print("Epochs:", self.base.max_epochs)


        for epoch in range(int(self.base.max_epochs)):
            print(f"\n=== Epoch {epoch + 1}/{int(self.base.max_epochs)} ===")

            with open(train_pair_file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)


            pair_stream = all_lines  # 直接用list


            print(f"Total {total_lines} lines detected. Start reading with batch size {self.batch_size}...")

            step = 0
            epoch_loss = 0.0  # 累计 loss
            start_time = time.time()  # 开始计时

            for BATCH in tqdm(self.base.data_generator.pairwise_reader(pair_stream, self.batch_size, with_idf=True),
                              total=total_lines // self.batch_size,
                              desc="Reading Pairs",
                              ncols=100):
                X, Y = BATCH
                if X[u'idf'] is None or X[u'idf'].shape[0] == 0:
                    continue

                    # 自动跳过最后一小批，不足 batch_size 的
                # if X[u'q'].shape[0] != self.batch_size:
                #     continue
                real_batch_size = tf.shape(X['q'])[0]  # 动态 batch size

                Q = tf.convert_to_tensor(self.base.re_pad(X['q'], real_batch_size), dtype=tf.int32)
                D_pos = tf.convert_to_tensor(self.base.re_pad(X['d'], real_batch_size), dtype=tf.int32)
                D_neg = tf.convert_to_tensor(self.base.re_pad(X['d_aux'], real_batch_size), dtype=tf.int32)
                #QW = tf.convert_to_tensor(self.base.re_pad(X['idf'], real_batch_size), dtype=tf.float32)
                QW = tf.convert_to_tensor(self.pad_idf(X['idf'], real_batch_size), dtype=tf.float32)

                # M_pos = tf.convert_to_tensor(self.base.gen_mask(X['q'], X['d']), dtype=tf.float32)
                # M_neg = tf.convert_to_tensor(self.base.gen_mask(X['q'], X['d_aux']), dtype=tf.float32)
                M_pos = tf.convert_to_tensor(self.pad_mask(self.base.gen_mask(X['q'], X['d']), real_batch_size),
                                             dtype=tf.float32)
                M_neg = tf.convert_to_tensor(self.pad_mask(self.base.gen_mask(X['q'], X['d_aux']), real_batch_size),
                                             dtype=tf.float32)

                # reshape
                # rs_M_pos = tf.reshape(M_pos, [real_batch_size, self.max_q_len, self.max_d_len, 1])
                # rs_M_neg = tf.reshape(M_neg, [real_batch_size, self.max_q_len, self.max_d_len, 1])
                # rs_M_pos = tf.expand_dims(M_pos, axis=-1)
                # rs_M_neg = tf.expand_dims(M_neg, axis=-1)

                rs_qw = tf.reshape(QW, [real_batch_size, self.max_q_len, 1])

                mu = tf.convert_to_tensor(self.mus, dtype=tf.float32)
                sigma = tf.convert_to_tensor(self.sigmas, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    _, score_pos = self.model(Q, D_pos, M_pos, rs_qw, mu, sigma)
                    _, score_neg = self.model(Q, D_neg, M_neg, rs_qw, mu, sigma)
                    loss = tf.reduce_mean(tf.maximum(0.0, 1 - score_pos + score_neg))

                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                epoch_loss += loss.numpy()
                step += 1

                if (step + 1) % self.eval_frequency == 0:
                    val_loss = self.evaluate(val_pair_file_path)
                    print(f"[Step {step + 1}] Validation loss: {val_loss:.4f}")

                if (step + 1) % self.checkpoint_steps == 0:
                    self.save_weights(os.path.join(checkpoint_dir, 'data.ckpt'))
                    print(f"[Step {step + 1}] Model checkpoint saved.")
                # 打印当前进度百分比
                if step % 100 == 0:
                    percent = 100.0 * step * self.batch_size / train_size
                    print(f"[Epoch {epoch + 1}] Step {step}: approx. {percent:.2f}% of data processed")

            #pair_stream.close()
            if step > 0:
                avg_loss = epoch_loss / step
                duration = time.time() - start_time
                print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}, Time: {duration:.2f}s")
            else:
                print(f"Epoch {epoch + 1} skipped (no valid training batches).")

        # 最终保存
        self.save_weights(os.path.join(checkpoint_dir, 'data.ckpt'))
        print("Training completed and final model saved.")

    def pad_mask(self, mask, real_batch_size):
        """pad mask到固定大小"""
        if isinstance(real_batch_size, tf.Tensor):
            real_batch_size = int(real_batch_size.numpy())
        padded = np.zeros((real_batch_size, self.max_q_len, self.max_d_len), dtype=np.float32)
        for i in range(mask.shape[0]):
            padded[i, :mask.shape[1], :mask.shape[2]] = mask[i]
        return padded

    def pad_idf(self, idf, real_batch_size):

        if isinstance(real_batch_size, tf.Tensor):
            real_batch_size = int(real_batch_size.numpy())
        padded = np.zeros((real_batch_size, self.max_q_len), dtype=np.float32)
        for i in range(idf.shape[0]):
            padded[i, :idf.shape[1]] = idf[i]
        return padded

    def load_weights(self, checkpoint_path,**kwargs):
        # 创建检查点对象
        #checkpoint = tf.train.Checkpoint(model=self)
        # 恢复模型权重
        #checkpoint.restore(checkpoint_path).assert_consumed()
        tf.keras.Model.load_weights(checkpoint_path,**kwargs).expect_partial()
        print(f"Model weights loaded from {checkpoint_path}")

    def save_weights(self, checkpoint_path,**kwargs):
        # 创建检查点对象
        #checkpoint = tf.train.Checkpoint(model=self)
        tf.keras.Model.save_weights(self, checkpoint_path, **kwargs)
        print(f"Model weights saved to {checkpoint_path}")
        # 保存模型权重
        #checkpoint.save(checkpoint_dir + '/data.ckpt')


    def evaluate(self, val_pair_file_path):
        #模型验证 读取验证集并计算平均验证 loss
        val_loss = 0
        n_val_batch = 0
        val_pair_stream = open(val_pair_file_path)
        for BATCH in self.val_data_generator.pairwise_reader(val_pair_stream, self.batch_size, with_idf=True):
            X_val, Y_val = BATCH

            real_batch_size = tf.shape(X_val['q'])[0]

            Q = tf.convert_to_tensor(self.re_pad(X_val[u'q'], real_batch_size), dtype=tf.int32)
            D_pos = tf.convert_to_tensor(self.re_pad(X_val[u'd'], real_batch_size), dtype=tf.int32)
            D_neg = tf.convert_to_tensor(self.re_pad(X_val[u'd_aux'], real_batch_size), dtype=tf.int32)
            QW = tf.convert_to_tensor(self.re_pad(X_val[u'idf'], real_batch_size), dtype=tf.float32)

            M_pos = tf.convert_to_tensor(self.gen_mask(X_val[u'q'], X_val[u'd']), dtype=tf.float32)
            M_neg = tf.convert_to_tensor(self.gen_mask(X_val[u'q'], X_val[u'd_aux']), dtype=tf.float32)

            rs_M_pos = tf.reshape(M_pos, [real_batch_size, self.max_q_len, self.max_d_len, 1])
            rs_M_neg = tf.reshape(M_neg, [real_batch_size, self.max_q_len, self.max_d_len, 1])
            rs_qw = tf.reshape(QW, [real_batch_size, self.max_q_len, 1])

            mu = tf.convert_to_tensor(self.mus, dtype=tf.float32)
            sigma = tf.convert_to_tensor(self.sigmas, dtype=tf.float32)

            _, score_pos = self.model(Q, D_pos, rs_M_pos, rs_qw, mu, sigma)
            _, score_neg = self.model(Q, D_neg, rs_M_neg, rs_qw, mu, sigma)

            loss = tf.reduce_mean(tf.maximum(0.0, 1 - score_pos + score_neg))
            val_loss += loss.numpy()
            n_val_batch += 1

        val_pair_stream.close()
        return val_loss / n_val_batch



    def test(self, test_point_file_path, test_size, output_file_path, checkpoint_dir=None, load_model=False):

        # 加载权重（如果需要）
        if load_model and checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, 'data.ckpt')
            self.load_weights(checkpoint_path)
            print("Model loaded from", checkpoint_path)
        else:
            print("Using current model parameters.")

        # 读取测试数据
        with open(test_point_file_path, 'r') as test_point_stream, open(output_file_path, 'w') as outfile:
            num_batches = int(np.ceil(float(test_size) / self.batch_size))  # 计算总的批次数

            for _ in range(num_batches):
                # 获取一批数据
                X, _ = next(self.test_data_generator.pointwise_generate(
                    test_point_stream, self.batch_size, with_idf=True, with_label=False))

                # 构建输入张量
                test_inputs_q = tf.convert_to_tensor(self.re_pad(X['q'], self.batch_size), dtype=tf.int32)
                test_inputs_d = tf.convert_to_tensor(self.re_pad(X['d'], self.batch_size), dtype=tf.int32)
                test_input_q_weights = tf.convert_to_tensor(self.re_pad(X['idf'], self.batch_size), dtype=tf.float32)
                test_mask = tf.convert_to_tensor(self.gen_mask(X['q'], X['d']), dtype=tf.float32)

                # 调整形状
                rs_test_mask = tf.reshape(test_mask, [self.batch_size, self.max_q_len, self.max_d_len, 1])
                rs_q_weights = tf.reshape(test_input_q_weights, [self.batch_size, self.max_q_len, 1])
                mu = tf.reshape(tf.convert_to_tensor(self.mus, dtype=tf.float32), [1, 1, self.n_bins])
                sigma = tf.reshape(tf.convert_to_tensor(self.sigmas, dtype=tf.float32), [1, 1, self.n_bins])

                # 前向传播
                _, scores = self.model(test_inputs_q, test_inputs_d, rs_test_mask, rs_q_weights, mu, sigma)

                # 写入预测结果
                for score in scores.numpy():
                    outfile.write(f"{score[0]}\n")

        print("Testing completed. Results saved to", output_file_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path", help="Path to the .py config file")

    # training args
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_file", "-f", help="Training pair file path")
    parser.add_argument("--validation_file", "-v", help="Validation pair file path")
    parser.add_argument("--train_size", "-z", type=int, help="Number of training samples")
    parser.add_argument("--load_model", "-l", action='store_true', help="Load pretrained model for fine-tuning")

    # test args
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test_file", help="Test pointwise file path")
    parser.add_argument("--test_size", type=int, default=0)
    parser.add_argument("--output_score_file", "-o", help="Output path for predicted scores")

    # embeddings & checkpoint
    parser.add_argument("--emb_file_path", "-e", help="Optional word2vec embeddings")
    parser.add_argument("--checkpoint_dir", "-s", required=True, help="Directory to save/load checkpoints")

    args = parser.parse_args()

    # 加载配置对象（traitlets 格式）
    conf = PyFileConfigLoader(args.config_file_path).load_config()

    # 可选地指定超参数（也可以从 config 中读取）
    qmaxlen = getattr(conf['Knrm'], 'qmaxlen', 20)
    dmaxlen = getattr(conf['Knrm'], 'dmaxlen', 100)
    vocabsize = getattr(conf['Knrm'], 'vocabulary_size', 50000)

    # 初始化模型
    model = Knrm(qmaxlen=qmaxlen, dmaxlen=dmaxlen, vocabsize=vocabsize, config=conf)

    # 如果指定了 word2vec 初始化路径
    if args.emb_file_path:
        model.emb_in = args.emb_file_path

    # 训练模式
    if args.train:
        model.train(
            train_pair_file_path=args.train_file,
            val_pair_file_path=args.validation_file,
            train_size=args.train_size,
            checkpoint_dir=args.checkpoint_dir,
            load_model=args.load_model
        )

    # 测试模式
    elif args.test:
        model.test(
            test_point_file_path=args.test_file,
            test_size=args.test_size,
            output_file_path=args.output_score_file,
            load_model=True,
            checkpoint_dir=args.checkpoint_dir
        )
    else:
        print("Please specify either --train or --test.")
