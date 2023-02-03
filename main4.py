import os
import shutil
import sys
import ipdb
import numpy as np
from scipy import sparse
import scipy.sparse as sp
# import seaborn as sn
# sn.set()
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
# from tensorflow.contrib.layers import apply_regularization, l2_regularizer
# import bottleneck as bn
from evaluate import *
import scipy
import argparse
from collections import Counter
import math

@nb.njit('float32[:,::1](float32[:,::1])', parallel=True)
def fastOrder(a):
    b = np.empty(a.shape, dtype=np.float32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.sort(a[i,:])
    return b

class MultiVAE(object):
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, lr2=1e-3, n_intent=10, dim=200, params=None, random_seed=None):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]

        self.lam = lam
        self.lr = lr
        self.lr2 = lr2
        self.max_len = args.max_len
        self.random_seed = random_seed
        self.n_intent = n_intent
        self.dim = dim
        self.n_users, self.n_items = params.n_users, params.n_items
        self.ssl_ratio = params.ssl_ratio
        self.aug_type = 1
        self.training_user, self.training_item = params.training_user, params.training_item
        self.ratings = params.ratings

        self.construct_placeholders()

    def construct_placeholders(self):
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
        self.keep_prob_ph2 = tf.placeholder_with_default(1.0, shape=None)

        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.is_training = tf.cast(self.is_training_ph, dtype=tf.bool)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)

        self.pad_train = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
        self.pad_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.max_len])
        self.input_ph2 = tf.placeholder(dtype=tf.float32, shape=[None, self.q_dims[0]])

        self.beta = tf.get_variable(name="beta", shape=[self.p_dims[0], self.q_dims[0]],
                                 initializer=tf.keras.initializers.glorot_normal(seed=self.random_seed))
        self.items = tf.get_variable(name="items", shape=[self.q_dims[0], self.dim],
                                         initializer=tf.keras.initializers.glorot_normal(seed=self.random_seed))
        self.att1 = tf.get_variable(name="att1", shape=[self.q_dims[0], self.p_dims[0]],
                                         initializer=tf.keras.initializers.glorot_normal(seed=self.random_seed))
        self.att2 = tf.get_variable(name="att2", shape=[self.p_dims[0], self.p_dims[0]],
                                         initializer=tf.keras.initializers.glorot_normal(seed=self.random_seed))
        self.temp = tf.placeholder_with_default(1., shape=None)

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], []

        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            bias_key = "bias_q_{}".format(i + 1)

            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.keras.initializers.glorot_normal(
                    seed=self.random_seed)))

            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

            # add summary stats
            tf.summary.histogram(weight_key, self.weights_q[-1])
            tf.summary.histogram(bias_key, self.biases_q[-1])

        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i + 1)
            bias_key = "bias_p_{}".format(i + 1)
            self.weights_p.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.keras.initializers.glorot_normal(
                    seed=self.random_seed)))

            self.biases_p.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

            # add summary stats
            tf.summary.histogram(weight_key, self.weights_p[-1])
            tf.summary.histogram(bias_key, self.biases_p[-1])

        self.weights_k, self.biases_k = [], []
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out = self.dim
                d_out *= 2  # mu & var
            weight_key = "weight_k_{}to{}".format(i, i + 1)
            self.weights_k.append(tf.get_variable(name=weight_key, shape=[d_in, d_out],
                                                  initializer=tf.keras.initializers.glorot_normal(seed=self.random_seed)))
            bias_key = "bias_k_{}".format(i + 1)
            self.biases_k.append(tf.get_variable(name=bias_key, shape=[d_out],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))

    def q_graph_k(self, x):
        mu_q, std_q, kl = None, None, None
        h = tf.nn.l2_normalize(x, -1)
        h = tf.nn.dropout(h, self.keep_prob_ph2)

        # norm loss
        self.norm_loss = 0

        for i, (w, b) in enumerate(zip(self.weights_k, self.biases_k)):
            h = tf.matmul(h, w, a_is_sparse=(i == 0)) + b

            # x = tf.transpose(h[:, :self.dim])
            # x_reducemean = x - tf.reduce_mean(x, axis=1, keep_dims=True)
            # numerator = tf.matmul(x_reducemean, tf.transpose(x_reducemean))
            # no = tf.norm(x_reducemean, ord=2, axis=1, keepdims=True)
            # denominator = tf.matmul(no, tf.transpose(no))
            # corrcoef = numerator / (denominator + 1e-10)
            # self.norm_loss += tf.norm(tf.linalg.band_part(corrcoef, 0, -1) - tf.diag(tf.diag_part(corrcoef)))

            if i != len(self.weights_k) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.dim]
                logvar_q = h[:, self.dim:]

                # # batchnorm
                # mu_q = tf.layers.batch_normalization(mu_q, scale=True, center=True, epsilon=1e-8)
                # logvar_q = tf.layers.batch_normalization(logvar_q, scale=False, center=True, epsilon=1e-8)

                std_q = tf.exp(0.5 * logvar_q) * 1
                kl = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q ** 2 - 1), axis=1))

            # x = tf.random_shuffle(h[:, :self.dim])
            # x = x - tf.reduce_mean(x, axis=0, keep_dims=True)
            # cov = tf.matmul(tf.transpose(x), x)
            # I_k = tf.eye(tf.shape(x)[1]) / tf.sqrt(tf.cast(tf.shape(x)[1], dtype=tf.float32))
            # self.norm_loss += tf.norm(cov / tf.norm(cov) - I_k)

        return mu_q, std_q, kl

    def build_graph(self):
        self._construct_weights()

        saver = tf.train.Saver(max_to_keep=50)
        logits, self.KL, self.rating, self.kl_k, self.rating_kloss = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)
        # log_softmax_var = tf.log(logits+1e-10)

        self.neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * self.input_ph, axis=-1))

        # reg = l2_regularizer(self.lam)
        # reg_var = apply_regularization(reg, self.weights_q + self.weights_p)

        # self.phi_theta_kl_loss = tf.constant(0, dtype=tf.float32)
        self.pad_phi = tf.nn.embedding_lookup(self.phi, self.pad_train)
        self.pad_theta = tf.expand_dims(self.theta, axis=1)
        self.phi_theta_kl_loss = tf.reduce_mean(tf.reduce_sum(self.pad_mask * tf.reduce_sum(
            self.pad_phi * tf.log(((self.pad_phi) / ((self.pad_theta) + 1e-10)) + 1e-10), -1), -1))

        self.recon_loss = - tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(self.rating) * self.input_ph2, axis=-1))
        # self.recon_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square((self.input_ph2 - self.rating) * self.input_ph), axis=-1))

        # update_list = self.weights_k + self.biases_k
        # update_list.append(self.items)
        # update_list.append(self.att1)
        # update_list.append(self.att2)
        # for var in tf.trainable_variables():
        #     for key in ['dense', 'phi']:
        #         if key in var.name:
        #             update_list.append(var)

        self.loss1 = self.neg_ll + self.anneal_ph * self.KL + self.phi_theta_kl_loss
        # self.loss1 = self.neg_ll + self.anneal_ph * self.KL
        # self.loss2 = self.recon_loss + self.anneal_ph * self.kl_k
        # self.loss2 = self.rating_kloss + self.anneal_ph * self.kl_k
        neg_ELBO = args.bal * (self.neg_ll + self.KL + self.phi_theta_kl_loss) + self.recon_loss + self.anneal_ph * self.kl_k
        # neg_ELBO = args.bal * (self.neg_ll + self.KL + self.phi_theta_kl_loss) + self.recon_loss + self.anneal_ph * self.kl_k + self.rating_kloss

        neg_ELBO += args.bal2 * self.norm_loss
        neg_ELBO += args.ssl_reg * self.ssl_loss_user

        # train_op = tf.train.AdamOptimizer(self.lr2).minimize(neg_ELBO)
        self.train_op1 = tf.train.AdamOptimizer(self.lr).minimize(self.loss1)
        # self.train_op2 = tf.train.AdamOptimizer(self.lr).minimize(self.loss2, var_list=update_list)
        self.train_op2 = tf.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', self.neg_ll)
        tf.summary.scalar('KL', self.KL)
        # tf.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.summary.merge_all()

        return saver, logits, self.loss1, merged

    def forward_pass(self):
        # 隐式反馈部分
        # q-network
        self.alpha = None
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                # self.alpha = tf.nn.softplus(h[:, :self.q_dims[-1]])
                # # self.alpha = h

                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q ** 2 - 1), axis=1))

        epsilon = tf.random_normal(tf.shape(std_q))
        self.sampled_z = mu_q + self.is_training_ph * epsilon * std_q
        # # logits = tf.matmul(self.sampled_z, self.beta, transpose_b=False)
        self.theta = tf.nn.softmax(self.sampled_z / self.temp, dim=-1)
        # self.theta = tf.nn.softmax(mu_q / self.temp, dim=-1)
        self.z = self.sampled_z
        # self.z = tf.nn.softmax(self.sampled_z, dim=-1)

        # self.alpha = tf.layers.batch_normalization(self.alpha)
        # self.alpha = tf.maximum(1e-6, tf.log(1. + tf.exp(self.alpha)))
        # gamma = args.prior * tf.ones_like(self.alpha)
        # pst_dist = tf.distributions.Dirichlet(self.alpha)
        # pri_dist = tf.distributions.Dirichlet(gamma)
        #
        # mean_z = pst_dist.mean()
        # self.sampled_z = pst_dist.sample()
        # # self.z = self.sampled_z
        # self.z = self.alpha
        # # self.z = self.alpha/(tf.reduce_sum(self.alpha, axis=1, keep_dims=True) + 1e-10)
        # # self.z = tf.nn.softmax(self.alpha, dim=-1)
        # # self.theta = self.is_training_ph * self.sampled_z + (1 - self.is_training_ph) * mean_z
        # self.theta = mean_z
        # KL = tf.reduce_mean(pst_dist.kl_divergence(pri_dist), -1)
        # # KL = tf.constant(0, dtype=tf.float32)

        # p-network
        # self.beta_norm = tf.nn.softplus(tf.layers.batch_normalization(self.beta))
        # logits = tf.matmul(self.z, tf.nn.softmax(self.beta, dim=-1), transpose_b=False) / 1
        logits = tf.matmul(self.z, self.beta, transpose_b=False) / 1

        # 三种Phi学习方式，前两种是利用第一层VAE去学习表征，分带有NORM和不带的，第三种是直接自定义随即向量
        # self.phi = tf.nn.softmax(tf.layers.batch_normalization(tf.layers.dense(self.weights_q[0], self.q_dims[-1], name='dense'),
        #                center=True, scale=False, training=self.is_training, trainable=True, name='phi') / self.temp)
        self.phi = tf.nn.softmax((tf.layers.dense(self.weights_q[0], self.q_dims[-1], name='dense')) / self.temp)
        # self.phi = tf.nn.softmax(tf.get_variable(name="phi", shape=[self.q_dims[0], self.q_dims[-1]], initializer=tf.keras.initializers.glorot_normal(seed=self.random_seed)) / self.temp)

        # 显式反馈部分
        # rating, kl_k, rating_kloss = tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)

        self.topk_intent = tf.nn.top_k(self.theta, k=self.n_intent)
        # self.phiT = tf.transpose(self.phi)
        self.phiT = tf.transpose(tf.stop_gradient(self.phi))
        self.cates_k = tf.nn.embedding_lookup(self.phiT, self.topk_intent[1])
        self.cates_k = self.cates_k / (tf.reduce_sum(self.cates_k, axis=1, keep_dims=True) + 1e-10)

        # q-network
        x_k = tf.expand_dims(self.input_ph2, axis=1) * self.cates_k
        self.see = x_k

        x_k = tf.reshape(x_k, (-1, self.q_dims[0]))
        mu_k, std_k, kl_k = self.q_graph_k(x_k)
        epsilon_k = tf.random_normal(tf.shape(std_k))
        z_k = mu_k + self.is_training_ph * epsilon_k * std_k

        # 对比学习
        h1 = tf.nn.l2_normalize(self.input_ph2, -1)
        h2 = tf.nn.l2_normalize(x_k, -1)
        h2 = tf.nn.dropout(h2, self.keep_prob_ph2)
        for i, (w, b) in enumerate(zip(self.weights_k, self.biases_k)):
            h1 = tf.matmul(h1, w, a_is_sparse=(i == 0)) + b
            h2 = tf.matmul(h2, w, a_is_sparse=(i == 0)) + b
            if i != len(self.weights_k) - 1:
                h1 = tf.nn.tanh(h1)
                h2 = tf.nn.tanh(h2)
            else:
                user_emb1 = h1[:, :self.dim]
                user_emb2 = h2[:, :self.dim]
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
        ttl_score_user = tf.matmul(normalize_user_emb1, normalize_user_emb2, transpose_a=False, transpose_b=True)
        # normalize_user_emb1 = tf.reshape(tf.tile(normalize_user_emb1, [1, self.n_intent]), (-1, self.p_dims[0]))
        pos_score_user = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.expand_dims(normalize_user_emb1, axis=1), tf.reshape(normalize_user_emb2, (-1, self.n_intent, self.p_dims[0]))), axis=-1), axis=-1)
        pos_score_user = tf.exp(pos_score_user / args.ssl_temp)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / args.ssl_temp), axis=1)
        self.ssl_loss_user = -tf.reduce_mean(tf.log(pos_score_user / ttl_score_user))


        # p-network
        rating_k = tf.matmul(z_k, self.items, transpose_b=True)

        # x = tf.transpose(rating_k)
        # x_reducemean = x - tf.reduce_mean(x, axis=1, keep_dims=True)
        # numerator = tf.matmul(x_reducemean, tf.transpose(x_reducemean))
        # no = tf.norm(x_reducemean, ord=2, axis=1, keepdims=True)
        # denominator = tf.matmul(no, tf.transpose(no))
        # corrcoef = numerator / (denominator + 1e-10)
        # self.norm_loss = tf.norm(tf.linalg.band_part(corrcoef, 0, -1) - tf.diag(tf.diag_part(corrcoef)))

        rating_kloss = - tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(rating_k) * x_k, axis=-1))
        rating_k = tf.reshape(rating_k, (-1, self.n_intent, self.q_dims[0]))
        self.topk_intent_norm = self.topk_intent[0] / (tf.reduce_sum(self.topk_intent[0], axis=1, keep_dims=True) + 1e-10)
        rating = tf.reduce_sum(rating_k * tf.expand_dims(self.topk_intent_norm, axis=-1), axis=1)



        return logits, KL, rating, kl_k, rating_kloss

parser = argparse.ArgumentParser(description="Options")
parser.add_argument('-kkk', action='store', dest='kkk', default=0)
parser.add_argument('-dev', action='store', dest='dev', default='1')
parser.add_argument('-result', action='store', dest='result', default='test')
parser.add_argument('-result2', action='store', dest='result2', default='')
parser.add_argument('-lam', action='store', dest='lam', default=0.0, type=np.float32)
parser.add_argument('-lr', action='store', dest='lr', default=0.001, type=np.float32)
parser.add_argument('-lr2', action='store', dest='lr2', default=0.0003, type=np.float32)
parser.add_argument('-step', action='store', dest='step', default=20, type=int)
parser.add_argument('-cap', action='store', dest='cap', default=1, type=np.float32)
parser.add_argument('-keep', action='store', dest='keep', default=0.5, type=np.float32)
parser.add_argument('-keep2', action='store', dest='keep2', default=0.5, type=np.float32)
parser.add_argument('-prior', action='store', dest='prior', default=0.5, type=np.float32)
parser.add_argument('-int', action='store', dest='int', default=5, type=int)
parser.add_argument('-dim', action='store', dest='dim', default=200, type=int)
parser.add_argument('-epoch', action='store', dest='epoch', default=200, type=int)
parser.add_argument('-lepoch', action='store', dest='lepoch', default=180, type=int)
parser.add_argument('-batch', action='store', dest='batch', default=500, type=int)
parser.add_argument('-bal', action='store', dest='bal', default=1, type=np.float32)
parser.add_argument('-bal2', action='store', dest='bal2', default=1, type=np.float32)
parser.add_argument('-temp', action='store', dest='temp', default=0.3, type=np.float32)
parser.add_argument('-anneal', action='store', dest='anneal', default=0.0006, type=np.float32)
parser.add_argument('-restore', action='store_true', default=False)
parser.add_argument('-ssl', type=float, default=0.1)
parser.add_argument('-layers', type=int, default=3, help='the number of epochs to train for')
parser.add_argument('-max_len', action='store', dest='max_len', default=20, type=int)
parser.add_argument('-ssl_temp', action='store', type=float, default=0.2)
parser.add_argument('-ssl_reg', action='store', type=float, default=0.001)
args = parser.parse_args()
kkk = args.kkk
os.environ['CUDA_VISIBLE_DEVICES'] = args.dev  # FLAGS.gpu_id, 指定gpu0卡

class Params:
    def __init__(self):
        self.n_layers = args.layers
        self.max_len = args.max_len
        self.ssl_ratio = args.ssl
params = Params()

# test_data = pd.read_csv("data/new5movie_test" + str(kkk) + ".csv", header=None).values[:, 0:2] - 1
# train_data = pd.read_csv("data/new5movie_train" + str(kkk) + ".csv", header=None).values[:, 0:2] - 1
# n_users, n_items = np.max(train_data, axis=0) + 1

# test_data = pd.read_csv("data/ml-1m/pro_sg/test_te.csv", header=0).values[:, 0:2]
# test_data_tr = pd.read_csv("data/ml-1m/pro_sg/test_tr.csv", header=0).values[:, 0:2]
# train_data = pd.read_csv("data/ml-1m/pro_sg/train.csv", header=0).values[:, 0:2]
# N = np.max(train_data[:, 0], axis=0) + 1
# train_data = np.concatenate((train_data, test_data_tr), axis=0)
# n_users, n_items = np.max(np.concatenate((train_data, test_data), axis=0), axis=0) + 1

vad_csv = pd.read_csv("data/ex_ml1m_test" + str(kkk) + ".csv", header=None).values[:, 0:3]
train_csv = pd.read_csv("data/ex_ml1m_train" + str(kkk) + ".csv", header=None).values[:, 0:3]
# vad_csv = pd.read_csv("data/ex_movie_test" + str(kkk) + ".csv", header=None).values[:, 0:3]
# train_csv = pd.read_csv("data/ex_movie_train" + str(kkk) + ".csv", header=None).values[:, 0:3]
# vad_csv = pd.read_csv("data/ex_yahoo_test" + str(kkk) + ".csv", header=None).values[:, 0:3]
# train_csv = pd.read_csv("data/ex_yahoo_train" + str(kkk) + ".csv", header=None).values[:, 0:3]
train_csv[:, 0:2] -= 1
vad_csv[:, 0:2] -= 1
n_users, n_items = np.max(train_csv[:, 0:2], axis=0) + 1
vad_csv = vad_csv[vad_csv[:, 2] >= 4, :]
# train_data = train_csv[train_csv[:, 2] >= 4, :]
train_data = train_csv
test_data = vad_csv
max_rating = np.max(train_csv[:, 2])

postrain = defaultdict(list)
for line in train_csv:
    postrain[line[0]].append(line[1])
posprobe = defaultdict(list)
for line in test_data:
    posprobe[line[0]].append(line[1])

params.ratings = train_csv[:, 2]/max_rating
train_data1 = sparse.csr_matrix((np.ones_like(train_data[:, 0]), (train_data[:, 0], train_data[:, 1])), dtype='float32', shape=(n_users, n_items))
train_data2 = sparse.csr_matrix((params.ratings, (train_csv[:, 0], train_csv[:, 1])), dtype='float32', shape=(n_users, n_items))
max_len = 0
pad_train = []
pad_mask = []

# 此处由于列表是排序后的，所以不需要再对字典的key排序,但为了防止不规则数据，还是做了排序，注意用户在训练集里不能有空的，否则序号对不上
postrain = dict(sorted(postrain.items(), key=lambda x: x[0]))
for li in postrain.values():
    if max_len < len(li):
        max_len = len(li)
for li in postrain.values():
    pad_train.append(li + [0] * (max_len - len(li)))
    pad_mask.append([1] * len(li) + [0] * (max_len - len(li)))
pad_train = np.array(pad_train)
pad_mask = np.array(pad_mask)

N = train_data1.shape[0]
idxlist = list(range(N))

# training batch size
batch_size = args.batch
batches_per_epoch = int(np.ceil(float(N) / batch_size))
print('n_users, n_items:', n_users, n_items, 'batch_size, batches_per_epoch:', batch_size, batches_per_epoch)
batch_size_vad = args.batch
total_anneal_steps = args.step
anneal_cap = args.cap
n_epochs = args.epoch
ANNEAL_RATE = args.anneal

p_dims = [200, 600, n_items]
params.n_users, params.n_items = n_users, n_items
dok_matrix = train_data2.todok()
params.training_user, params.training_item = [], []
for (user, item), value in dok_matrix.items():
    params.training_user.append(user)
    params.training_item.append(item)
tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=args.lam, lr=args.lr, lr2=args.lr2, n_intent=args.int, dim=args.dim, params=params, random_seed=98765)

saver, logits_var, loss_var, merged_var = vae.build_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    negtrainmatrix = (9999999 * train_data1.toarray()).astype('float32')
    recall, precision, map, ndcg = [], [], [], []
    if args.restore:
        load_epoch = args.lepoch
        saver.restore(sess, "model/disent-{}-{}".format(args.result, str(load_epoch)))
        # update_count = load_epoch * batches_per_epoch
        update_count = 0.0
        for epoch in range(300):
            np.random.shuffle(idxlist)
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)

                X = train_data1[idxlist[st_idx:end_idx]]
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')
                y = train_data2[idxlist[st_idx:end_idx]]
                if sparse.isspmatrix(y):
                    y = y.toarray()
                y = y.astype('float32')
                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                else:
                    anneal = anneal_cap
                temp = args.temp

                feed_dict = {vae.input_ph: X, vae.keep_prob_ph: args.keep, vae.keep_prob_ph2: args.keep2,
                             vae.anneal_ph: anneal, vae.temp: temp, vae.is_training_ph: 1,
                             vae.pad_train: pad_train[idxlist[st_idx:end_idx]],
                             vae.pad_mask: pad_mask[idxlist[st_idx:end_idx]], vae.input_ph2: y}
                print('training batch', bnum)
                _, l1, l2, l3, l4, l5, l6 = sess.run([vae.train_op2, vae.neg_ll, vae.KL, vae.recon_loss, vae.kl_k, vae.phi_theta_kl_loss, vae.ssl_loss_user], feed_dict=feed_dict)
                print(l1, l2, l3, l4, l5, l6)
                update_count += 1
            if epoch > 90 and epoch < 310 and epoch % 50 == 0:
                saver.save(sess, "model/disent-ex-{}-{}-{}".format(args.result, args.result2, str(epoch)))

            X = train_data1
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
            y = train_data2
            if sparse.isspmatrix(y):
                y = y.toarray()
            y = y.astype('float32')
            # a, b, c, d, e, epcoh_rating = sess.run((vae.theta, vae.phi, vae.topk_intent, vae.see, vae.cates_k, logits_var),
            #     feed_dict={vae.input_ph: X, vae.input_ph2: y, vae.temp: args.temp})
            # print(fastOrder(a), fastOrder(b), c, d, e)
            epcoh_rating, epcoh_rating2 = sess.run((logits_var, vae.rating), feed_dict={vae.input_ph: X, vae.input_ph2: y})
            epcoh_rating = epcoh_rating2 - negtrainmatrix
            recall_batch, precision_batch, map_batch, ndcg_batch = evaluate11(posprobe, epcoh_rating, [5, 10])
            print('load_epoch:', load_epoch, precision_batch[1], recall_batch[1], map_batch[1], ndcg_batch[1])
            precision.append(precision_batch)
            recall.append(recall_batch)
            map.append(map_batch)
            ndcg.append(ndcg_batch)
            evaluation = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(map), pd.DataFrame(ndcg)], axis=1)
            filename = "dumper2/disent_ex" + '_' + str(args.result) + str(args.result2) + '_' + str(args.lr) + '_' + str(
                args.int) + '_' + str(args.keep) + '_' + str(args.keep2) + '_' + str(args.temp) + '_' + str(args.bal) + '_' + str(args.bal2) + '_' + str(args.ssl_reg)
            evaluation.to_csv(filename + ".csv", header=False, index=False)
    else:
        update_count = 0.0
        for epoch in range(n_epochs+2):
            np.random.shuffle(idxlist)
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)

                X = train_data1[idxlist[st_idx:end_idx]]
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')
                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                else:
                    anneal = anneal_cap
                temp = max(1 * math.exp(-ANNEAL_RATE * update_count), args.temp)
                print('temp:', temp)

                feed_dict = {vae.input_ph: X, vae.keep_prob_ph: args.keep, vae.anneal_ph: anneal, vae.temp: temp, vae.is_training_ph: 1,
                             vae.pad_train: pad_train[idxlist[st_idx:end_idx]], vae.pad_mask: pad_mask[idxlist[st_idx:end_idx]]}
                print('training batch', bnum)
                _, l1, l2, l3, l4, l5 = sess.run([vae.train_op1, vae.neg_ll, vae.KL, vae.phi_theta_kl_loss, vae.theta, vae.phi], feed_dict=feed_dict)
                # _, l1, l2, l3, l4, aa, bb = sess.run([train_op_var, vae.neg_ll, vae.KL, vae.recon_loss, vae.kl_k], feed_dict=feed_dict)
                print(l1, l2, l3)
                print(fastOrder(l4), fastOrder(l5))
                update_count += 1
            if epoch > 40 and epoch % 10 == 0:
                saver.save(sess, "model/disent-{}-{}".format(args.result, str(epoch)))

            print('test begin')
            X = train_data1
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
            y = train_data2
            if sparse.isspmatrix(y):
                y = y.toarray()
            y = y.astype('float32')
            epcoh_rating, epcoh_rating2 = sess.run((logits_var, vae.rating), feed_dict={vae.input_ph: X, vae.input_ph2: y})
            epcoh_rating = epcoh_rating - negtrainmatrix
            recall_batch, precision_batch, map_batch, ndcg_batch = evaluate11(posprobe, epcoh_rating, [10, 20])
            print('epoch:', epoch, precision_batch[1], recall_batch[1], map_batch[1], ndcg_batch[1])
            precision.append(precision_batch)
            recall.append(recall_batch)
            map.append(map_batch)
            ndcg.append(ndcg_batch)
            evaluation = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(map), pd.DataFrame(ndcg)], axis=1)
            filename = "dumper2/disent" + '_' + str(args.result) + '_' + str(args.lr) + '_' + str(args.int) + '_' + str(args.keep) + '_' + str(args.keep2) + '_' + str(args.temp) + '_' + str(args.anneal)
            evaluation.to_csv(filename + ".csv", header=False, index=False)

















