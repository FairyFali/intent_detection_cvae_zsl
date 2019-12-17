#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import re
import sys
import time

import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
from tensorflow.python.ops import variable_scope
from sklearn.metrics import classification_report

import decoder_fn_lib
import utils
from models.seq2seq import dynamic_rnn_decoder
from utils import gaussian_kld
from utils import get_bi_rnn_encode
from utils import get_bow
from utils import get_rnn_encode
from utils import norm_log_liklihood
from utils import sample_gaussian


class BaseTFModel(object):
    global_t = tf.placeholder(dtype=tf.int32, name="global_t")
    learning_rate = None
    scope = None

    @staticmethod
    def print_model_stats(tvars):
        total_parameters = 0
        for variable in tvars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            print("Trainable %s with %d parameters" % (variable.name, variable_parametes))
            total_parameters += variable_parametes
        print("Total number of trainable parameters is %d" % total_parameters)

    @staticmethod
    def get_rnncell(cell_type, cell_size, keep_prob, num_layer):
        # thanks for this solution from @dimeldo
        cells = []
        for _ in range(num_layer):
            if cell_type == "gru":
                cell = rnn_cell.GRUCell(cell_size)
            else:
                cell = rnn_cell.LSTMCell(cell_size, use_peepholes=False, forget_bias=1.0)

            if keep_prob < 1.0:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            cells.append(cell)

        if num_layer > 1:
            cell = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        else:
            cell = cells[0]

        return cell

    @staticmethod
    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

    def train(self, global_t, sess, train_feed):
        raise NotImplementedError("Train function needs to be implemented")

    def valid(self, *args, **kwargs):
        raise NotImplementedError("Valid function needs to be implemented")

    def batch_2_feed(self, *args, **kwargs):
        raise NotImplementedError("Implement how to unpack the back")

    def optimize(self, sess, config, loss, log_dir):
        if log_dir is None:
            return
        # optimization
        if self.scope is None:
            tvars = tf.trainable_variables()
        else:
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        grads = tf.gradients(loss, tvars)
        if config.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, tf.constant(config.grad_clip)) # avoid gradient explosion
        # add gradient noise
        if config.grad_noise > 0:
            grad_std = tf.sqrt(config.grad_noise / tf.pow(1.0 + tf.to_float(self.global_t), 0.55))
            grads = [g + tf.truncated_normal(tf.shape(g), mean=0.0, stddev=grad_std) for g in grads]

        if config.op == "adam":
            print("Use Adam")
            optimizer = tf.train.AdamOptimizer(config.init_lr)
        elif config.op == "rmsprop":
            print("Use RMSProp")
            optimizer = tf.train.RMSPropOptimizer(config.init_lr)
        else:
            print("Use SGD")
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_ops = optimizer.apply_gradients(zip(grads, tvars))
        self.print_model_stats(tvars)
        train_log_dir = os.path.join(log_dir, "checkpoints")
        print("Save summary to %s" % log_dir)
        self.train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)


class RnnCVAE(BaseTFModel):

    def __init__(self, sess, config, api, log_dir, forward, scope=None): # forward???
        self.vocab = api.vocab
        self.rev_vocab = api.rev_vocab
        self.vocab_size = len(self.vocab)
        self.seen_intent = api.seen_intent
        self.rev_seen_intent = api.rev_seen_intent
        self.seen_intent_size = len(self.rev_seen_intent)
        self.unseen_intent = api.unseen_intent
        self.rev_unseen_intent = api.rev_unseen_intent
        self.unseen_intent_size = len(self.rev_unseen_intent)
        self.sess = sess
        self.scope = scope
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.sent_cell_size = config.sent_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.label_embed_size = config.label_embed_size
        self.latent_size = config.latent_size

        self.seed = config.seed
        self.use_ot_label = config.use_ot_label
        self.use_rand_ot_label = config.use_rand_ot_label  # Only valid if use_ot_label is true, whether use all other label
        self.use_rand_fixed_ot_label = config.use_rand_fixed_ot_label  # valid when use_ot_label=true and use_rand_ot_label=true
        if self.use_ot_label:
            self.rand_ot_label_num = config.rand_ot_label_num  # valid when use_ot_label=true and use_rand_ot_label=true
        else:
            self.rand_ot_label_num = self.seen_intent_size-1

        with tf.name_scope("io"):
            # all dialog context and known attributes
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="labels") # each utterance have a label, [batch_size,]
            self.ot_label_rand = tf.placeholder(dtype=tf.int32, shape=(None,None), name="ot_labels_rand")
            self.ot_labels_all = tf.placeholder(dtype=tf.int32, shape=(None, None), name="ot_labels_all") #(batch_size, len(api.label_vocab)-1)

            # target response given the dialog context
            self.io_tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_tokens")
            self.io_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_lens")
            self.output_labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_labels")

            # optimization related variables
            self.learning_rate = tf.Variable(float(config.init_lr), trainable=False, name="learning_rate")
            self.learning_rate_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, config.lr_decay))
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior") # whether use prior
            self.prior_mulogvar = tf.placeholder(dtype=tf.float32, shape=(None, config.latent_size*2), name="prior_mulogvar")

            self.batch_size = tf.placeholder(dtype=tf.int32, name="batch_size")

        max_out_len = array_ops.shape(self.io_tokens)[1]
        # batch_size = array_ops.shape(self.io_tokens)[0]
        batch_size = self.batch_size

        with variable_scope.variable_scope("labelEmbedding", reuse=tf.AUTO_REUSE):
            self.la_embedding = tf.get_variable("embedding", [self.seen_intent_size, config.label_embed_size], dtype=tf.float32)
            label_embedding = embedding_ops.embedding_lookup(self.la_embedding, self.output_labels) # not use

        with variable_scope.variable_scope("wordEmbedding", reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, config.embed_size], dtype=tf.float32, trainable=False)
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], dtype=tf.float32,
                                         shape=[self.vocab_size, 1])
            embedding = self.embedding * embedding_mask # boardcast, first row is all 0.

            io_embedding = embedding_ops.embedding_lookup(embedding, self.io_tokens) # 3 dim

            if config.sent_type == "bow":
                io_embedding, _ = get_bow(io_embedding)

            elif config.sent_type == "rnn":
                sent_cell = self.get_rnncell("gru", self.sent_cell_size, config.keep_prob, 1)
                io_embedding, _ = get_rnn_encode(io_embedding, sent_cell, self.io_lens,
                                                     scope="sent_rnn", reuse=tf.AUTO_REUSE)
            elif config.sent_type == "bi_rnn":
                fwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                bwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                io_embedding, _ = get_bi_rnn_encode(io_embedding, fwd_sent_cell, bwd_sent_cell, self.io_lens,
                                                    scope="sent_bi_rnn", reuse=tf.AUTO_REUSE) # equal to x of the graph, (batch_size, 300*2)
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

            # print('==========================', io_embedding) # Tensor("models_2/wordEmbedding/sent_bi_rnn/concat:0", shape=(?, 600), dtype=float32)

            # convert label into 1 hot
            my_label_one_hot = tf.one_hot(tf.reshape(self.labels, [-1]), depth=self.seen_intent_size, dtype=tf.float32) # 2 dim
            if config.use_ot_label:
                if config.use_rand_ot_label:
                    ot_label_one_hot = tf.one_hot(tf.reshape(self.ot_label_rand, [-1]), depth=self.seen_intent_size, dtype=tf.float32)
                    ot_label_one_hot = tf.reshape(ot_label_one_hot, [-1, self.seen_intent_size*self.rand_ot_label_num])
                else:
                    ot_label_one_hot = tf.one_hot(tf.reshape(self.ot_labels_all, [-1]), depth=self.seen_intent_size, dtype=tf.float32)
                    ot_label_one_hot = tf.reshape(ot_label_one_hot, [-1, self.seen_intent_size*(self.seen_intent_size - 1)]) # (batch_size, len(api.label_vocab)*(len(api.label_vocab)-1))

        with variable_scope.variable_scope("recognitionNetwork", reuse=tf.AUTO_REUSE):
            recog_input = io_embedding
            self.recog_mulogvar = recog_mulogvar = layers.fully_connected(recog_input, config.latent_size * 2, activation_fn=None, scope="muvar") # config.latent_size=200
            recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=1) # recognition network output. (batch_size, config.latent_size)

        with variable_scope.variable_scope("priorNetwork", reuse=tf.AUTO_REUSE):
            # p(xyz) = p(z)p(x|z)p(y|xz)
            # prior network parameter, assum the normal distribution
            # prior_mulogvar = tf.constant([[1] * config.latent_size + [0] * config.latent_size]*batch_size,
            #                              dtype=tf.float32, name="muvar") # can not use by this manner
            prior_mulogvar = self.prior_mulogvar
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)

            # use sampled Z or posterior Z
            latent_sample = tf.cond(self.use_prior, # bool input
                                    lambda: sample_gaussian(prior_mu, prior_logvar), # equal to shape(prior_logvar)
                                    lambda: sample_gaussian(recog_mu, recog_logvar)) # if ... else ..., (batch_size, config.latent_size)
            self.z = latent_sample

        with variable_scope.variable_scope("generationNetwork", reuse=tf.AUTO_REUSE):
            bow_loss_inputs = latent_sample # (part of) response network input
            label_inputs = latent_sample
            dec_inputs = latent_sample

            # BOW loss
            if config.use_bow_loss:
                bow_fc1 = layers.fully_connected(bow_loss_inputs, 400, activation_fn=tf.tanh, scope="bow_fc1") # MLPb network fc layer
                # error1:ValueError: The last dimension of the inputs to `Dense` should be defined. Found `None`.
                if config.keep_prob < 1.0:
                    bow_fc1 = tf.nn.dropout(bow_fc1, config.keep_prob)
                self.bow_logits = layers.fully_connected(bow_fc1, self.vocab_size, activation_fn=None, scope="bow_project") # MLPb network fc output

            # Y loss, include the other y.
            my_label_fc1 = layers.fully_connected(label_inputs, 400, activation_fn=tf.tanh, scope="my_label_fc1")
            if config.keep_prob < 1.0:
                my_label_fc1 = tf.nn.dropout(my_label_fc1, config.keep_prob)

            # my_label_fc2 = layers.fully_connected(my_label_fc1, 400, activation_fn=tf.tanh, scope="my_label_fc2")
            # if config.keep_prob < 1.0:
            #     my_label_fc2 = tf.nn.dropout(my_label_fc2, config.keep_prob)

            self.my_label_logits = layers.fully_connected(my_label_fc1, self.seen_intent_size,
                                                          scope="my_label_project")  # MLPy fc output
            my_label_prob = tf.nn.softmax(self.my_label_logits)  # softmax output, (batch_size, label_vocab_size)
            self.my_label_prob = my_label_prob
            pred_my_label_embedding = tf.matmul(my_label_prob, self.la_embedding)  # predicted my label y. (batch_size, label_embed_size)

            if config.use_ot_label:
                if config.use_rand_ot_label: # use one random other label
                    ot_label_fc1 = layers.fully_connected(label_inputs, 400, activation_fn=tf.tanh, scope="ot_label_fc1")
                    if config.keep_prob <1.0:
                        ot_label_fc1 = tf.nn.dropout(ot_label_fc1, config.keep_prob)
                    self.ot_label_logits = layers.fully_connected(ot_label_fc1, self.rand_ot_label_num*self.seen_intent_size, scope="ot_label_rand_project")
                    ot_label_logits_split = tf.reshape(self.ot_label_logits,[-1, self.rand_ot_label_num, self.seen_intent_size])
                    ot_label_prob_short = tf.nn.softmax(ot_label_logits_split)
                    ot_label_prob = tf.reshape(ot_label_prob_short, [-1, self.rand_ot_label_num*self.seen_intent_size]) # (batch_size, self.rand_ot_label_num*self.label_vocab_size)
                    pred_ot_label_embedding = tf.reshape(tf.matmul(ot_label_prob_short, self.la_embedding),
                                                         [self.label_embed_size * self.rand_ot_label_num])  # predicted other label y2.
                else:
                    ot_label_fc1 = layers.fully_connected(label_inputs, 400, activation_fn=tf.tanh, scope="ot_label_fc1")
                    if config.keep_prob <1.0:
                        ot_label_fc1 = tf.nn.dropout(ot_label_fc1, config.keep_prob)
                    self.ot_label_logits = layers.fully_connected(ot_label_fc1, self.seen_intent_size*(self.seen_intent_size-1), scope="ot_label_all_project")
                    ot_label_logits_split = tf.reshape(self.ot_label_logits, [-1, self.seen_intent_size-1, self.seen_intent_size])
                    ot_label_prob_short = tf.nn.softmax(ot_label_logits_split)
                    ot_label_prob = tf.reshape(ot_label_prob_short, [-1, self.seen_intent_size*(self.seen_intent_size-1)]) # (batch_size, self.label_vocab_size*(self.label_vocab_size-1))
                    pred_ot_label_embedding = tf.reshape(tf.matmul(ot_label_prob_short, self.la_embedding),
                                                         [self.label_embed_size*(self.seen_intent_size-1)]) # predicted other all label y. (batch_size, self.label_embed_size*(self.label_vocab_size-1))
                    # note:matmul can calc (3, 4, 5) × (5, 4) = (3, 4, 4)
            else: # only use label y.
                self.ot_label_logits = None
                pred_ot_label_embedding = None

            # Decoder, Response Network
            if config.num_layer > 1:
                dec_init_state = []
                for i in range(config.num_layer):
                    temp_init = layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state-%d" % i)
                    if config.cell_type == 'lstm':
                        temp_init = rnn_cell.LSTMStateTuple(temp_init, temp_init)

                    dec_init_state.append(temp_init)

                dec_init_state = tuple(dec_init_state)
            else:
                dec_init_state = layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state")
                if config.cell_type == 'lstm':
                    dec_init_state = rnn_cell.LSTMStateTuple(dec_init_state, dec_init_state)

        with variable_scope.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            dec_cell = self.get_rnncell(config.cell_type, self.dec_cell_size, config.keep_prob, config.num_layer)
            dec_cell = OutputProjectionWrapper(dec_cell, self.vocab_size)

            if forward: # test
                loop_func = decoder_fn_lib.context_decoder_fn_inference(None, dec_init_state, embedding,
                                                                        start_of_sequence_id=self.go_id,
                                                                        end_of_sequence_id=self.eos_id,
                                                                        maximum_length=self.max_utt_len,
                                                                        num_decoder_symbols=self.vocab_size,
                                                                        context_vector=None) # a function
                dec_input_embedding = None
                dec_seq_lens = None
            else: # train
                loop_func = decoder_fn_lib.context_decoder_fn_train(dec_init_state, None)
                dec_input_embedding = embedding_ops.embedding_lookup(embedding, self.io_tokens) # x 's embedding (batch_size, utt_len, embed_size)
                dec_input_embedding = dec_input_embedding[:, 0:-1, :] # ignore the last </s>
                dec_seq_lens = self.io_lens - 1 # input placeholder

                if config.keep_prob < 1.0:
                    dec_input_embedding = tf.nn.dropout(dec_input_embedding, config.keep_prob)

                # apply word dropping. Set dropped word to 0
                if config.dec_keep_prob < 1.0:
                    keep_mask = tf.less_equal(tf.random_uniform((batch_size, max_out_len-1), minval=0.0, maxval=1.0),
                                              config.dec_keep_prob)
                    keep_mask = tf.expand_dims(tf.to_float(keep_mask), 2)
                    dec_input_embedding = dec_input_embedding * keep_mask
                    dec_input_embedding = tf.reshape(dec_input_embedding, [-1, max_out_len-1, config.embed_size])

                # print("=======", dec_input_embedding) # Tensor("models/decoder/strided_slice:0", shape=(?, ?, 200), dtype=float32)

            dec_outs, _, final_context_state = dynamic_rnn_decoder(dec_cell, loop_func,
                                                                   inputs=dec_input_embedding,
                                                                   sequence_length=dec_seq_lens) # dec_outs [batch_size, seq, features]

            if final_context_state is not None:
                final_context_state = final_context_state[:, 0:array_ops.shape(dec_outs)[1]]
                mask = tf.to_int32(tf.sign(tf.reduce_max(dec_outs, axis=2))) # get softmax vec's max index
                self.dec_out_words = tf.multiply(tf.reverse(final_context_state, axis=[1]), mask)
            else:
                self.dec_out_words = tf.argmax(dec_outs, 2) # (batch_size, utt_len), each element is index of word


        if not forward:
            with variable_scope.variable_scope("loss", reuse=tf.AUTO_REUSE):
                labels = self.io_tokens[:, 1:] # not include the first word <s>, (batch_size, utt_len)
                label_mask = tf.to_float(tf.sign(labels))

                labels = tf.one_hot(labels, depth=self.vocab_size, dtype=tf.float32)

                print(dec_outs)
                print(labels)
                # Tensor("models_1/decoder/dynamic_rnn_decoder/transpose_1:0", shape=(?, ?, 892), dtype=float32)
                # Tensor("models_1/loss/strided_slice:0", shape=(?, ?), dtype=int32)
                # rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels) # response network loss
                rc_loss = tf.nn.softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels) # response network loss
                # logits_size=[390,892] labels_size=[1170,892]
                rc_loss = tf.reduce_sum(rc_loss * label_mask, reduction_indices=1) # (batch_size,), except the word unk
                self.avg_rc_loss = tf.reduce_mean(rc_loss) # scalar
                # used only for perpliexty calculation. Not used for optimzation
                self.rc_ppl = tf.exp(tf.reduce_sum(rc_loss) / tf.reduce_sum(label_mask))

                """ as n-trial multimodal distribution. """
                tile_bow_logits = tf.tile(tf.expand_dims(self.bow_logits, 1), [1, max_out_len - 1, 1]) # (batch_size, max_out_len-1, vocab_size)
                bow_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tile_bow_logits, labels=labels) * label_mask # labels shape less than logits shape, (batch_size, max_out_len-1)
                bow_loss = tf.reduce_sum(bow_loss, reduction_indices=1) # (batch_size, )
                self.avg_bow_loss  = tf.reduce_mean(bow_loss) # scalar

                # the label y
                my_label_loss = tf.nn.softmax_cross_entropy_with_logits(logits=my_label_prob, labels=my_label_one_hot) # label (batch_size,)
                self.avg_my_label_loss = tf.reduce_mean(my_label_loss)
                if config.use_ot_label:
                    ot_label_loss = -tf.nn.softmax_cross_entropy_with_logits(logits=ot_label_prob, labels=ot_label_one_hot)
                    self.avg_ot_label_loss = tf.reduce_mean(ot_label_loss)
                else:
                    self.avg_ot_label_loss = 0.0

                kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar) # kl divergence, (batch_size,)
                self.avg_kld = tf.reduce_mean(kld) # scalar
                if log_dir is not None:
                    kl_weights = tf.minimum(tf.to_float(self.global_t)/config.full_kl_step, 1.0)
                else:
                    kl_weights = tf.constant(1.0)

                self.kl_w = kl_weights
                self.elbo = self.avg_rc_loss + kl_weights * self.avg_kld # Restructure loss and kl divergence
                #=====================================================================================================total loss====================================================#
                if config.use_rand_ot_label:
                    aug_elbo = self.avg_bow_loss + 1000*self.avg_my_label_loss + 10*self.avg_ot_label_loss + self.elbo # augmented loss
                    # (1/self.rand_ot_label_num)*
                else:
                    aug_elbo = self.avg_bow_loss + 1000*self.avg_my_label_loss + 10*self.avg_ot_label_loss + self.elbo  # augmented loss
                    # (1/(self.label_vocab_size-1))*

                tf.summary.scalar("rc_loss", self.avg_rc_loss)
                tf.summary.scalar("elbo", self.elbo)
                tf.summary.scalar("kld", self.avg_kld)
                tf.summary.scalar("bow_loss", self.avg_bow_loss)
                tf.summary.scalar("my_label_loss", self.avg_my_label_loss)
                tf.summary.scalar("ot_label_loss", self.avg_ot_label_loss)

                self.summary_op = tf.summary.merge_all()

                self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar) # probability
                self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar) # probability
                self.est_marginal = tf.reduce_mean(rc_loss + bow_loss - self.log_p_z + self.log_q_z_xy)

            self.optimize(sess, config, aug_elbo, log_dir)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        print('model establish finish!')

    def batch_2_feed(self, batch, global_t, use_prior, repeat=1, most_similarity=None):
        utts, utts_lens, intents = batch
        ## print(batch)
        # ot_label_rand
        batch_size = len(utts_lens)
        # np.random.seed(self.seed)
        ot_label_rand = []
        if self.use_rand_fixed_ot_label:
            # fixed
            if most_similarity == None: raise Exception("most_simiarity is none error!")
            rand_num = self.rand_ot_label_num
            for i,x in enumerate(intents):
                t = most_similarity[x][:rand_num]
                ot_label_rand.append(t)
        else:
            rand_num = self.rand_ot_label_num
            ot_label_rand_cand = np.random.randint(self.seen_intent_size-1, size=[batch_size, rand_num])
            for i,x in enumerate(ot_label_rand_cand):
                t = list(map(lambda a: a if a<intents[i] else a+1, x))
                ot_label_rand.append(t)
        # ot_label_all
        ot_label_all_cand = [list(range(self.seen_intent_size-1))] * batch_size
        ot_label_all = []
        for i,x in enumerate(ot_label_all_cand):
            t = []
            for j,y in enumerate(x):
                if y < intents[i]:
                    t.append(y)
                else:
                    t.append(y+1)
            ot_label_all.append(t)

        # print('=====+', intents)
        # print('=====-', ot_label_rand)

        # prior_mulogvar
        prior_mulogvar = [[0]*self.latent_size + [1]*self.latent_size]*batch_size

        feed_dict = {self.labels: intents,
                     self.ot_labels_all: np.array(ot_label_all),
                     self.ot_label_rand: np.array(ot_label_rand),
                     self.io_tokens: utts,
                     self.io_lens: np.array([self.max_utt_len] * batch_size), # modify
                     self.use_prior: use_prior,
                     self.batch_size: batch_size,
                     self.prior_mulogvar:np.array(prior_mulogvar)}
        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key is self.use_prior:
                    tiled_feed_dict[key] = val
                    continue
                if key is self.batch_size:
                    tiled_feed_dict[key] = val
                    continue
                multipliers = [1]*len(val.shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if global_t is not None:
            feed_dict[self.global_t] = global_t

        return feed_dict

    def train(self, global_t, sess, train_feed, update_limit=5000):
        most_similarity = train_feed.most_similarity

        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        kl_losses = []
        bow_losses = []
        local_t = 0
        start_time = time.time()

        latent_z = []
        outpout_labels = []

        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break
            feed_dict = self.batch_2_feed(batch, global_t, use_prior=False, most_similarity=most_similarity)
            if self.use_ot_label:
                _, sum_op, elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss, my_label_loss, ot_label_loss, z = sess.run([self.train_ops,
                                                                                                             self.summary_op,
                                                                                                             self.elbo,
                                                                                                             self.avg_bow_loss,
                                                                                                             self.avg_rc_loss,
                                                                                                             self.rc_ppl,
                                                                                                             self.avg_kld,
                                                                                                             self.avg_my_label_loss,
                                                                                                             self.avg_ot_label_loss,

                                                                                                             self.z],
                                                                                                     feed_dict)
            else:
                _, sum_op, elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss, my_label_loss, z = sess.run([self.train_ops,
                                                                                                             self.summary_op,
                                                                                                             self.elbo,
                                                                                                             self.avg_bow_loss,
                                                                                                             self.avg_rc_loss,
                                                                                                             self.rc_ppl,
                                                                                                             self.avg_kld,
                                                                                                             self.avg_my_label_loss,

                                                                                                             self.z],
                                                                                                     feed_dict)
                ot_label_loss = None

            self.train_summary_writer.add_summary(sum_op, global_t)
            elbo_losses.append(elbo_loss)
            bow_losses.append(bow_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)

            latent_z.extend(z)
            outpout_labels.extend(feed_dict[self.labels])

            if self.use_ot_label:
                loss_names = ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss", "my_label_loss",
                              "ot_label_loss"]
            else:
                loss_names = ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss", "my_label_loss"]

            global_t += 1
            local_t += 1
            if local_t % int(train_feed.num_batch / 10) == 0: # Print 10 times in total
                kl_w = sess.run(self.kl_w, {self.global_t: global_t})
                if self.use_ot_label:
                    self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
                                loss_names, [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses, my_label_loss, ot_label_loss], "kl_w %f" % kl_w)
                else:
                    self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
                                    loss_names, [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses, my_label_loss], "kl_w %f" % kl_w)

        # finish one epoch!
        epoch_time = time.time() - start_time
        if self.use_ot_label:
            avg_losses = self.print_loss("Epoch Done "+str(epoch_time), loss_names,
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses, my_label_loss, ot_label_loss],
                                     "step time %.4f, " % (epoch_time / train_feed.num_batch))
        else:
            avg_losses = self.print_loss("Epoch Done " + str(epoch_time), loss_names,
                                         [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses, my_label_loss],
                                         "step time %.4f, " % (epoch_time / train_feed.num_batch))


        return global_t, avg_losses[0], latent_z, outpout_labels

    def valid(self, name, sess, valid_feed):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        bow_losses = []
        kl_losses = []
        my_label_losses = []

        total_batch_num = 10

        for i in range(total_batch_num):
            batch = valid_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1, most_similarity=valid_feed.most_similarity)
            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss, my_label_loss  = sess.run(
                [self.elbo, self.avg_bow_loss, self.avg_rc_loss,
                 self.rc_ppl, self.avg_kld, self.avg_my_label_loss], feed_dict)

            elbo_losses.append(elbo_loss)
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)
            bow_losses.append(bow_loss)
            kl_losses.append(kl_loss)
            my_label_losses.append(my_label_loss)

        avg_losses = self.print_loss(name, ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss", "my_label_loss"],
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses, my_label_losses], "")
        return avg_losses[0]

    def test(self, sess, test_feed, num_batch=None, repeat=1, dest=sys.stdout):
        local_t = 0
        recall_bleus = []
        prec_bleus = []

        latent_z = []
        output_labels = []

        sim = test_feed.sim
        print('################################\n', sim)

        total = 0
        precision_count = 0
        clf = {v:[] for k,v in self.rev_unseen_intent.items()}

        report_pred_label = []
        report_true_label = []
        while True:
            batch = test_feed.next_batch()

            if batch is None or (num_batch is not None and local_t > num_batch):
                break
            total += len(batch[1])
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=repeat, most_similarity=test_feed.most_similarity)
            word_outs, label_prob, z = sess.run([self.dec_out_words, self.my_label_prob, self.z], feed_dict)
            sample_words = np.split(word_outs, repeat, axis=0)
            sample_label = np.split(label_prob, repeat, axis=0)

            latent_z.extend(z)
            output_labels.extend(feed_dict[self.labels])

            true_outs = feed_dict[self.io_tokens]
            true_labels = feed_dict[self.labels]
            utts_lens = feed_dict[self.io_lens]
            local_t += 1

            if dest != sys.stdout:
                if local_t % (test_feed.num_batch / 10) == 0:
                    print("%.2f >> " % (test_feed.ptr / float(test_feed.num_batch)))
                    dest.write("%.2f >> " % (test_feed.ptr / float(test_feed.num_batch)))

            report_true_label.extend(true_labels[::repeat])

            for b_id in range(test_feed.batch_size):
                dest.write("Batch %d index %d \n" % (local_t, b_id))
                start = np.maximum(0, utts_lens[b_id]-5)
                # print the true outputs
                true_tokens = [self.vocab[e] for e in true_outs[b_id].tolist() if e not in [0, self.eos_id, self.go_id]]
                true_str = " ".join(true_tokens).replace(" ' ", "'")
                label_str = self.unseen_intent[true_labels[b_id]]
                # print the predicted outputs
                dest.write("Target (%s) >> %s\n" % (label_str, true_str))
                local_tokens = []
                flag = False
                for r_id in range(repeat):
                    pred_outs = sample_words[r_id]
                    # pred_label = np.argmax(sample_label[r_id], axis=1)[b_id]
                    vec = sample_label[r_id][b_id] # (seen_intent_size,)
                    vec2 = np.matmul(vec, sim)
                    pred_label = np.argmax(vec2) #====================================#
                    if pred_label == true_labels[b_id]:
                        flag=True
                    clf[true_labels[b_id]].append(vec)
                    pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist() if e != self.eos_id and e != 0]
                    pred_str = " ".join(pred_tokens).replace(" ' ", "'")
                    dest.write("Sample %d (%s) >> %s\n" % (r_id, self.unseen_intent[pred_label], pred_str))
                    local_tokens.append(pred_tokens)
                if flag:
                    precision_count += 1
                    report_pred_label.append(true_labels[b_id])
                else:
                    report_pred_label.append(pred_label)

                max_bleu, avg_bleu = utils.get_bleu_stats(true_tokens, local_tokens)
                recall_bleus.append(max_bleu)
                prec_bleus.append(avg_bleu)
                # make a new line for better readability
                dest.write("\n")
        # print(report_true_label, report_pred_label)
        # print(len(report_true_label), len(report_pred_label))

        # The most easily misclassified
        count = {k:np.mean(v, axis=0).tolist() for k,v in clf.items()}
        print(count)
        dest.write(str(count) + '\n')
        a = np.array(count[0])
        b = np.array(count[1])
        c = np.array([a/(b+a), b/(a+b)]).transpose()
        c[np.isnan(c)] = 0
        c[np.isinf(c)] = 0
        test_feed.sim = c

        avg_recall_bleu = float(np.mean(recall_bleus))
        avg_prec_bleu = float(np.mean(prec_bleus))
        avg_f1 = 2*(avg_prec_bleu*avg_recall_bleu) / (avg_prec_bleu+avg_recall_bleu+10e-12)
        report = "Avg recall BLEU %f, avg precision BLEU %f and F1 %f (only 1 reference response. Not final result)" \
                 % (avg_recall_bleu, avg_prec_bleu, avg_f1)
        print(report)
        dest.write(report + "\n")
        dest.write("total sample " + str(total) + ", correct sample "+ str(precision_count) + " precision rate is " + str(precision_count/total) + "\n")
        result = classification_report(report_true_label, report_pred_label, digits=6)
        dest.write(result + '\n')
        print("Done testing")

        return latent_z, output_labels


