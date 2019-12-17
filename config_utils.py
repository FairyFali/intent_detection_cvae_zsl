# -*- coding: utf-8 -*-
# @Time   : 2019/11/7 11:19:50
# @Author : Wang Fali

class KgCVAEConfig(object):
    description= None
    use_ot_label = True # whether use other label, use other label in training(if turn off kgCVAE -> CVAE)
    use_rand_ot_label = True # Only valid if use_ot_label is true, whether use all other label
    rand_ot_label_num = 1 # can not exceed the label_vocab_size, valid when use_ot_label=true and use_rand_ot_label=true
    use_rand_fixed_ot_label = True # valid when use_ot_label=true and use_rand_ot_label=true

    use_bow_loss = True
    update_limit = 3000  # the number of mini-batch before evaluating the models

    # how to encode utterance.
    # bow: add word embedding together
    # rnn: RNN utterance encoder
    # bi_rnn: bi_directional RNN utterance encoder
    sent_type = "bi_rnn"

    # latent variable (gaussian variable)
    latent_size = 300  # the dimension of latent variable, 200
    full_kl_step = 10000  # how many batch before KL cost weight reaches 1.0, 10000
    dec_keep_prob = 1.0  # do we use word drop decoder [Bowman el al 2015], 1.0

    # Network general
    cell_type = "gru"  # gru or lstm
    embed_size = 300  # word embedding size
    label_embed_size = 10  # label embedding size
    sent_cell_size = 300  # utterance encoder hidden size, 300
    dec_cell_size = 400  # response decoder hidden size, 400
    max_utt_len = 40  # max number of words in an utterance
    num_layer = 1  # number of context RNN layers

    # Optimization parameters
    op = "adam"
    grad_clip = 5.0  # gradient abs max cut
    init_w = 0.08  # uniform random from [-init_w, init_w]
    batch_size = 30  # mini-batch size
    init_lr = 0.001  # initial learning rate, 0.001
    lr_hold = 1  # only used by SGD
    lr_decay = 0.6  # only used by SGD
    keep_prob = 1.0  # drop out rate
    improve_threshold = 0.996  # for early stopping
    patient_increase = 2.0  # for early stopping
    early_stop = True
    max_epoch = 30  # max number of epoch of training
    grad_noise = 0.0  # inject gradient noise?

    seed = 2019


