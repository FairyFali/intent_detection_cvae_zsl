# -*- coding: utf-8 -*-
# @Time   : 2019/11/19 15:53:10
# @Author : Wang Fali
import os
import time

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils import KgCVAEConfig as Config
from data_api.corpus import Corpus
from models.cvae import RnnCVAE
from collections import Counter

# constants
tf.app.flags.DEFINE_string("word2vec_path", "D:/workspace/数据/glove.6B/glove.6B.200d.txt", "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", "data", "Raw data directory.")
tf.app.flags.DEFINE_string("dataset", "ATIS", "dataset.")
tf.app.flags.DEFINE_string("work_dir", "working", "Experiment results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "", "the dir to load checkpoint for forward only")
FLAGS = tf.app.flags.FLAGS

config = Config()

# get data set
api = Corpus(FLAGS.data_dir, FLAGS.dataset, word2vec=FLAGS.word2vec_path, word2vec_dim=config.embed_size)

# convert the word to ids
corpus= api.get_corpus()
train_corpus, valid_corpus, test_corpus = corpus['train'], corpus['valid'], corpus['test']

print('train_label', set(api.train_corpus['label'].tolist()))
print(Counter(api.train_corpus['label'].tolist()).most_common())
print('valid_label', set(api.valid_corpus['label'].tolist()))
print('test_label', set(api.test_corpus['label'].tolist()))
print(Counter(api.test_corpus['label'].tolist()).most_common())


if FLAGS.dataset == 'ATIS':
    from data_api.data_utils import ATISDataLoader

    train_feed = ATISDataLoader("Train", train_corpus, config)
    valid_feed = ATISDataLoader("Valid", valid_corpus, config)
    test_feed = ATISDataLoader("Test", test_corpus, config)


test_feed.epoch_init(1, shuffle=False)
print(test_feed.next_batch())
print(test_feed.num_batch)

