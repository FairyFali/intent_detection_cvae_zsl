# -*- coding: utf-8 -*-
# @Time   : 2019/11/7 10:08:59
# @Author : Wang Fali

import os
import time

import numpy as np
import tensorflow as tf
from beeprint import pp
import pickle

from config_utils import KgCVAEConfig as Config
from data_api.corpus import Corpus
from models.cvae import RnnCVAE

# constants
tf.app.flags.DEFINE_string("word2vec_path", "D:/workspace/数据/glove.6B/glove.6B.300d.txt", "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", "data", "Raw data directory.")
tf.app.flags.DEFINE_string("dataset", "SNIPS", "dataset.")
tf.app.flags.DEFINE_string("work_dir", "working", "Experiment results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "", "the dir to load checkpoint for forward only")
FLAGS = tf.app.flags.FLAGS

def main():
    # config for train
    config = Config()

    # config for validation
    valid_config = Config()
    valid_config.keep_prob = 1.0
    valid_config.dec_keep_prob = 1.0
    valid_config.batch_size = 60

    # config for test
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.dec_keep_prob = 1.0
    test_config.batch_size = 1

    # pp(config)

    # get data set
    api = Corpus(FLAGS.data_dir, FLAGS.dataset, word2vec=FLAGS.word2vec_path, word2vec_dim=config.embed_size)

    # convert the word to ids
    corpus = api.get_corpus()
    sim = api.sim
    train_corpus, test_corpus = corpus['train'], corpus['test']

    # test the vocabulary and the embedding
    # print(api.vocab[0], api.word2vec[0])

    if FLAGS.dataset == 'ATIS':
        from data_api.data_utils import ATISDataLoader
        train_feed = ATISDataLoader("Train", train_corpus, config)
        valid_feed = train_feed
        test_feed = ATISDataLoader("Test", test_corpus, config)
    elif FLAGS.dataset == 'SNIPS':
        from data_api.data_utils import SNIPSDataLoader
        train_feed = SNIPSDataLoader("Train", train_corpus, config, sim)
        valid_feed = train_feed
        test_feed = SNIPSDataLoader("Test", test_corpus, config, sim)

    # train_feed.epoch_init(config.batch_size)
    # print(train_feed.next_batch()) # (utts, utts_lens, intents), ((batch_size, config.max_utt_len), (batch_size,), (batch_size,))

    if FLAGS.forward_only or FLAGS.resume:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "run" + str(int(time.time()))) # new a log directory

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as sess:
        initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
        scope = "models"
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            model = RnnCVAE(sess, config, api, log_dir=None if FLAGS.forward_only else log_dir, forward=False,
                              scope=scope)
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            valid_model = RnnCVAE(sess, valid_config, api, log_dir=None, forward=False, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            test_model = RnnCVAE(sess, test_config, api, log_dir=None, forward=True, scope=scope)

        print("Created conputation graphs.")
        if api.word2vec is not None and not FLAGS.forward_only:
            print("Loaded word2vec")
            sess.run(model.embedding.assign(np.array(api.word2vec)))

        # print(sess.run(tf.add([1,2], [2,3])))

        # write config to a file for logging
        if not FLAGS.forward_only:
            with open(os.path.join(log_dir, "run.log"), "w") as f: # wb ==> w
                f.write(pp(api.sim, output=False) + '\n')
                f.write(pp(config.__dict__, output=False) + '\n')

        # create a folder by force
        chk_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(chk_dir):
            os.mkdir(chk_dir)

        ckpt = tf.train.get_checkpoint_state(chk_dir)
        print("Create models with fresh parameters.")

        sess.run(tf.global_variables_initializer())

        if ckpt:  # if have model parameters previous then load model parameters
            print("Reading cvae models parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)

        repeat = 1
        if not FLAGS.forward_only: # train
            checkpoint_path = os.path.join(chk_dir, model.__class__.__name__+".ckpt")
            global_t = 1
            patience = 10
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            print("begin train.")
            start_time = time.time()
            for epoch in range(config.max_epoch):
                print("train epoch", epoch)
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

                # begin training
                if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                    train_feed.epoch_init(config.batch_size, shuffle=True)
                global_t, train_loss, train_latent_z, train_outpout_labels = model.train(global_t, sess, train_feed, update_limit=config.update_limit)

                # begin validation
                valid_feed.epoch_init(valid_config.batch_size, shuffle=False)
                valid_loss = valid_model.valid("ELBO_VALID", sess, valid_feed)

                test_feed.epoch_init(test_config.batch_size, shuffle=True)
                test_model.test(sess, test_feed, num_batch=5)

                done_epoch = epoch + 1
                # only save a models if the dev loss is smaller
                # Decrease learning rate if no improvement was seen over last 3 times.
                if config.op == "sgd" and done_epoch > config.lr_hold:
                    sess.run(model.learning_rate_decay_op)
                if valid_loss < best_dev_loss:
                    if valid_loss <= dev_loss_threshold * config.improve_threshold:
                        patience = max(patience, done_epoch * config.patient_increase)
                        dev_loss_threshold = valid_loss
                    # still save the best train model
                    if FLAGS.save_model:
                        print("Save model!!")
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                    best_dev_loss = valid_loss
                if config.early_stop and patience <= done_epoch:
                    print("!!Early stop due to run out of patience!!")
                    break
                if epoch % 2 == 0:
                    print("test and valid", epoch, "epoch.")
                    test_output_path = "test_epoch{}.txt".format(epoch)
                    # valid_output_path = "valid_epoch{}.txt".format(epoch)

                    dest_f = open(os.path.join(log_dir, test_output_path), "w", encoding='utf-8')
                    test_feed.epoch_init(test_config.batch_size, shuffle=False)
                    latent_z, output_labels = test_model.test(sess, test_feed, num_batch=None, repeat=repeat, dest=dest_f)
                    dest_f.close()

                    # save the test latent z and true label
                    # with open(os.path.join(log_dir, "latent_z_{}.pkl".format(epoch)), "wb") as f:
                    #     pickle.dump(latent_z, f)
                    # with open(os.path.join(log_dir, "output_labels_{}.pkl".format(epoch)), "wb") as f:
                    #     pickle.dump(output_labels, f)
                    # save the train latent z
                    # np.save(os.path.join(log_dir, "train_latent_z_{}.npy".format(epoch)), np.array(train_latent_z))
                    # np.save(os.path.join(log_dir, "train_outpout_labels_{}.npy".format(epoch)), np.array(train_outpout_labels))

                    # dest_f = open(os.path.join(log_dir, valid_output_path), "w", encoding='utf-8')
                    # valid_feed.epoch_init(test_config.batch_size, shuffle=False)
                    # test_model.test(sess, valid_feed, num_batch=None, repeat=repeat, dest=dest_f)
                    dest_f.close()

            test_output_path = "test_epoch{}.txt".format(epoch+1)
            dest_f = open(os.path.join(log_dir, test_output_path), "w", encoding='utf-8')
            test_feed.epoch_init(test_config.batch_size, shuffle=False)
            test_model.test(sess, test_feed, num_batch=None, repeat=repeat, dest=dest_f)
            dest_f.close()

            print("Best validation loss %f" % best_dev_loss)
            print("Done training")
            print("train total time", time.time()-start_time)
        else:  # valid and test
            # begin validation

            dest_f = open(os.path.join(log_dir, "test.txt"), "w")
            test_feed.epoch_init(test_config.batch_size, shuffle=False)
            latent_z, output_labels = test_model.test(sess, test_feed, num_batch=None, repeat=repeat, dest=dest_f)
            dest_f.close()

            np.save(os.path.join(log_dir, "latent_z_{}.npy".format(config.max_epoch)), np.array(latent_z))
            np.save(os.path.join(log_dir, "output_labels_{}.npy".format(config.max_epoch)), np.array(output_labels))


if __name__ == "__main__":
    if FLAGS.forward_only:
        if FLAGS.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main()



