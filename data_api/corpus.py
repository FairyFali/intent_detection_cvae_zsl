# -*- coding: utf-8 -*-
# @Time   : 2019/11/7 14:08:01
# @Author : Wang Fali

import pandas as pd
import nltk
from collections import Counter
import numpy as np
import os

class Corpus(object):

    def __init__(self, corpus_path, dataset, max_vocab_cnt=10000, word2vec=None, word2vec_dim=None):
        '''

        :param corpus_path: data folder
        :param max_vocab_cnt: vocabulary size
        :param word2vec: glove
        :param word2vec_dim: 300
        '''
        self._path = corpus_path
        self.word2vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.sil_utt = ["<s>", "<sil>", "</s>"]
        self.dataset = dataset
        data, partition_to_n_row = self.load_data()

        if dataset == 'SNIPS':
            seen_intent = ['music', 'search', 'movie', 'weather', 'restaurant']
            unseen_intent = ['playlist', 'book']
            intent_map = {'PlayMusic':'music', 'GetWeather':'weather', 'BookRestaurant':'restaurant', 'SearchScreeningEvent':'search', 'RateBook':'book', 'SearchCreativeWork':'movie', 'AddToPlaylist':'playlist'}
            # df[df['A'].isin([2,3])]
            # data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
            data['intent'] = data['label'].map(intent_map)

            self.train_corpus = self.process(data[data['intent'].isin(seen_intent)])
            self.test_corpus = self.process(data[data['intent'].isin(unseen_intent)])
            self.seen_intent = seen_intent
            self.unseen_intent = unseen_intent

            # similarity between seen_intent and unseen_intent, sim[i][j] means the sim of seen_intent[i] and unseen_intent[j]
       #      self.sim = np.array([[0.59090424, 0.40909573],
               # [0.54641646, 0.4535835 ],
               # [0.4894069 , 0.5105932 ],
               # [0.53488654, 0.46511352],
               # [0.21101166, 0.7889883 ]]) # [5,2]
            # self.sim = [[0.54583406, 0.45416597],
            #            [0.52325845, 0.47674152],
            #            [0.49470285, 0.5052971 ],
            #            [0.5174646 , 0.48253548],
            #            [0.3408701 , 0.6591299 ]]
            # self.sim = [[0.7923042, 0.20769577],
            #  [0.44046, 0.55954003],
            #  [0.32713783, 0.6728622],
            #  [0.72433466, 0.27566534],
            #  [0.38274676, 0.6172532]]
            # self.sim = [[0.8816665 , 0.11833352],
            #            [0.4112154 , 0.58878464],
            #            [0.25317717, 0.74682283],
            #            [0.8098597 , 0.19014026],
            #            [0.32808515, 0.6719148 ]]
            self.sim = [[0.49545258, 0.50454742],
                       [0.49377695, 0.50622305],
                       [0.46389095, 0.53610905],
                       [0.53910239, 0.46089761],
                       [0.51074833, 0.48925167]]


        self.build_vocab(max_vocab_cnt)
        self.load_word2vec()
        print("Done loading corpus")



    def load_data(self):
        texts = []
        labels = []
        partition_to_n_row = {}
        for partition in ['train', 'valid', 'test']:
            seq_in = self._path + "/" + self.dataset + "/" + partition + ".seq.in"
            label_path = self._path + "/" + self.dataset + "/" + partition + ".label"
            if not os.path.exists(seq_in) or not os.path.exists(label_path):
                continue

            with open(seq_in, encoding='utf-8') as fp:
                lines = fp.read().splitlines()
                texts.extend(lines)
                partition_to_n_row[partition] = len(lines)
            with open(label_path, encoding='utf-8') as fp:
                labels.extend(fp.read().splitlines())

        df = pd.DataFrame([texts, labels]).T
        df.columns = ['text', 'label']
        return df, partition_to_n_row


    def process(self, df):
        df['content_words'] = df['text'].apply(lambda s: ['<s>']+nltk.WordPunctTokenizer().tokenize(s.lower())+['</s>'])
        df['all_lenes'] = df['content_words'].apply(lambda x: len(x))
        max_len = df['all_lenes'].max()
        min_len = df['all_lenes'].min()
        mean_len = df['all_lenes'].mean()
        print("Max utt len %d, Min utt len %d, mean utt len %.2f" % (max_len, min_len, mean_len))

        return df


    def build_vocab(self, max_vocab_cnt):
        all_words = []
        for tokens in self.train_corpus['content_words']:
            all_words.extend(tokens)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))


        self.vocab = ["<pad>", "<unk>"] + [t for t, _ in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab["<unk>"]
        print("<sil> index %d" % self.rev_vocab.get("<sil>", -1))
        print("<unk> index %d" % self.rev_vocab.get("<unk>", -1))

        self.rev_seen_intent = {t: idx for idx, t in enumerate(self.seen_intent)}
        self.rev_unseen_intent = {t: idx for idx, t in enumerate(self.unseen_intent)}
        print(self.seen_intent)
        print(self.rev_seen_intent)
        print("%d labels in train data" % len(self.seen_intent))
        print(self.unseen_intent)
        print(self.rev_unseen_intent)
        print("%d labels in test data" % len(self.unseen_intent))


    def load_word2vec(self):
        '''
        load the word2vec in accodressing to the vocab
        :return:
        '''
        if self.word2vec_path is None:
            return
        raw_word2vec = {}
        with open(self.word2vec_path, encoding='UTF-8') as f:
            for l in f:
                w, vec = l.split(" ", 1)
                raw_word2vec[w] = vec
        self.word2vec = []
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            self.word2vec.append(vec)
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))


    def get_corpus(self):
        '''
        utt convert the token to index
        :return:
        '''

        self.train_corpus['content_words'] = self.train_corpus['content_words'].apply(lambda x: [self.rev_vocab.get(w, self.unk_id) for w in x])
        self.train_corpus['intent'] = self.train_corpus['intent'].apply(lambda x: self.rev_seen_intent[x])

        self.test_corpus['content_words'] = self.test_corpus['content_words'].apply(lambda x: [self.rev_vocab.get(w, self.unk_id) for w in x])
        self.test_corpus['intent'] = self.test_corpus['intent'].apply(lambda x: self.rev_unseen_intent[x])

        return {'train':self.train_corpus, 'test':self.test_corpus}


