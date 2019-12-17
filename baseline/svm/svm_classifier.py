# -*- coding: utf-8 -*-#
# Name:         svm_classifier
# Description:  
# Author:       fali wang
# Date:         2019/11/28 16:55

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import pandas as pd
import logging
from nlp.utils.clean_text import clean_en_text
from nlp.utils.basic_log import Log

import nltk
from collections import Counter
import numpy as np


class SVMClassifier(object):

    def __init__(self, dataset, corpus_path, model_file):
        '''

        :param corpus_path: data folder
        :param model_file
        '''
        self.model_file = model_file
        self._path = corpus_path  # D:\workspace\数据
        self.dataset = dataset
        data, partition_to_n_row = self.load_data()
        chi_features, tf_idf_model, chi_model = self.select_features(data)
        self.tf_idf_model = tf_idf_model
        self.chi_model = chi_model
        y = data['label'].values

        self.train_x = chi_features[0:partition_to_n_row['train']]
        self.train_y = y[0:partition_to_n_row['train']]
        self.valid_x = chi_features[partition_to_n_row['train']:partition_to_n_row['train'] + partition_to_n_row['valid']]
        self.valid_y = y[partition_to_n_row['train']:partition_to_n_row['train'] + partition_to_n_row['valid']]
        self.test_x = chi_features[partition_to_n_row['train'] + partition_to_n_row['valid']:None]
        self.test_y = y[partition_to_n_row['train'] + partition_to_n_row['valid']:None]

        print(type(self.train_x))
        print(self.train_x)
        print("Done loading corpus")

    def load_data(self):
        texts = []
        labels = []
        partition_to_n_row = {}
        for partition in ['train', 'valid', 'test']:
            with open(self._path + "/" + self.dataset + "/" + partition + ".seq.in") as fp:
                lines = fp.read().splitlines()
                texts.extend(lines)
                partition_to_n_row[partition] = len(lines)
            with open(self._path + "/" + self.dataset + "/" + partition + ".label") as fp:
                labels.extend(fp.read().splitlines())

        df = pd.DataFrame([texts, labels]).T
        df.columns = ['text', 'label']
        return df, partition_to_n_row

    def select_features(self, data_set):
        dataset = [clean_en_text(data) for data in data_set['text']]
        tf_idf_model = TfidfVectorizer(ngram_range=(1, 1),
                                       binary=True,
                                       sublinear_tf=True)
        tf_vectors = tf_idf_model.fit_transform(dataset)

        # 选出前1/6的词用来做特征
        k = int(tf_vectors.shape[1] / 5)
        chi_model = SelectKBest(chi2, k=10)
        chi_features = chi_model.fit_transform(tf_vectors, data_set['label'])
        print('tf-idf:\t\t' + str(tf_vectors.shape[1]))
        print('chi:\t\t' + str(chi_features.shape[1]))

        return chi_features, tf_idf_model, chi_model

    def train(self):
        # 这里采用的是线性分类模型,如果采用rbf径向基模型,速度会非常慢.
        self.clf_model = svm.SVC(kernel='linear', verbose=True) # rbf, linear
        # print(self.clf_model)
        self.clf_model.fit(self.train_x, self.train_y)
        valid_score = self.clf_model.score(self.valid_x, self.valid_y)
        test_score = self.clf_model.score(self.test_x, self.test_y)
        print('验证集测试准确率:', valid_score)
        print('测试集测试准确率:', test_score)
        print('total 准确率:', (valid_score * self.valid_x.shape[0] + test_score * self.test_x.shape[0])/ ( self.valid_x.shape[0] + self.test_x.shape[0]))
        self.save_model()

    def save_model(self):
        with open(self.model_file, "wb") as file:
            pickle.dump((self.tf_idf_model, self.chi_model, self.clf_model), file)
        if file:
            file.close()

if __name__ == '__main__':
    svm_model = SVMClassifier('ATIS', r'D:\workspace\数据', './model.pkl')
    svm_model.train()