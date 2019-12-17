import pickle

from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel
import jieba
import os


class LdaTopicModel(object):

    def __init__(self, model_path,
                 config_path,
                 train=False,
                 file_path=None):
        self.model_path = model_path
        self.config_path = config_path
        if not train:
            self.dictionary, self.tf_idf = self.load_config()
            self.model = self.load_model()
        else:
            self.file_path = file_path
            self.dictionary, self.tf_idf, self.model = self.train()

    def train(self):
        corpus = self.preprocess()
        dictionary = Dictionary(corpus)
        doc2bow = [dictionary.doc2bow(text) for text in corpus]

        tf_idf = TfidfModel(doc2bow)
        corpus_tf_idf = tf_idf[doc2bow]

        model = LdaModel(corpus_tf_idf, num_topics=2)
        return dictionary, tf_idf, model

    def save_model(self):
        self.model.save(self.model_path)

    def load_model(self):
        try:
            model = LdaModel.load(self.model_path)
        except FileNotFoundError:
            model = None
        return model

    def predict(self, text):
        line_cut = list(jieba.cut(text))
        doc2bow = self.dictionary.doc2bow(line_cut)
        corpus_tf_idf = self.tf_idf[doc2bow]
        return self.model[corpus_tf_idf]

    def save_config(self):
        with open(self.config_path, 'wb') as file:
            pickle.dump((self.dictionary, self.tf_idf), file)

    def load_config(self):
        with open(self.config_path, 'rb') as file:
            dictionary, tf_idf = pickle.load(file)
        return dictionary, tf_idf

    def preprocess(self):
        # 读取文件夹下所有的文件名
        files = os.listdir(self.file_path)
        corpus = []
        for file in files:
            dir_ = os.path.join(self.file_path, file)
            with open(dir_, 'r', encoding='utf-8') as file_:
                line = file_.read()
                line_cut = list(jieba.cut(line))
                corpus.append(line_cut)
        return corpus
