# coding: utf-8
from gensim.models import word2vec, KeyedVectors

from nlp.utils.pre_process import process_data


class GensimWord2VecModel:

    def __init__(self, train_file,
                 model_path,
                 embed_size=100,
                 vocab_path=None):
        """
        用gensim word2vec 训练词向量
        :param train_file: 分好词的文本
        :param model_path: 模型保存的路劲
        """
        self.train_file = train_file
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.embed_size = embed_size
        # self.model = self.load()
        self.model = self.load_text()
        if not self.model:
            self.model = self.train()
            # self.save()
            self.save_text()

    def train(self):
        sentences = process_data(self.train_file)
        model = word2vec.Word2Vec(sentences, min_count=2, window=3, size=self.embed_size, workers=4)
        return model

    def vector(self, word):
        return self.model.wv.get_vector(word)

    def similar(self, word):
        return self.model.wv.similar_by_word(word, topn=10)

    def save(self):
        self.model.save(self.model_path)

    def save_text(self):
        self.model.wv.save_word2vec_format(self.model_path, self.vocab_path, False)

    def load(self):
        # 加载模型文件
        try:
            model = word2vec.Word2Vec.load(self.model_path)
        except FileNotFoundError:
            model = None
        return model

    def load_text(self):
        try:
            model = KeyedVectors.load_word2vec_format(self.model_path, self.vocab_path, binary=False)
        except FileNotFoundError:
            model = None
        return model
