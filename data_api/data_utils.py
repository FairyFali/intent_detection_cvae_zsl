# -*- coding: utf-8 -*-
# @Time   : 2019/11/7 15:41:58
# @Author : Wang Fali

import numpy as np


# Data feed
class LongDataLoader(object):
    """A special efficient data loader for TBPTT"""
    batch_size = 0
    ptr = 0
    num_batch = None # number of batch
    batch_indexes = None
    indexes = None #
    data_lens = None # len of each utt, []
    data_size = None # number of utt
    prev_alive_size = 0
    name = None

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, cur_grid, prev_grid):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size
        # self.indexes is the index of utt, order by asc
        # self.data_lens is the len of utt, by index

        self.ptr = 0 # current batch
        self.batch_size = batch_size # 30
        self.prev_alive_size = batch_size # 30

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = [] # The element is a list, storing each batch
        for i in range(temp_num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        left_over = self.data_size-temp_num_batch*batch_size # Remainder data

        # shuffle batch indexes
        if shuffle:
            self._shuffle_batch_indexes()

        self.num_batch = len(self.batch_indexes) # number of batch
        print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_id = self.ptr
            if self.ptr > 0:
                prev_id = self.ptr-1
            else:
                prev_id = None
            self.ptr += 1
            return self._prepare_batch(cur_id=current_id, prev_id=prev_id)
        else:
            return None


class ATISDataLoader(LongDataLoader):
    def __init__(self, name, data, config):
        '''
        :param name: Train/Test/Valid
        :param data: utterance data
        :param config: config
        '''
        self.name = name
        self.data = data.reset_index(drop=True)
        self.data_size = len(data) # utt number
        self.data_lens = all_lens = [len(line) for line in self.data['content_words']] # each utt's length
        self.max_utt_size = config.max_utt_len # 40
        print(self.name, "Max len %d and min len %d and avg len %f" % (np.max(all_lens),
                                                            np.min(all_lens),
                                                            float(np.mean(all_lens))))
        self.indexes = list(np.argsort(all_lens)) # Sort by sentence length, element is sentence index

        self.most_similarity =   [[6, 13, 3, 8, 5, 2, 11, 10, 9, 1, 14, 15, 12, 17, 16, 4, 7],
                                 [17, 3, 14, 0, 11, 8, 15, 7, 4, 10, 9, 5, 13, 2, 6, 12, 16],
                                 [11, 6, 13, 16, 0, 10, 5, 8, 9, 3, 12, 14, 7, 4, 15, 17, 1],
                                 [0, 5, 8, 6, 13, 11, 2, 9, 17, 1, 10, 14, 4, 15, 16, 12, 7],
                                 [7, 15, 17, 10, 3, 8, 9, 14, 0, 16, 12, 5, 1, 2, 11, 13, 6],
                                 [0, 3, 8, 2, 13, 6, 9, 11, 10, 7, 4, 17, 15, 12, 1, 14, 16],
                                 [13, 0, 2, 16, 3, 11, 5, 8, 10, 12, 9, 14, 15, 17, 7, 1, 4],
                                 [9, 15, 4, 17, 14, 10, 5, 2, 11, 3, 1, 8, 0, 12, 6, 13, 16],
                                 [0, 3, 12, 5, 2, 6, 10, 13, 11, 9, 4, 1, 16, 17, 15, 7, 14],
                                 [7, 2, 5, 8, 10, 3, 0, 15, 17, 12, 11, 4, 14, 6, 13, 16, 1],
                                 [2, 8, 11, 9, 3, 0, 6, 4, 15, 17, 12, 7, 5, 16, 13, 14, 1],
                                 [2, 13, 6, 14, 0, 3, 10, 8, 5, 17, 12, 16, 1, 9, 7, 15, 4],
                                 [8, 2, 16, 11, 9, 10, 6, 13, 0, 3, 4, 17, 14, 5, 7, 15, 1],
                                 [6, 0, 2, 11, 3, 5, 16, 8, 12, 10, 9, 15, 14, 1, 7, 17, 4],
                                 [11, 17, 3, 7, 1, 2, 0, 15, 9, 16, 4, 10, 6, 8, 13, 12, 5],
                                 [7, 4, 9, 10, 0, 14, 3, 1, 8, 17, 5, 11, 13, 2, 6, 12, 16],
                                 [2, 6, 13, 12, 11, 8, 10, 14, 3, 0, 4, 9, 17, 5, 7, 15, 1],
                                 [1, 14, 3, 7, 4, 11, 10, 9, 8, 0, 15, 5, 12, 2, 16, 6, 13]]


    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:self.max_utt_size-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (self.max_utt_size-len(tokens))
        else:
            return tokens

    def _prepare_batch(self, cur_id, prev_id):
        # the batch index, the starting point and end point for segment
        batch_ids = self.batch_indexes[cur_id]
        rows = self.data.loc[batch_ids].copy() # batch

        rows["utts"] = rows["content_words"].apply(lambda x:self.pad_to(x, do_pad=True))

        utts = np.array(rows['utts'].tolist())
        utts_lens = np.array(rows['all_lenes'].tolist())
        intents = np.array(rows['label'])

        return utts, utts_lens, intents


class SNIPSDataLoader(LongDataLoader):
    def __init__(self, name, data, config, sim):
        '''
        :param name: Train/Test/Valid
        :param data: utterance data
        :param config: config
        '''
        self.name = name
        self.data = data.reset_index(drop=True) # reset the index
        self.data_size = len(data)  # utt number
        self.data_lens = all_lens = [len(line) for line in self.data['content_words']]  # each utt's length
        self.max_utt_size = config.max_utt_len  # 40
        print(self.name, "Max len %d and min len %d and avg len %f" % (np.max(all_lens),
                                                                       np.min(all_lens),
                                                                       float(np.mean(all_lens))))
        self.indexes = list(np.argsort(all_lens))  # Sort by sentence length, element is sentence index

        self.sim = sim
        self.most_similarity = [[1, 3, 2, 4], [2, 0, 3, 4], [1, 3, 0, 4], [2, 1, 0, 4], [1, 0, 2, 3]]

    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:self.max_utt_size - 1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (self.max_utt_size - len(tokens))
        else:
            return tokens

    def _prepare_batch(self, cur_id, prev_id):
        # the batch index, the starting point and end point for segment
        batch_ids = self.batch_indexes[cur_id]
        rows = self.data.loc[batch_ids].copy()  # batch

        rows["utts"] = rows["content_words"].apply(lambda x: self.pad_to(x, do_pad=True)) # ids

        utts = np.array(rows['utts'].tolist())
        utts_lens = np.array(rows['all_lenes'].tolist())
        intents = np.array(rows['intent'].tolist())

        return utts, utts_lens, intents

