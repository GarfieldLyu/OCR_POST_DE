# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import random
from nlc_preprocess import get_tokenizer
import numpy as np
import logging

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

''' cuda multiprocessing'''
# from multiprocessing import set_start_method
# try:
    # set_start_method('spawn')
# except RuntimeError:
    # pass

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
_START_VOCAB = ["<pad>", "<sos>", "<eos>", "<unk>"]


class Corpus:  # randomly split training and testing data
    def __init__(self, corpus_dict):  # corpus = [{barcode: {error_ratio: xx, book: [[],[],[]...], page_num:xxx}}]
        self.corpus = corpus_dict
        self.barcodes = list(self.corpus.keys())
        # print(self.barcodes)

    def pages_from_id(self, barcode):
        # print(barcode):q
        # print(self.corpus[barcode].keys())
        pages = self.corpus[barcode][b'book']
        return pages

    def get_pairs_flat(self, barcode, ratio=0.8):
        pages = self.pages_from_id(barcode)
        # print('page number: {}'.format(len(pages)))
        train_number = random.sample(range(len(pages)), int(len(pages)*ratio))
        test_number = [p for p in range(len(pages)) if p not in train_number]
        train, test = [],[]
        for p in train_number:
            train += pages[p]
        for p in test_number:
            test += pages[p]
        return train, test

    def get_random_ids(self, number):
        return random.sample(self.barcodes, number)

    def divide_to_groups(self, number=4):  # number books in one group
        barcodes = self.barcodes
        random.shuffle(barcodes)
        # print(barcodes)
        groups = []
        while barcodes:
            groups.append(barcodes[:number])
            barcodes = barcodes[number:]
            # print(barcodes)
        
        return groups

    def get_pairs_by_group(self, number=4, ratio=0.8):  # number books in one group
        self.group_ids = self.divide_to_groups(number)  # remember the random book id
        print('random group book ids: {}'.format(self.group_ids))
        Train_all = []
        Test_all = []
        for group in self.group_ids:
            Train = []
            Test = []
            for barcode in group:
                train, test = self.get_pairs_flat(barcode, ratio)
                Train += train
                Test += test

            Train_all.append(Train)
            Test_all.append(Test)

        return Train_all, Test_all


class LoadData:
    def __init__(self, tokenizer, max_vocab_size):

        self.max_vocab_size = max_vocab_size
        self.tokenizer = get_tokenizer(tokenizer)
    
    def build_vocab_on_the_fly(self, train_corpus):  # read training instances directly
        if type(train_corpus[0][0]) != str:
            train_corpus = [(s.decode('utf8', 'ignore'), t.decode('utf8', 'ignore')) for (s, t) in train_corpus]
        vocab = {}
        ocr = [p[0] for p in train_corpus]
        truth = [p[1] for p in train_corpus]
        length = [len(s) for s in ocr] + [len(t) for t in truth]
        for sent in ocr+truth:
            # sent = sent.decode('utf8', 'ignore')
            tokens = self.tokenizer(sent)
            for w in tokens:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > self.max_vocab_size:
            vocab_list = vocab_list[:self.max_vocab_size]

        self.vocab = dict([(y, x) for (x, y) in enumerate(vocab_list)])
        self.vocab_reverse = dict([(x, y) for (x, y) in enumerate(vocab_list)])
        self.max_length = max(length)
        print('vocabulary size:  {} | max setence length: {}'.format(len(self.vocab), self.max_length))

    def tokens_to_ids(self, sentence):
        # if type(sentence) != str:
            # sentence = sentence.decode('utf8', 'ignore')
        tokens = self.tokenizer(sentence)
        return [SOS_ID] + [self.vocab.get(w, UNK_ID) for w in tokens] + [EOS_ID]  # <sos>, <eos>

    def corpus_to_ids(self, corpus):  # corpus: list of tuples
        data = []
        for (inp, out) in corpus:
            inp_id = self.tokens_to_ids(inp)
            out_id = self.tokens_to_ids(out)
            data.append((inp_id, out_id))
        return data

    def train_valid_split(self, ratio=0.1):
        print('split training and validation dataset.')
        index = int(len(self.data)*(1-ratio))
        self.train = self.data[:index] 
        self.valid = self.data[index:] 
        print('train: {}'.format(len(self.train)))
        print('valid: {}'.format(len(self.valid)))

    def custom_data(self, instances):
        custom = CustomData(instances, self.max_length)
        return custom

    def custom_input(self, instances):
        custom = CustomDataInput(instances, self.max_length)
        return custom
    
    def prepare_corpus(self, Train, Test):
        # here convert data type to str
        if type(Train[0][0]) != str:   # unicode type
            Train = [(s.decode('utf8', 'ignore'), t.decode('utf8', 'ignore')) for (s, t) in Train]
        if type(Test[0][0]) != str:
            Test = [(s.decode('utf8', 'ignore'), t.decode('utf8', 'ignore')) for (s, t) in Test]

        self.build_vocab_on_the_fly(Train)
        self.data = self.corpus_to_ids(Train)
        self.test = self.corpus_to_ids(Test)
        self.train_valid_split()
        # convert to torch dataset object
        self.train = self.custom_data(self.train)
        self.valid = self.custom_data(self.valid)
        self.test = self.custom_data(self.test)

    def prepare_other_corpus(self, Test):  # corpus to ids for other test data using existed vocab
        if type(Test[0][0]) != str:
            Test = [(s.decode('utf8', 'ignore'), t.decode('utf8', 'ignore')) for (s, t) in Test]
        test_ids = self.corpus_to_ids(Test)
        test_custom = self.custom_data(test_ids)
        return test_custom

    def prepare_only_input(self, inputs):
        if type(inputs[0]) != str:
            inputs = [s.decode('utf8', 'ignore') for s in inputs]
        inputs_ids = [self.tokens_to_ids(inp) for inp in inputs] 
        inputs_custom = self.custom_input(inputs_ids)
        return inputs_custom


class CustomData(Dataset):  # pytorch dataset class
    def __init__(self, instances, maxlen):
        self.maxlen = maxlen
        self.padded_data = [ (pad_data(s[0], maxlen), pad_data(s[1], maxlen)) for s in instances]

    def __len__(self):
        return len(self.padded_data)

    def __getitem__(self, idx):
        x = self.padded_data[idx][0]
        y = self.padded_data[idx][1]
        return x, y


class CustomDataInput(Dataset):
    def __init__(self, inputs, maxlen):
        self.maxlen = maxlen
        self.padded_data = [pad_data(s, maxlen) for s in inputs]

    def __len__(self):
        return len(self.padded_data)

    def __getitem__(self, idx):
        x = self.padded_data[idx]
        return x


def pad_data(s, maxlen):
    padded = np.zeros((maxlen,), dtype=np.int64)
    if len(s) > maxlen:
        padded[:] = s[:maxlen]
    else:
        padded[:len(s)] = s
    return padded


if __name__ == '__main__':
    print('test')
