# -*- coding: utf-8 -*-


import os
import pickle
import numpy as np
import nltk
from nltk import ngrams

_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def get_data(path):
    with open(path, 'rb')as f:
        sentence_pair = pickle.load(f)
    return sentence_pair


def char_tokenize(sentence):
    return list(sentence)


def ngram_tokenize(sentence, order=3):
    grams = [''.join(n) for n in list(ngrams(sentence, order))]
    return grams


def ngram_tokenize_by_char(sentence, order=3):
    grams = [list(n) for n in list(ngrams(sentence, order))]
    return grams


def get_tokenizer(tokenizer):
    if tokenizer == 'char':
        return char_tokenize
    elif tokenizer == 'ngram':
        return ngram_tokenize
    else:
        return char_tokenize

