#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
parse book content to proper structure
remove special characters, split into pages, paragraphs, sentences
'''

import string
import nltk
import re
from gensim.utils import tokenize
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.util import skipgrams
import pandas as pd
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# print sys.version
stop = set(stopwords.words('german'))


def defineCharacter():
    punc = string.punctuation
    # punc = punc.replace('.','').replace(',','') ## keep comma
    punc += "?\xe2\x80\x93\xe2\x80\x99\x9c\x9e"
    return punc


def removePunctuation(text):
    punc = defineCharacter()
    textAfter = text.replace('-\n', '').translate(None, punc).strip().replace('/', ' ')  # remove punctuation
    textAfter = re.sub('\n+', '\n', textAfter)  # remove repetitive \n
    textAfter = re.sub(' +', ' ', textAfter).strip()       ## remove repetitive space
    return textAfter


def text2Token(paragraphs):
    # split into paragraphs by '\n\n'
    # paragraphs = textAfter.split('\n\n')
    print('the book has %d paragraphs. ' %len(paragraphs))
    tokensAllParagraph = []
    for p in paragraphs:
        tokens = tokenize(p)
        if tokens:
            tokensAllParagraph.append(tokens)
    return tokensAllParagraph


class skipNgramTokenize:
    def __init__(self, order=6, n=2, k=10):

        self.order = order
        self.n = n
        self.k = k

    def tokenize(self, passage):
        tokens = [token.lower() for token in nltk.word_tokenize(passage) if (token not in stop and len(token)>1 and not hasDigit(token))]
        if len(tokens) >= 5:
            return tokens
        else:
            return []

    def ngramTokenize(self, passage):
        tokens = self.tokenize(passage)
        Ngrams = [' '.join(gram) for gram in ngrams(tokens, self.order)]
        return Ngrams

    def skipgramTokenize(self, passage):
        tokens = self.tokenize(passage)
        Skipgrams = [' '.join(gram) for gram in skipgrams(tokens, self.n, self.k)]
        return Skipgrams


def hasDigit(string):
    if type(string) == str:
        if filter(str.isdigit, string):
            return True
        else:
            return False
    elif type(string) == unicode:
        if filter(unicode.isdigit, string):
            return True
        else:
            return False


def clean(text):
    text = unidecode(unicode(text, encoding='utf-8')).strip()
    textAfter = removePunctuation(text)
    paragraphs = textAfter.split('\n')
    return paragraphs


def parse(text):
    paragraphs = clean(text)
    tokensAll = text2Token(paragraphs)
    return tokensAll


def getTxtByID(barcode):
    with open('/home/lyu/travelogues/code/manifests/%s.txt' %barcode, 'r')as f:
        txt = f.read()
    return txt


# compute overall length of tokens
def tokenLength(tokens):
    LengthWithFreq = []
    LengthWithoutFreq = []
    for token in tokens:
        for t in token:
            length = len(t)
            if length == 1:
                pass
            else:
                LengthWithFreq.append(length)
    tokensFlat = list(set([t for token in tokens for t in token]))
    for token in tokensFlat:
        length = len(token)
        if length == 1:
            pass
        else:
            LengthWithoutFreq.append(length)
    return LengthWithFreq, LengthWithoutFreq


def tokenLengthStatistic(LengthWithFreq, LengthWithoutFreq, barcode):
    seriesLengthWithFreq = pd.Series(LengthWithFreq)
    seriesLengthWithoutFreq = pd.Series(LengthWithoutFreq)
    valueCountsWith = seriesLengthWithFreq.value_counts().sort_index(ascending=True)
    valueCountsWithout = seriesLengthWithoutFreq.value_counts().sort_index(ascending=True)

    with open('TokenLengthWithFreq%s.csv' %barcode,'wb')as f:
        for key, value in valueCountsWith.iteritems():
            writer = csv.writer(f)
            writer.writerow( [key, value])

    with open('TokenLengthWithoutFreq%s.csv' %barcode,'wb')as f:
        for key, value in valueCountsWithout.iteritems():
            writer = csv.writer(f)
            writer.writerow([key, value])


def tokenLengthWrapup(tokens, barcode):
    LengthWithFreq, LengthWithoutFreq = tokenLength(tokens)
    tokenLengthStatistic(LengthWithFreq, LengthWithoutFreq, barcode)


if __name__ == '__main__':
    punc = defineCharacter()
    barcode = 'Z165523902'
    text = getTxtByID(barcode)

    tokens = parse(text)
    tokenLengthWrapup(tokens, barcode)

