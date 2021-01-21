#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
construct n-gram character based language model for dewiki
created by @lijun, 31-05-2018, good weather
'''

import pickle
# import parseText
import re
import os
import string
from random import random
from collections import *
import nltk
from nltk.tokenize import sent_tokenize
import json

punc = string.punctuation
punc += "?\xe2\x80\x93\xe2\x80\x99\x9c\x9e"
dir_root = "/home/lyu/travelogues/"


# @path address already deprecated
def read(path = '/home/lyu/travelogues/code/WikipediaDE/dewiki_samples.pickle'):
    with open(path, 'rb')as f:
        dewiki_samples = pickle.load(f)
    return dewiki_samples


def parseText(article):
    if type(article) == unicode:
        article = article.encode('utf-8')
    article = article.translate(None, punc).strip()  # remove punctuation
    article = re.sub('\n+', ' ', article)  # remove \n
    article = re.sub(' +', ' ', article).strip()  # remove repeative space

    return article
    

def parseWikisource(path=dir_root+'/OCR/PKL/wikisource_json.pickle'):
    dir_wikisource = dir_root + '/OCR/PKL/pureText_wikisource.pickle'
    if os.path.isfile(dir_wikisource):
        with open(dir_wikisource, 'rb')as f:
            pureText = pickle.load(f)
    else:
        with open(path, 'rb')as f:
            wikisource = pickle.load(f)
        Text = [page['text'] for page in wikisource]
        pureText = [parseText(article) for article in Text]
        with open(dir_wikisource, 'wb')as f:
            pickle.dump(pureText, f)

    return pureText


def text2lines(path='pickle/wikisource_json.pickle'):
    with open(path,'rb')as f:
        wikisource = pickle.load(f)

    Text = [page['text'] for page in wikisource]
    lines = [s for text in Text for s in sent_tokenize(text)]
    
    with open('pickle/wikisource_sentences.txt', 'w')as f:
        for s in lines:
            s = re.sub('\n+', '', s)
            s = re.sub(' +', ' ', s)
            if len(s) > 3:
                f.write(s+'\n')


def parseWikiSamples():
    # first remove all punctuation
    dewiki_samples = read()
    pureText = []
    for article in dewiki_samples:
        article = parseText(article)
        pureText.append(article)
    with open(dir_root+'OCR/PKL/pureText_dewiki_samples.pickle', 'wb')as f:
        pickle.dump(pureText, f)
    return pureText


def getWikiPureText(path=dir_root+'OCR/PKL/pureText_dewiki_samples.pickle'):
    # pureText = read(path)
    # return ' '.join(pureText)
    with open(path, 'rb')as f:
        pureText = pickle.load(f)
    return pureText


def getNewsDTA17(path=dir_root+'XMLParser/parsed/Zeitung_17.json'):
    dir_text = dir_root + 'OCR/PKL/Zeitung17.pickle'
    if os.path.isfile(dir_text):
        with open(dir_text, 'rb')as f:
            Zeitung17_text = pickle.load(f)
    else:
        # read json file
        with open(path)as f:
            Zeitung17_text = json.load(f)['texts']  # Zeitung_17 contains keys: persons, type, places, texts
        # parse raw text
        Zeitung17_text = [parseText(text) for text in Zeitung17_text]
        # save to pickle file
        with open(dir_text, 'wb')as f:
            pickle.dump(Zeitung17_text, f)
            print('Successfully save Zeitung_17 texts to {}'.format(dir_text))

    return Zeitung17_text


def get_all_training_corpus():
    # include dewiki, wikisource, newspaper from 17th century from DTA
    print('Extract dewiki texts...')
    dewiki = getWikiPureText()
    print('Extract wikisource texts...')
    wikisource = parseWikisource()
    print('Extract DTA news17 texts...')
    zeitung17 = getNewsDTA17()

    dewiki = dewiki+wikisource+zeitung17
    print('Including {} texts in total...'.format(len(dewiki)))
    return dewiki


def trainNGramCharLM(text, order):
    lm = defaultdict(Counter)
    pad = '~' * order
    text = pad + text

    for i in range(len(text)-order):
        history, char = text[i:i+order], text[i+order]
        lm[history][char] += 1

    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c, cnt/s) for c, cnt in counter.items()]

    outlm = {hist: normalize(chars) for hist, chars in lm.items()}

    return outlm


def generate_letter(lm, history, order):
    history = history[-order:]
    dist = lm[history]
    x = random()

    for c, v in dist:
        x = x-v
        if x <= 0:
            return c


def wrapupTrainNGramCharLM(order=6):
    # text = getWikiPureText()
    text = get_all_training_corpus().lower()
    text = ' '.join(text)
    lm = trainNGramCharLM(text, order)

    with open('pickle/LM_char_level_%dgram.pickle' %order, 'wb')as f:
        pickle.dump(lm, f)

    return lm


def ngramLM(order=2):
    grams = []
    wiki = getWikiPureText()
    for article in wiki:
        tokens = nltk.word_tokenize(article.lower())
        grams += list(nltk.ngrams(tokens, order))

    freq_dist = nltk.FreqDist(grams)

    for key in freq_dist.keys():
        freq_dist[key] = freq_dist[key]/float(len(grams))
    
    # save language model
    with open('pickle/LM_word_level_%dgram.pickle'%order,'wb')as f:
        pickle.dump(freq_dist, f)
    #r eturn freq_dist


if __name__ == '__main__':
    # parseWikiSamples()
    # get_all_training_corpus()
    # wrapupTrainNGramCharLM(order=2)
    # ngramLM(1)
    text2lines()
