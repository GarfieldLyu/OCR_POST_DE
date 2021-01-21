#!/usr/bin/env python
# -*- coding: utf-8 -*-

# apply word segmentation to words
import pickle
import sys
import os
import ngramWiki
import pycrfsuite
import re
from time import time
import pickle
import random
from sklearn.model_selection import KFold

# dir_root = '/home/lyu/travelogues/'


def prepare_corpus():
    # build sentence and character tag
    current = time()
    path = 'crf_tag_text.pickle'
    if os.path.isfile(path):
        with open(path, 'rb')as f:
            prepared_tag_text = pickle.load(f)

    else:
        dewiki = ngramWiki.get_all_training_corpus()
        # random select samples from dewiki
        dewiki = random.sample(dewiki, 10000)
        prepared_tag_text = []
        for text in dewiki:
            lengths = [len(char) for char in text.split()]
            positions = []

            next_pos = 0
            for length in lengths:
                next_pos = next_pos + length
                positions.append(next_pos)

            # concate = text.replace(' *','')
            concate = re.sub(' *', '', text)
            chars = [c for c in concate]
            labels = [0 if not i in positions else 1 for i, c in enumerate(concate)]
            prepared_tag_text.append(list(zip(chars, labels)))
    
        with open(path, 'wb')as f:
            pickle.dump(prepared_tag_text, f)
    
    after = time()
    print('it costs ' + str((after-current)/3600) + ' hours to tag text.')
    return prepared_tag_text


def create_char_features(text, i):
    
    features = [
            'bias',
            'char='+text[i][0]
            ]
    if i >= 1:
        features.extend([
            'char-1='+text[i-1][0],
            'char-1:0='+text[i-1][0]+text[i][0],
            ])
    else:
        features.append('BOS')

    if i >= 2:
        features.extend([
            'char-2='+text[i-2][0],
            'char-2:0='+text[i-2][0]+text[i-1][0]+text[i][0],
            'char-2:-1='+text[i-2][0]+text[i-1][0],
            ])
    if i >= 3:
        features.extend([
            'char-3='+text[i-3][0],
            'char-3:0='+text[i-3][0]+text[i-2][0]+text[i-1][0]+text[i][0],
            ])

    if i >= 4:
        features.extend([
            'char-4='+text[i-4][0],
            'char-4:0='+text[i-4][0]+text[i-3][0]+text[i-2][0]+text[i-1][0]+text[i][0],
        ])

    if i >= 5:
        features.extend([
            'char-5='+text[i-5][0],
            'char-5:0='+text[i-5][0]+text[i-4][0]+text[i-3][0]+text[i-2][0]+text[i-1][0]+text[i][0],
        ])

    if i >= 6:
        features.extend([
            'char-6='+text[i-6][0],
            'char-6:0='+text[i-6][0]+text[i-5][0]+text[i-4][0]+text[i-3][0]+text[i-2][0]+text[i-1][0]+text[i][0],
        ])

    if i >= 7:
        features.extend([
            'char-7='+text[i-7][0],
            'char-7:0='+text[i-7][0]+text[i-6][0]+text[i-5][0]+text[i-4][0]+text[i-3][0]+text[i-2][0]+text[i-1][0]+text[i][0],
        ])
    if i >= 8:
        features.extend([
            'char-8='+text[i-8][0],
            'char-8:0='+text[i-8][0]+text[i-7][0]+text[i-6][0]+text[i-5][0]+text[i-4][0]+text[i-3][0]+text[i-2][0]+text[i-1][0]+text[i][0],
        ])

    return features


def create_text_features(text):
    return [create_char_features(text, i) for i in range(len(text))]


def create_char_labels(text):
    return [str(part[1]) for part in text]


def create_X_Y(prepared_tag_text):
    current = time()
    print('start constructing training and test dataset......')
    X = [create_text_features(text) for text in prepared_tag_text]
    Y = [create_char_labels(text) for text in prepared_tag_text]
    
    after = time()
    print('it costs ' + str((after-current)/3600) + ' hours to build training and testing data.')
    # return X_train, Y_train, X_test, Y_test
    return X, Y


def build_crf(X_train, Y_train):
    current = time()
    print('start training crf tagger......')
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, Y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': 60,
        'feature.possible_transitions': True
        })

    dir_crf = 'dewiki_segmentation.crfsuite'
    trainer.train(dir_crf)
    tagger = pycrfsuite.Tagger()
    tagger.open(dir_crf)
    
    after = time()
    print('it costs ' + str((after-current)/3600) + ' hours to train crf tagger.')
    return tagger


def evaluate(tagger, X_test, Y_test):
    tp = 0
    fp = 0
    fn = 0
    n_correct = 0
    n_false = 0

    for x, y in zip(X_test, Y_test):
        prediction = tagger.tag(x)
        zipped = list(zip(prediction, y))

        # print zipped
        tp += len([l for l, c in zipped if l==c and l=='1'])
        fp += len([l for l, c in zipped if l=='1' and c=='0'])
        fn += len([l for l, c in zipped if l=='0' and c=='1'])
        n_correct += len([l for l, c in zipped if l==c])
        n_false += len([l for l, c in zipped if l!=c])

    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    accuracy = float(n_correct)/(n_correct+n_false)
    print(precision, recall, accuracy)


def wrapup_crf():
    # build and train crf tagger.
    kfold = KFold(10, True, 1)
    prepared_tag_text = prepare_corpus()
    for train, test in kfold.split(prepared_tag_text):
        train_data = [prepared_tag_text[i] for i in train]
        test_data = [prepared_tag_text[i] for i in test]
        X_train, Y_train = create_X_Y(train_data)
        X_test, Y_test = create_X_Y(test_data)

        # X_train, Y_train, X_test, Y_test = create_X_Y(prepared_tag_text)
        tagger = build_crf(X_train, Y_train)
        evaluate(tagger, X_test, Y_test)


def get_tagger(dir_crf='dewiki_segmentation.crfsuite'):  # trained
    tagger = pycrfsuite.Tagger()
    tagger.open(dir_crf)
    return tagger


def segment_text(tagger, text):
    text = text.replace(' ', '')
    prediction = tagger.tag(create_text_features(text))
    complete = ""
    for i, p in enumerate(prediction):
        if p == "1":
            complete += " " + text[i]
        else:
            complete += text[i]
    return complete


def segment_text_pipe(dir_crf, text):
    tagger = pycrfsuite.Tagger()
    tagger.open(dir_crf)
    seg_text = segment_text(tagger, text)
    return seg_text


if __name__ == '__main__':
    # try a trained crftagger.
    current = time()
    # wrapup_crf()
    tagger = get_tagger()
    text = 'vnnDhabenSpass'
    print(segment_text(tagger, text))

    after = time()
    print('it costs ' +str((after-current)/3600) + 'hours in total.')
