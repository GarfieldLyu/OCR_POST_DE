#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import keras
import nlc_preprocess, ocr_corrector
from nlc_preprocess import get_tokenizer
import time
import os
import pickle
from argparse import ArgumentParser
from tqdm import tqdm
tokenizer = get_tokenizer('char')


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tokenizer", help="tokenizer: [char, ngram, char2ngram]")
    parser.add_argument("-dir", "--directory", default="/home/lyu/travelogues/OCR/")
    parser.add_argument("-name", "--name",help="name the model")
    parser.add_argument("-glu", "--glu", type=int, help="use multi-layer gated linear units?")
    parser.add_argument("-blocks", "--return_blocks", type=int, default=0, help="multi_layer attention")
    parser.add_argument("-k", "--kernels", type=int, help="kernel size")
    parser.add_argument("-layers", "--layer_num", default=3, type=int, help="layer numbers")
    parser.add_argument("-units", "--units", type=int, default=256)
    parser.add_argument("-emb_dim", "--embedding_dim", type=int, default=128)
    parser.add_argument("-epochs", "--epochs", type=int, default=10)
    parser.add_argument("-batch", "--batch", type=int, default=64)
    parser.add_argument("-n_features", "--n_features", type=int, default=4000)
    parser.add_argument("-custom", "--custom", type=int, default=0, help="use custom loss or not")
    parser.add_argument("-alpha", "--alpha", type=float, help="custom loss ratio alpha")
    parser.add_argument("-use_cnn", "--use_cnn", type=int, default=1, help="use cnn units?")
    # CNN parameters
    args = parser.parse_args()
    return args


def update_args():
    args = get_cmd_args()
    if args.glu == 0:
        args.kernels = range(args.kernels, args.kernels+args.layer_num)
    return args


def get_random_splits(directory):
    data_dir = directory + 'PKL/GTdata/Experiment_corpus_str.pkl'
    with open(data_dir, 'rb')as f:
        corpus = pickle.load(f)
    return corpus[0], corpus[1]  # train, test, 3 groups each


def build_vocab(pairs, max_vocab_size=4000):
    vocab, vocab_reverse = {}, {}
    for (inp, out) in pairs:
        tokens = tokenizer(inp) + tokenizer(out)
        for w in tokens:
            if w in vocab:
                vocab[w] += 1
            else:
                vocab[w] = 1
    vocab_list = nlc_preprocess._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocab_size:
        vocab_list = vocab_list[:max_vocab_size]
    vocab = dict([(y, x) for (x, y) in enumerate(vocab_list)])
    vocab_reverse = dict([(x, y) for (x, y) in enumerate(vocab_list)])
    return vocab, vocab_reverse


def sents_2_ids(pairs, vocab):
    data_x, data_y = [], []
    for (inp, out) in pairs:
        tokens_inp = tokenizer(inp)
        tokens_out = tokenizer(out)
        ids_inp = [1] + [vocab.get(w, 2) for w in tokens_inp] + [2]
        ids_out = [1] + [vocab.get(w, 2) for w in tokens_out] + [2]
        data_x.append(ids_inp)
        data_y.append(ids_out)
    return data_x, data_y


def ids_2_tensors(data_x, data_y, max_length):
    target = [sent[1:] for sent in data_y]
    input_tensor = keras.preprocessing.sequence.pad_sequences(data_x, maxlen=max_length, padding='post', value=0)
    output_tensor = keras.preprocessing.sequence.pad_sequences(data_y, maxlen=max_length, padding='post', value=0)
    target_tensor = keras.preprocessing.sequence.pad_sequences(target, maxlen=max_length, padding='post', value=0)

    ''' without one_hot_encode'''
    input_tensor = np.array(input_tensor)
    output_tensor = np.array(output_tensor)
    target_tensor = np.array(target_tensor)
    print(input_tensor.shape, output_tensor.shape)
    return input_tensor, output_tensor, target_tensor


def main():
    args = update_args()
    print('read randomly split  dataset...')
    Train_groups, Test_groups = get_random_splits(args.directory)
    for flag in range(2, len(Train_groups)):
        print('Train experiment group {}'.format(flag))
        Train_data = Train_groups[flag]
        vocab, vocab_reverse = build_vocab(Train_data)
        data_x, data_y = sents_2_ids(Train_data, vocab)
        max_length = max([len(s) for s in data_y])
        print('vocab size: {} || max length: {}'.format(len(vocab), max_length))
        args.vocab, args.vocab_reverse, args.n_features, args.max_length = vocab, vocab_reverse, len(vocab), max_length
        input_tensor, output_tensor, target_tensor = ids_2_tensors(data_x, data_y, max_length)
        
        ''' start train models  '''
        args.model_dir = args.directory + 'Model/proposal/{}_{}.model'.format(args.name, flag)
        nlc_convrnn = ocr_corrector.ConvRnn(args)
        model = nlc_convrnn.create_model(input_tensor, output_tensor, target_tensor)

        results_dir = args.directory + 'Evaluation/proposal/' + args.name+'_'+str(flag) + '/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        out_txt = open(os.path.join(results_dir, 'wercer.txt'), 'w')

        ''' start test models in 3 groups'''
        for j in tqdm(range(len(Test_groups))):
            Test_data = Test_groups[j]
            before = time.time()
            test_x, test_y = sents_2_ids(Test_data, vocab)
            test_y = [y[1:-1] for y in test_y]
            W, C, outputs = nlc_convrnn.generate_in_batch([test_x, test_y], args.batch)
            W_avg = sum(W)/len(W)
            C_avg = sum(C)/len(C)
            after = time.time()
            print('{}: AVG WER: {} || CER: {}'.format(j, W_avg, C_avg))
            print('It takes {} mins to test.'.format((after-before)/60))
            out_txt.write('AVG WER: {} || CER: {}\n'.format(W_avg, C_avg))
            with open(os.path.join(results_dir, 'post_{}.pkl'.format(j)), 'wb')as f:
                pickle.dump(outputs, f)
        out_txt.close()
        print('================================================')


if __name__ == '__main__':
    main()


