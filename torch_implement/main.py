# -*- coding: utf-8 -*-

import torch
import pickle
import sys
import os
import logging
from dataset import LoadData, CustomData, Corpus
from seq2seq import Train
from argparse import ArgumentParser

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--tokenizer', default='char', help='tokenizer')
    # parser.add_argument('-d', '--dataset', help='[travel, century17]')
    parser.add_argument('-random', '--random_group', type=int, default=0, help='random split books?')
    parser.add_argument('-d', '--dataset', help='name the dataset')
    parser.add_argument('-m', '--mode', help='train or test')
    parser.add_argument('-dir', '--directory', help='main directory')
    parser.add_argument('-enc_units', '--enc_units', type=int, default=256)
    parser.add_argument('-dec_units', '--dec_units', type=int, default=256)
    parser.add_argument('-emb', '--embedding_dim', type=int, default=128)
    parser.add_argument('-epoch', '--epoch', type=int, default=10)
    parser.add_argument('-batch', '--batch', type=int, default=64)
    parser.add_argument('-n_features', '--n_features', type=int, default=10000)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.3)
    parser.add_argument('-clip', '--clip', type=float, default=0.1)
    parser.add_argument('-model_name', '--model_name', help='model name')
    parser.add_argument('-tf', '--tf', type=float, default=1.0, help='teacher forcing ratio')
    parser.add_argument('-sparse_max', '--sparse_max', type=int, default=0)
    args = parser.parse_args()

    return args


def load_corpus(args):
    # args = get_cmd_args()
    # prepare the whole corpus
    data_dir = args.directory + 'data/' + args.dataset + '.PKL'
    if args.random_group:  # randomly split the books to groups and save Train/Test
        with open(args.directory + 'data/GT_corpus.PKL', 'rb')as f:
            corpus_dict = pickle.load(f, encoding='bytes')

        corpus = Corpus(corpus_dict)
        Train_groups, Test_groups = corpus.get_pairs_by_group(number=args.random_group)
        # save the training and testing data.
        
        with open(data_dir, 'wb')as f:
            pickle.dump((Train_groups, Test_groups), f)
            print('saved split train/test data to {}.'.format(args.dataset))

    else:
        with open(data_dir, 'rb')as f:
            Train_groups, Test_groups = pickle.load(f)

    return Train_groups, Test_groups


def main():
    args = get_cmd_args()
    Train_groups, Test_groups = load_corpus(args)
    group_num = len(Test_groups)
    for flag in range(group_num):
        print('Experiment: {}'.format(flag))
        args.model_dir = args.directory + 'Model/{}{}.model'.format(args.model_name, flag)
        log_file = args.directory + 'log/{}{}.log'.format(args.model_name, flag)
        logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)
        Train_data = Train_groups[flag]
        Test_data = Test_groups[flag]
        loaddata = LoadData(args.tokenizer, args.n_features)
        loaddata.prepare_corpus(Train_data, Test_data)
        args.inp_dim = args.out_dim = len(loaddata.vocab)
        args.max_len = loaddata.max_length
        vocab = loaddata.vocab  # get white space index
        # vocab_reverse = loaddata.vocab_reverse
        print(args.inp_dim, args.max_len, vocab.get(' '))
        # print(type(list(vocab.keys())[10]))
        task = Train(args.inp_dim, args.out_dim, args.embedding_dim, args.enc_units, args.dec_units, args.dropout,
                     args.dropout, args.epoch, args.clip, args.sparse_max, args.tf, loaddata, args.batch,
                     device, args.model_dir)
        if args.mode == 'train':
            logging.info('start training...')
            task.start_train(loaddata.train, loaddata.valid)

            # also test
            logging.info('start testing themselves: ')
            task.test_in_batch(loaddata.test)
            # test other books
            test_others = [i for i in range(group_num) if i != flag]

            for j in test_others:
                logging.info('start testing other books: {}'.format(j))
                Test_other = Test_groups[j]
                test_data = loaddata.prepare_other_corpus(Test_other)
                task.test_in_batch(test_data)
        else:
            logging.info('start testing...')
            task.test_in_batch(loaddata.test)
            test_others = [i for i in range(group_num) if i != flag]

            for j in test_others:
                logging.info('start testing other books: {}'.format(j))
                Test_other = Test_groups[j]
                test_data = loaddata.prepare_other_corpus(Test_other)
                task.test_in_batch(test_data)
            
            for test in Test_groups:
                test_inp = [t[0] for t in test]
                test_out = [t[1] for t in test]
                translation = task.translate_in_batch(test_inp)
                out = args.directory + 'log/test_text_model{}.txt'.format(flag)
                with open(out, 'a',encoding='utf8') as f:
                    for inp, pred, truth in zip(test_inp, translation, test_out):
                        f.write(inp.decode(errors='ignore'))
                        f.write('\n')
                        f.write(pred)
                        f.write('\n')
                        f.write(truth.decode(errors='ignore'))
                        f.write('\n\n')
            

if __name__ == '__main__':
    main()
