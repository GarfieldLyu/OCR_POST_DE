#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from argparse import ArgumentParser
from keras.callbacks import *
from keras.optimizers import *
import statistic, nlc_preprocess, networks
from nlc_preprocess import get_tokenizer, prepare_nlc_data
# tf > 2.x runs eager execution by default, while custom loss doesn't work under eager mode.
tf.compat.v1.disable_eager_execution()


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tokenizer", help="tokenizer: [char, ngram, char2ngram]")
    parser.add_argument("-d", "--dataset", help="dataset used for training, pickle pairs.")
    parser.add_argument("-dir", "--directory", help="project directory.")
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
    parser.add_argument("-alpha", "--alpha", type=float, default=0.1, help="custom loss ratio alpha")
    parser.add_argument("-use_cnn", "--use_cnn", type=int, default=1,  help="use cnn units?, train standard enc-dec model without cnn")
    args = parser.parse_args()
    return args


def update_args():
    args = get_cmd_args()
    args.tokenizer = get_tokenizer(args.tokenizer)
    args.data_path = os.path.join(args.directory, 'PKL/' + args.dataset + '.pkl')
    args.vocab_path = os.path.join(args.directory, 'PKL/' + args.dataset + '_vocab.pkl')
    args.model_dir = os.path.join(args.directory, 'Model/' + args.name + '.model')
    if args.glu == 0:
        args.kernels = range(args.kernels, args.kernels+args.layer_num)
    return args


def prepare_data(args):
    print(' read and pad dataset...')
    data_x, data_y, max_length, vocab_size, vocab, vocab_reverse = prepare_nlc_data(args.data_path, args.vocab_path, args.tokenizer, max_vocab_size=args.n_features)
    args.vocab, args.vocab_reverse, args.n_features, args.max_length = vocab, vocab_reverse, vocab_size, max_length
    target = [sent[1:] for sent in data_y]
    input_tensor = keras.preprocessing.sequence.pad_sequences(data_x, maxlen=max_length, padding='post', value=0)
    output_tensor = keras.preprocessing.sequence.pad_sequences(data_y, maxlen=max_length, padding='post', value=0)
    target_tensor = keras.preprocessing.sequence.pad_sequences(target, maxlen=max_length, padding='post', value=0)
    print(input_tensor.shape, output_tensor.shape)
    return input_tensor, output_tensor, target_tensor, args


class ConvRnn:
    def __init__(self, arguments):
        self.args = arguments
        self.model = None

    def train(self, input_tensor, output_tensor, target_tensor):
        self.input_tensor, self.output_tensor, self.target_tensor, = input_tensor, output_tensor, target_tensor
        if not self.args.use_cnn:
            print('train attention based encoder decoder model without cnn...')
            model = networks.define_simple_enc_dec(self.args.units, self.args.embedding_dim, self.args.n_features, self.args.max_length, custom=self.args.custom, alpha=self.args.alpha)
        else:
            if self.args.glu == 0:
                model = networks.define_cnn_rnn(self.args.units, self.args.embedding_dim, self.args.n_features, self.args.max_length, self.args.kernels, custom=self.args.custom, alpha=self.args.alpha)
            else:
                if self.args.return_blocks == 1:
                    all_blocks = True
                    model = networks.define_glu_rnn(self.args.units, self.args.embedding_dim, self.args.n_features, self.args.max_length, self.args.kernels, self.args.layer_num, all_blocks, custom=self.args.custom, alpha=self.args.alpha)
                else:
                    model = networks.define_glu_rnn_single(self.args.units, self.args.embedding_dim, self.args.n_features, self.args.max_length, self.args.kernels, self.args.layer_num, custom=self.args.custom, alpha=self.args.alpha)

        self.target_tensor = keras.utils.to_categorical(self.target_tensor, num_classes=self.args.n_features)
        model.fit([self.input_tensor, self.output_tensor], self.target_tensor, batch_size=self.args.batch,
                  epochs=self.args.epochs, validation_split=float(1) / 10, shuffle=True, verbose=2)
        model.save(self.args.model_dir)
        return model

    def create_model(self, inp, out, target):
        if os.path.exists(self.args.model_dir):
            print('load model from ' + self.args.model_dir)
            if self.args.custom == 0:
                model = load_model(self.args.model_dir)
            else:
                model = load_model(self.args.model_dir, custom_objects={'loss': networks.custom_loss(inp, out, self.args.alpha)})

        else:
            print('start train new model {}...'.format(self.args.name))
            model = self.train(inp, out, target)
        self.model = model
        return model

    def batch_iter(self, lists, batch):
        l = len(lists)
        for ndx in range(0, l, batch):
            yield lists[ndx: ndx + batch]

    def generate_in_batch(self, inputs, batch, toids=0):  # inputs tuples = (ocr_lists, truth_lists)
        # tokens to ids, if inputs are char sentences, convert to ids. otherwise not.
        if toids:
            ocr_input, truth_lists = [], []
            for ocr, truth in inputs:
                ocr_ids = nlc_preprocess.sentence_to_token_ids(ocr, self.args.vocab, self.args.tokenizer)
                ocr_input.append(ocr_ids)
                truth_ids = nlc_preprocess.sentence_to_token_ids(truth, self.args.vocab, self.args.tokenizer)[1:-1]
                truth_lists.append(truth_ids)
        else:
            ocr_input, truth_lists = inputs[0], inputs[1]

        outputs = []
        for inputs_batch in self.batch_iter(ocr_input, batch):
            number = len(inputs_batch)
            encoder_inputs = keras.preprocessing.sequence.pad_sequences(inputs_batch, maxlen=self.args.max_length, padding='post', value=0)
            decoder_inputs = np.zeros(shape=(number, self.args.max_length))
            decoder_inputs[:, 0] = nlc_preprocess.SOS_ID
            for i in range(1, self.args.max_length):
                output = self.model.predict([encoder_inputs, decoder_inputs])
                output = output.argmax(axis=2)
                decoder_inputs[:, i] = output[:, i-1]

            out_batch = decoder_inputs[:, 1:]
            for b in range(number):
                out = out_batch[b, :].tolist()
                out_s = []
                for char_id in out:
                    if char_id == nlc_preprocess.EOS_ID:
                        break
                    else:
                        out_s.append(int(char_id))
                outputs.append(out_s)

        W, C = self.edit_dist(outputs, truth_lists, self.args.vocab.get(' '))
        outputstr = [''.join([self.args.vocab_reverse[o] for o in sent]) for sent in outputs]
        return W, C, outputstr

    def edit_dist(self, recs, refs, ws):
        W, C = [],[]
        for rec, ref in zip(recs, refs):
            w = statistic.word_error_rate(rec, ref, ws)
            c = statistic.char_error_rate(rec, ref)
            W.append(w)
            C.append(c)
        return W, C

    def generate(self, sentence):
        encoder_input = nlc_preprocess.sentence_to_token_ids(sentence, self.args.vocab, self.args.tokenizer)
        encoder_input = keras.preprocessing.sequence.pad_sequences([encoder_input], maxlen=self.args.max_length, padding='post', value=0)
        decoder_input = np.zeros(shape=(1, self.args.max_length))
        decoder_input[:, 0] = nlc_preprocess.SOS_ID
        for i in range(1, self.args.max_length):
            output = self.model.predict([encoder_input, decoder_input]).argmax(axis=2)
            decoder_input[:, i] = output[:, i - 1]
        return decoder_input[0, 1:]

    def decode(self, decoding):
        text = ''
        for i in decoding:
            if i == nlc_preprocess.EOS_ID:
                break
            if i == nlc_preprocess.UNK_ID:  # _UNK
                continue
            text += self.args.vocab_reverse[i]
        return text

    def translate(self, sentence):   # ocr sentence to post_hoc sentence.
        # print(sentence)
        decoding = self.generate(sentence)
        text = self.decode(decoding)
        return text


def main():
    args = update_args()
    inp, out, target, args = prepare_data(args)
    OCR_corrector = ConvRnn(args)
    model = OCR_corrector.create_model(inp, out, target)
    OCR_corrector.translate('IchhabeeinHund')
    '''
    It is also possible to translate sentences in batch 
    OCR_corrector.generate_in_batch(sentences_list, batch, toids=0)
    '''


if __name__ == '__main__':
    main()



