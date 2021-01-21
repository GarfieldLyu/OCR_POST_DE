# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
import time
import os
# sys.path.append('..')
import statistic
import numpy as np
import logging
import entmax
from entmax import sparsemax
from entmax.losses import SparsemaxLoss
from Model import Encoder, Attention, Decoder, Seq2Seq
import dataset
from dataset import pad_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

''' # cuda multiprocessing
from multiprocessing import set_start_method
try:
    set_start_method('spawn', True)
except RuntimeError:
    pass
'''


class Train:
    def __init__(self, inp_dim, out_dim, emb_dim, enc_hid, dec_hid, enc_drop, dec_drop, epoch, clip,
                 sparse_max, tf, Data, batch, device, model_dir):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.enc_hid = enc_hid
        self.dec_hid = dec_hid
        self.enc_drop = enc_drop
        self.dec_drop = dec_drop
        self.tf = tf
        self.max_length = Data.max_length
        self.batch = batch
        self.device = device
        self.vocab = Data.vocab
        self.vocab_reverse = Data.vocab_reverse
        self.tokenizer = Data.tokenizer
        self.DATA = Data
        self.model_dir = model_dir

        self.attn = Attention(enc_hid, dec_hid, sparse_max=sparse_max)
        self.enc = Encoder(inp_dim, emb_dim, enc_hid, dec_hid, enc_drop)
        self.dec = Decoder(out_dim, emb_dim, enc_hid, dec_hid, dec_drop, self.attn)
        self.model = Seq2Seq(self.enc, self.dec, device).to(device)

        self.model.apply(self.init_weights)
        self.count_parameters()
        self.optimizer = optim.Adam(self.model.parameters())
        if sparse_max:
            self.criterion = SparsemaxLoss(ignore_index=0)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad_idx 0
        self.epoch = epoch
        self.clip = clip

        self.load_model(model_dir)

    def init_weights(self, m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def count_parameters(self):
        param_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('The model has {} trainable parameters'.format(param_num))
        return param_num

    def train(self, train_data):
        print('train model...')
        self.model.train()
        epoch_loss = 0
        epoch_accu = 0
        iterator = iter(dataloader.DataLoader(train_data, batch_size=self.batch, num_workers=1, shuffle=True, pin_memory=True))  ## add pin_memory here to use GPU

        for batch in iterator:
            src = batch[0].permute(1, 0)
            trg = batch[1].permute(1, 0)   # trg = [trg sent len, batch size]
            src = src.to(self.device)
            trg = trg.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, trg, self.tf)   # output = [trg sent len, batch size, output dim]
            output = output[1:].view(-1, output.shape[-1])  # output = [(trg sent len - 1) * batch size, output dim]
            trg = trg[1:].view(-1)  # trg = [(trg sent len - 1) * batch size]
            predict = self.prob_to_index(output)
            accuracy = self.Accuracy(predict, trg)
            loss = self.criterion(output, trg)
            # print('loss: ', loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_accu += accuracy

        return epoch_loss / len(iterator), epoch_accu / len(iterator)

    def evaluate(self, valid_data):
        ws = self.vocab.get(' ') # get white space
        self.model.eval()
        epoch_loss = 0
        epoch_accu = 0
        epoch_wer = 0
        epoch_cer = 0
        epoch_wer_ocr = 0
        epoch_cer_ocr = 0
        # print('batch: ', self.batch)
        iterator = iter(dataloader.DataLoader(valid_data, batch_size=self.batch, num_workers=1, shuffle=True, pin_memory=True))  ## add pin_memory here to use GPU
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch[0].permute(1, 0).to(self.device)
                trg = batch[1].permute(1, 0).to(self.device)
                wer, cer = self.Edit_dist_in_batch(src, trg, ws)
                epoch_wer_ocr += wer
                epoch_cer_ocr += cer
                output = self.model(src, trg, 0)  # turn off teacher forcing
                # output = output[1:].view(-1, output.shape[-1]) # squeeze output
                output = output[1:]
                predict = self.prob_to_index(output)
                trg = trg[1:]
                
                wer, cer = self.Edit_dist_in_batch(predict, trg, ws)
                epoch_wer += wer
                epoch_cer += cer

                predict = predict.view(-1) # squeeze both
                trg = trg.view(-1)
                output = output.view(-1, output.shape[-1])
                accuracy = self.Accuracy(predict, trg)
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
                epoch_accu += accuracy
        # print('Accuracy for a batch: ', epoch_accu)
        iter_length = len(iterator)
        return epoch_loss/iter_length, epoch_accu/iter_length, epoch_wer_ocr/iter_length, epoch_cer_ocr/iter_length, epoch_wer/iter_length, epoch_cer/iter_length

    def start_train(self, train_data, valid_data):
        best_valid_loss = float('inf')
        for epoch in range(self.epoch):
            start = time.time()
            # print('training numbers: ', len(train_iterator))
            
            train_loss, train_accuracy = self.train(train_data)
            valid_loss, valid_accuracy, valid_wer_ocr, valid_cer_ocr, valid_wer, valid_cer = self.evaluate(valid_data)
            end = time.time()
            mins = int((end-start)/60)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_dir)
            print('Epoch: {}  | Time: {} m'.format(epoch+1, mins))
            print('\tTrain Loss: {} | Train Accuracy: {}'.format(train_loss, train_accuracy))
            print('\tVal. Loss: {} |  Val Accuracy: {}'.format(valid_loss, valid_accuracy))
            print('\tVal. WER_OCR: {} |  Val CER_OCR: {}'.format(valid_wer_ocr, valid_cer_ocr))
            print('\tVal. WER_After: {} |  Val CER_After: {}'.format(valid_wer, valid_cer))

            logging.info('Epoch: {}  | Time: {} m'.format(epoch+1, mins))
            logging.info('\tTrain Loss: {} | Train Accuracy: {}'.format(train_loss, train_accuracy))
            logging.info('\tVal. Loss: {} |  Val Accuracy: {}'.format(valid_loss, valid_accuracy))
            logging.info('\tVal. WER_OCR: {} |  Val CER_OCR: {}'.format(valid_wer_ocr, valid_cer_ocr))
            logging.info('\tVal. WER_After: {} |  Val CER_After: {}'.format(valid_wer, valid_cer))

    def load_model(self, model_dir):
        if os.path.isfile(model_dir):
            print('loading model from ', model_dir)
            self.model.load_state_dict(torch.load(model_dir, map_location=self.device))  # not sure if it needs return value
        else:
            print('loading model failed.')

    def prob_to_index(self, out):
        _, predict = torch.max(out, -1)
        # predict = predict.view(-1)
        return predict
    
    def get_prob(self, out):
        prob, _ = torch.max(out, -1)
        return prob

    def test(self, test):
        self.load_model(self.model_dir)
        self.translate(test)

    def test_in_batch(self, test_data):
        self.load_model(self.model_dir)
        loss, accuracy, wer_ocr, cer_ocr, wer_after, cer_after = self.evaluate(test_data)
        print('loss: {} | accuracy: {} | wer_ocr: {} | cer_ocr: {} | wer_after: {} | cer_after: {}'.format(loss, accuracy, wer_ocr, cer_ocr, wer_after, cer_after))

    def Accuracy(self, pred, truth):  # both are squeezed
        num_correct = (pred == truth).sum().item()  # here the prediction after 2 (<eos>) is still 2.
        return float(num_correct)/len(truth.nonzero())

    def Edit_dist_in_batch(self, pred_batch, truth_batch, ws):  # both are in batch, shape: [length, batch]
        # numpy array
        WER = 0
        CER = 0
        pred_batch = pred_batch.cpu().data.numpy()  # convert GPU tensor to cpu
        truth_batch = truth_batch.cpu().data.numpy()
        # print(pred_batch.shape, truth_batch.shape)
        batch = truth_batch.shape[1]
        for i in range(batch):
            pred = pred_batch[:, i]
            truth = truth_batch[:, i]  # todo: clean pad and eos, 0 and 2
            pred = self.remove_func_id(pred)
            truth = self.remove_func_id(truth)
            # print(pred, pred.shape)
            pred = list(pred)
            truth = list(truth)
            WER += statistic.word_error_rate(pred, truth, ws)  
            CER += statistic.char_error_rate(pred, truth)
        return WER/self.batch, CER/self.batch

    def remove_func_id(self, arr):  # remove the functional index e.g., <sos>, <eos>,<pad>
        # exclude = np.concatenate((np.where(arr == 0)[0], np.where(arr == 1)[0]))
        mask_id = np.ones(arr.shape, bool)
        # eos_id = arr.shape[0]
        exclude_eos = np.where(arr == 2)[0]  # find the start of <eos>
        if exclude_eos.any():
            eos_id = exclude_eos[0]
            # mask_id[exclude] = False
            mask_id[eos_id: ] = False
        arr_clean = arr[mask_id]
        # print('remove func id: ', arr_clean)
        return arr_clean

    def translate(self, sent):  # generate correction for sent
        sent_out = ''
        encoder_input = self.DATA.tokens_to_ids(sent)
        encoder_input = torch.from_numpy(pad_data(encoder_input, self.max_length)).view(-1, 1).to(self.device)  # todo: fix padding
        # print('input: ', encoder_input)
        decoder_input = np.zeros(shape=(self.max_length, 1), dtype=np.int64)
        decoder_input[0, :] = dataset.SOS_ID
        decoder_input = torch.from_numpy(decoder_input).to(self.device)
        # print('input shape: {}, output shape: {}'.format(encoder_input.shape, decoder_input.shape))
        output = self.model(encoder_input, decoder_input, 0)  # turn off teacher forcing
        # _, output = torch.max(output, -1)
        output = self.prob_to_index(output).view(-1).tolist()
        # print('output: ', output)
        for i in output[1:]:
            if i == dataset.EOS_ID:
                break
            if i == dataset.UNK_ID:
                continue
            sent_out += self.vocab_reverse[i]
        # print('Correct sentence: ', sent_out)
        return sent_out

    def translate_in_batch(self, sents):  # sentence lists, translate sentences parallel in batch level
        sents_out = []
        probs_out = []
        input_ids = self.DATA.prepare_only_input(sents)
        iterator = iter(dataloader.DataLoader(input_ids, batch_size=self.batch, num_workers=1, shuffle=False, pin_memory=True))
        for batch in iterator:
            # print('batch: ', batch)
            src = batch.permute(1, 0).to(self.device)
            # print('src: ', src)
            number = src.shape[1]   # number of instances
            decoder_input = np.zeros(shape=(self.max_length, number), dtype=np.int64)
            decoder_input[0, :] = dataset.SOS_ID
            decoder_input = torch.from_numpy(decoder_input).to(self.device)
            output = self.model(src, decoder_input, 0)
            output = output[1:]
            predict = self.prob_to_index(output)
            predict = predict.cpu().data.numpy()
            probs = self.prob_in_batch(output)
            sent_b = []
            prob_b = []
            # print(predict.shape, predict)
            for b in range(number):
                out = predict[:, b].tolist()
                probability = probs[:, b].tolist()
                sent = ''
                prob_words = []
                p_w = []
                for i, p in zip(out, probability):
                    if i == dataset.EOS_ID:
                        if p_w:
                            prob_words.append(sum(p_w)/len(p_w))
                            prob_words.append(p_w)
                        break
                    if i == dataset.UNK_ID:   # remove unk token
                        continue
                    token = self.vocab_reverse[i]
                    sent += token
                    if token == ' ': # whitespace, new word
                        p_w.append(p)
                        if p_w:
                            prob_words.append(sum(p_w)/len(p_w))  # average probs of word
                            prob_words.append(p_w)
                        else:
                            pass
                        p_w = []
                    else:
                        p_w.append(p)
                sent_b.append(sent)
                prob_b.append(prob_words)

            sents_out += sent_b
            probs_out += prob_b
        return sents_out, probs_out
    
    def prob_in_batch(self, out):
        probs = self.get_prob(out)
        probs = probs.cpu().data.numpy()
        return probs


if __name__ == '__main__':
    print('hi')





