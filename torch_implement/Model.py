# -*- coding: utf-8 -*-
"""encoder-decoder model implemented by pytorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
try:
    import entmax
except:
    os.system('pip3 install entmax --user')

from entmax import sparsemax
from entmax.losses import SparsemaxLoss

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

''' cuda multiprocessing'''
# from multiprocessing import set_start_method
# try:
    # set_start_method('spawn')
# except RuntimeError:
    # pass


class Encoder(nn.Module):
    def __init__(self, inp_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.inp_dim = inp_dim  # the same as input features/vocab size
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.embedding = nn.Embedding(inp_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, int(enc_hid_dim/2), bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src): # lengths of src instances
        # max_length = src.shape[0]
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded) 
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, sparse_max):
        super(Attention, self).__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn = nn.Linear( enc_dim + dec_dim, dec_dim)
        self.v = nn.Parameter(torch.rand(dec_dim))
        self.sparse_max = sparse_max

    def forward(self, hidden, enc_out):
        batch_size = enc_out.shape[1]
        src_len = enc_out.shape[0]
        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        enc_out = enc_out.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_out), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch size, 1, dec hid dim]
        attention = torch.bmm(v, energy).squeeze(1)
        # attention= [batch size, src len]
        if self.sparse_max:
            score = sparsemax(attention, dim=1)
        else:
            score = F.softmax(attention, dim=1)
        return score


class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.out_dim = out_dim
        # self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim)
        self.out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, enc_out):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))  # embedded = [1, batch size, emb dim]
        a = self.attention(hidden, enc_out)
        a = a.unsqueeze(1)  # a = [batch size, 1, src len]
        enc_out = enc_out.permute(1, 0, 2)
        weighted = torch.bmm(a, enc_out)  # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)  # weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)  # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output = self.out(torch.cat((output, weighted, embedded), dim=1))  # output = [bsz, output dim]
        return output, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, tfr=1.0):  # teacher forcing ratio
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = src.shape[1]
        # print('test batch size', batch_size)
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.out_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        output = trg[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < tfr
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
        return outputs

