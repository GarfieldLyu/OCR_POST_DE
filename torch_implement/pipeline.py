# -*- coding: utf-8 -*-
import seq2seq
import dataset
import pickle
import torch
import re
import time
import os
import dataScrapy

punc = '!"#$%&\'()*+/,.-:;<=>?@[\\]^_`{|}~â\x80\x93â\x80\x99\x9c\x9e»…！«“–Äº'
data_dir = '/home/lyu/OCR-post/OCR_posthoc/data/all_books.PKL'
model_dir = '/home/lyu/OCR-post/OCR_posthoc/Model/all_books0.model'
barcodes_dir = '/home/lyu/OCR-post/OCR_posthoc/data/updated_list/Barcodes_list_18_new.txt'
corpus_dir = '/home/lyu/OCR-post/OCR_posthoc/data/travelogues_corpus/18c-books/books/'
output_dir = '/home/lyu/OCR-post/OCR_posthoc/data/travelogues_corpus/18c-books/corrected/'
tokenizer = 'char'
features = 100000
batch = 32
embed = 128
enc = 256
dropout = 0.3
epoch = 10
clip = 0.1
sparse = 1
tf = 0.5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_ocr_model():
    data_dir = '/home/lyu/OCR-post/OCR_posthoc/data/all_books.PKL'
    model_dir = '/home/lyu/OCR-post/OCR_posthoc/Model/all_books0.model'
    with open(data_dir, 'rb')as f:
        Train, _ = pickle.load(f)  # Train, Test
    Train_data = Train[0] # one group with 12 books
    tokenizer = 'char'
    features = 100000  # maximum vocabulary size
    DATA = dataset.LoadData(tokenizer, features)
    DATA.build_vocab_on_the_fly(Train_data) # initialize vocabs
    # initialize hyperparameters for model
    batch = 8
    inp_dim = out_dim = len(DATA.vocab)
    embed = 128
    enc = dec = 256
    dropout = 0.3
    epoch = 10
    clip = 0.1
    sparse = 1
    tf = 0.5  # teacher forcing ratio, 0.5 by default, only used for training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    OCR = seq2seq.Train(inp_dim, out_dim, embed, enc, dec, dropout, dropout, epoch, clip, sparse, tf, DATA, batch, device, model_dir)
    return OCR


def read_barcodes(barcodes_dir):
    with open(barcodes_dir, 'r')as f:
        barcodes = f.read()
    barcodes = barcodes.split('\n')
    barcodes = [b for b in barcodes if b]
    print(len(barcodes))
    # print(barcodes[90])
    return barcodes

  
def read_book(barcode, corpus_dir):
    book_dir = os.path.join(corpus_dir, barcode+'.txt')
    if os.path.isfile(book_dir):
        with open(book_dir, 'r')as f:
            book = f.read()
    else:
        print('re download book.')
        book = dataScrapy.getContentForManifest(corpus_dir, barcode)
    return book


def count_char(s):
    flag = 0
    for i in s:
        if i.isalpha():
            flag += 1
    return flag


def clean_page(page):
    page = page.replace('-\n', '').replace('.', '\n').replace(',', '\n').replace('/', '\n')
    for s in punc:
        page = page.replace(s, '')
    page = re.sub('\n+', '\n', page)
    page = re.sub(' +', ' ', page).strip().lstrip()
    return page


def sentenize(page, max_length=90):
    sents = page.split('\n')
    sents_valid = []
    for s in sents:
        # s = s.replace('Ä', '').replace('º', '')
        if len(s) > max_length:
            words = s.split(' ')
            num = int(len(words) / 2)
            s_l = ' '.join(words[:num]).strip().lstrip()
            s_r = ' '.join(words[num:]).strip().lstrip()
            if s_l:
                sents_valid.append(s_l)
            if s_r:
                sents_valid.append(s_r)
        else:
            s = s.strip().lstrip()
            if s:
                sents_valid.append(s)
    return sents_valid


def clean_book(book):  # clean raw book, return a list of pages with valid content (>50 characters), remove punc
    book_valid = []
    for page in book:
        if count_char(page) > 50:
            page = clean_page(page)
            book_valid.append(page)
    return book_valid


def has_digit(s):
    for char in s:
        if char.isdigit():
            return 1
    return 0


def is_short(s):
    num_w = len(s.split())
    avg_char = len(s)/num_w
    if avg_char <= 3 and num_w <= 3:
        return 1
    return 0


def digits_prob(s):
    i = 0
    for char in s:
        if char.isdigit():
            i += 1
    prob = i/len(s)
    return prob


def fix_digits(s):  # apply on word
    after_split = []
    prob = digits_prob(s)
    if prob == 1.0:
        after_split.append(s)
    elif prob >= 0.5:
        if 'o' in s:
            # print('o exists.')
            s = re.sub('o', '0', s) # 14o8 -> 1408
        s = re.sub('[a-z]', '', s)
        after_split.append(s)
    else:
        after_split = split_char_digit(s)
        # after_split += s
    return after_split


def split_char_digit(s):
    after = [a for a in re.split('([0-9]+)', s) if a != '']
    return after


def translate_page(sents_page, OCR):  # combine neural model and manual rules
    digit_sents = []
    digit_sents_ids = []  # treat seperately since the models can't correct numbers
    char_sents = []
    char_sents_ids = []  # translate in batch for efficiency
    short_sents = []
    short_sents_ids = []
    for index, sent in enumerate(sents_page):
        if is_short(sent):
            short_sents.append(' '.join([a for w in sent.split() for a in fix_digits(w)]))
            short_sents_ids.append(index)
        else:
            if has_digit(sent):  # there is at least 1 digit in this long sentence.
                # sent = split_char_digit(sent)
                words = sent.split()
                new_sent = []
                for j, word in enumerate(words):
                    if has_digit(word):
                        after_digit = fix_digits(word)
                        for after in after_digit:
                            if not has_digit(after):
                                # print(after)
                                new_sent.append( OCR.translate(after) )
                            else:
                                new_sent.append(after)
                    else:
                        if is_short(word):
                            new_sent.append(word)
                        else:
                            subsent, _ = OCR.translate_in_batch([word])  # check if it's short
                            new_sent += subsent
                digit_sents.append(' '.join(new_sent))
                digit_sents_ids.append(index)

            else:
                char_sents.append(sent)
                char_sents_ids.append(index)

    # recover the order from the index
    sents_order = list.copy(sents_page)
    for i, sent in zip(short_sents_ids, short_sents):
        sents_order[i] = sent
    for i, sent in zip(digit_sents_ids, digit_sents):
        sents_order[i] = sent
    if char_sents:
        sents_batch, probs_batch = OCR.translate_in_batch(char_sents)
        for i, sent in zip(char_sents_ids, sents_batch):
            sents_order[i] = sent
    else:
        pass

    return sents_order


def pipeline(barcode, OCR):
    # OCR = load_ocr_model()
    # barcodes = read_barcodes(barcodes_dir)
    # barcode = barcodes[0]
    print('start book {}...'.format(barcode))
    START = time.time()
    book = read_book(barcode, corpus_dir)
    if book:
        book = book.split('\n##########\n')  # split to pages.
        # print(len(book))
        book_valid = clean_book(book)
        for i, page in enumerate(book_valid):
            sents = sentenize(page)
            sents_after = translate_page(sents, OCR)
            # write the page
            current_out_dir = os.path.join(output_dir, barcode)
            if not os.path.exists(current_out_dir):
                os.mkdir(current_out_dir)
            name_ocr = os.path.join(current_out_dir, 'page_{}_ocr.txt'.format(i))
            name_trans = os.path.join(current_out_dir, 'page_{}_post.txt'.format(i))
            with open(name_ocr, 'w')as f:
                for sent in sents:
                    f.write(sent + '\n')
            with open(name_trans, 'w')as f:
                for s in sents_after:
                    f.write(s + '\n')
        AFTER = time.time()
        mins = (AFTER - START)/60
        print('\nIt takes {} minutes for book {}.'.format(mins, barcode))
    else:
        pass


def pipeline_corpus():
    OCR = load_ocr_model()
    barcodes = read_barcodes(barcodes_dir)
    for i, barcode in enumerate(barcodes):
        print('start book {}'.format(i))
        if os.path.exists(os.path.join(output_dir, barcode)):
            continue
        else:
            print('add missed book.')
            pipeline(barcode, OCR)


def main():
    pipeline_corpus()


if __name__ == '__main__':
    start = time.time()
    main()
    # read_barcodes(barcodes_dir)
    after = time.time()
    print('It took {} mins'.format((after-start)/60))
