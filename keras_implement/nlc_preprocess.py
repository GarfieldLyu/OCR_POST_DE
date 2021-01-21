import os
import pickle
from nltk import ngrams

_PAD = "<pad>"
_SOS = "<sos>"
_EOS = "<eos>"
_UNK = "<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def get_data(path):
    with open(path, 'rb')as f:
        sentence_pair = pickle.load(f, encoding='utf8', errors='ignore')
    return sentence_pair


def char_tokenize(sentence):
    return list(sentence)


def ngram_tokenize(sentence, order=3):
    # tokenize sentence to sub-words sequence
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


def create_vocab(data_pairs, vocab_path, tokenizer, max_vocab_size=20000):
    if not os.path.isfile(vocab_path):
        print('creating vocabulary {} from data ...'.format(vocab_path))
        vocab = {}
        for (inp, out) in data_pairs:
            tokens = tokenizer(inp) + tokenizer(out)
            for w in tokens:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print('vocabulary size: ' + str(len(vocab_list)))
        if len(vocab_list) > max_vocab_size:
            vocab_list = vocab_list[:max_vocab_size]
        vocab_dict = dict([(y, x) for (x, y) in enumerate(vocab_list)])
        vocab_reverse = dict([(x, y) for (x, y) in enumerate(vocab_list)])
        with open(vocab_path, 'wb')as f:
            pickle.dump(vocab_dict, f)
    else:
        with open(vocab_path, 'rb')as f:
            vocab_dict = pickle.load(f)
        vocab_reverse = {}
        for x, y in vocab_dict.items():
            vocab_reverse[y] = x

    return vocab_dict, vocab_reverse


def sentence_to_token_ids(sentence, vocab, tokenizer):
    tokens = tokenizer(sentence)
    return [SOS_ID] + [vocab.get(w, UNK_ID) for w in tokens]+[EOS_ID]


def data_to_token_ids(pairs, vocab, tokenizer):
    data_x, data_y = [], []
    for (inp, out) in pairs:
        ocr = sentence_to_token_ids(inp, vocab, tokenizer)
        clean = sentence_to_token_ids(out, vocab, tokenizer)
        data_x.append(ocr)
        data_y.append(clean)
    return data_x, data_y


def id_to_token(ids, vocab_path):
    with open(vocab_path, 'rb')as f:
        vocab = pickle.load(f)
    vocab_reverse = dict([(y, x) for (x, y) in vocab.items()])
    return ''.join(vocab_reverse[i] for i in ids)


def prepare_nlc_data(data_path, vocab_path, tokenizer, max_vocab_size):
    data_pairs = get_data(path=data_path)
    vocab, vocab_reverse = create_vocab(data_pairs, vocab_path, tokenizer, max_vocab_size=max_vocab_size)
    vocab_size = len(vocab)
    data_x, data_y = data_to_token_ids(data_pairs, vocab, tokenizer)
    max_length = max([len(y) for y in data_y])
    return data_x, data_y, max_length, vocab_size, vocab, vocab_reverse

