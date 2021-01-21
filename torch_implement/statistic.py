import codecs
import nltk


def char_error_rate(rec_sent, ref_sent):
    return nltk.edit_distance(rec_sent, ref_sent)/float(len(ref_sent))

def word_error_rate(rec_sent, ref_sent, ws):  # sent: list of ids, whitespace index

    rec_sent = [str(e) for e in rec_sent]
    ref_sent = [str(e) for e in ref_sent]
    rec_words = ''.join(rec_sent).split(str(ws))
    ref_words = ''.join(ref_sent).split(str(ws))
    return nltk.edit_distance(rec_words, ref_words)/float(len(ref_words))


