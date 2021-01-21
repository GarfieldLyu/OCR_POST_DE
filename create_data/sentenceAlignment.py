#!/usr/bin/env python
# -*- coding: utf-8 -*-
# construct groundtruth sentence pair
import pickle
import re
import parseText
from Bio.pairwise2 import format_alignment
from Bio import pairwise2
from datasketch import MinHash, MinHashLSH, MinHashLSHForest
from nltk import ngrams
from collections import defaultdict
import time
import sys
reload(sys)
sys.setdefaultencoding('utf8')


def read_file(path):
    return file(path).read()


def gen_grams(text, order=10, target=0):
    grams = ngrams(text.split(), order)
    grams = [' '.join(n) for n in grams]
    if target == 0:
        # generate more grams for target for word segmentation problem
        print('generate ngrams for reference')
        for n in range(order+1, order+5):
            gram = ngrams(text.split(), n)
            gram = [' '.join(g) for g in gram ]
            grams += gram
    else:
        print('generate ngrams for target')
    return grams


def target_lsh(grams):
    lsh_forest = MinHashLSHForest(num_perm=4000, l=200)
    lsh = MinHashLSH(threshold=0.5, num_perm=4000)
    # minhashes = {}
    for c, i in enumerate(grams):
        minhash = MinHash(num_perm=4000)
        i = i.replace(' ', '')
        for d in ngrams(i,3):
            minhash.update(''.join(d))

        lsh_forest.add(c,minhash)
        lsh_forest.index()
        lsh.insert(c, minhash)
    return lsh_forest, lsh


def refer_query(lsh_forest, lsh, reference, grams):
    minhash = MinHash(num_perm=4000)
    reference = reference.replace(' ', '')
    for d in ngrams(reference,3):
        minhash.update(''.join(d))

    query_result = lsh_forest.query(minhash, 1)
    query_result_thr = lsh.query(minhash)

    if query_result and query_result_thr:
        result = grams[query_result[0]]
        result_similar = [grams[item] for item in query_result_thr]
        if result in result_similar:
            return result
    else:
        return False


def refine_pair(sent1, sent2):

    def get_alignments(sent1, sent2):
        alignments = pairwise2.align.localms(sent1, sent2, 2, -1, -1, -0.5)
        alignment = pairwise2._clean_alignments(alignments)[0]  # retrieve the first alignment, (sent1, sents)
        return alignment[0], alignment[1]

    def clean_alignments(align1, align2):
        start, end = 0, 0
        if align1.startswith('--'):
            start, end = re.search(r'^-+', align1).span()
        elif align2.startswith('--'):
            start, end = re.search(r'^-+', align2).span()

        align1_clean = align1[end:]  # clean the extra head of align2 or align2
        align2_clean = align2[end:]

        if align1_clean.endswith('--'):
            start, end = re.search(r'-+$', align1_clean).span()
            align2_clean = align2_clean[:start]
        elif align2.endswith('--'):
            start, end = re.search(r'-+$', align2_clean).span()
            align1_clean = align1_clean[:start]
        # remove other - in middle
        align1 = align1_clean.replace('-', '')
        align2 = align2_clean.replace('-', '')
        return align1, align2

    (align1, align2) = get_alignments(sent1, sent2)
    return clean_alignments(align1, align2)


def gen_pairs(target, reference):   # both are a list of n-gram-tokens string, n=5
    lsh_forest, lsh = target_lsh(target)
    pairs = []
    for item in reference:
        result = refer_query(lsh_forest, lsh,  item, target)  # lsh search.
        if result:
            align_1, align_2 = refine_pair(item, result)   # refinement
            pairs.append((align_1, align_2))

    return pairs


def make_pairs_for_book(ocr_book, truth_book, order=5):
    Pairs_book = []
    for page_ocr, page_truth in zip(ocr_book, truth_book):
        page_truth = page_truth.replace('\t', ' ').lstrip().strip()
        page_truth = re.sub(' +', ' ', page_truth)
        target_grams = gen_grams(page_ocr, order=order, target=1)
        reference_grams = gen_grams(page_truth, order=order, target=0)
        pairs = gen_pairs(reference_grams, target_grams)
        # print len(pairs)
        Pairs_book.append(pairs)
    return Pairs_book


def wrapup_lsh_books(barcodes, GT_dir, order=5):   # build sentence pairs for multiple books in barcodes.
    # GT_dir = '/home/lyu/travelogues/GT_books/'    # books under GT_dir
    for barcode in barcodes:
        ocr = file(GT_dir+barcode+'.ocr_clean').read().split('\n\n')[:-1]
        truth = file(GT_dir+barcode+'.truth_clean').read().split('\n\n')[:-1]
        Pairs_book = make_pairs_for_book(ocr, truth, order=order)
        # save pairs of book
        print(sum([len(pair) for pair in Pairs_book]))
        with open(GT_dir + '{}_sentpairs_{}.pickle'.format(barcode, order), 'wb')as f:
            pickle.dump(Pairs_book, f)
            print('successfully saved book {} to pickle.'.format(barcode))


if __name__ == '__main__':

    barcodes = ['Z158515308']
    GT_dir = '/home/lyu/travelogues/GT_books/'
    start = time.time()
    # wrapup_lsh()
    wrapup_lsh_books(barcodes, GT_dir, order=5)
    end = time.time()

    print('It takes ' + str((end-start)/3600) + ' hours.')
