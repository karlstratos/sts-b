# python evaluate_word_embeddings.py ../../data/word_representations/glove.840B.300d.txt

import argparse
import numpy as np
import os

from collections import Counter
from nltk import word_tokenize
from scipy.stats import pearsonr, spearmanr
from util import cos_numpy


class WordEmbeddingEvaluator:

    def __init__(self, hparams):
        self.hparams = hparams
        self.load_sts_data()
        self.load_word_embeddings()

    def run(self):
        p_train, s_train, num_preds_train \
            = self.compute_correlations(self.examples_train)
        p_val, s_val, num_preds_val \
            = self.compute_correlations(self.examples_val)
        p_test, s_test, num_preds_test \
            = self.compute_correlations(self.examples_test)
        return argparse.Namespace(p_train=p_train, s_train=s_train,
                                  num_preds_train=num_preds_train,
                                  p_val=p_val, s_val=s_val,
                                  num_preds_val=num_preds_val,
                                  p_test=p_test, s_test=s_test,
                                  num_preds_test=num_preds_test)

    def compute_correlations(self, examples):
        preds = []
        golds = []
        for toks1, toks2, score in examples:
            emb1 = self.get_rep(toks1)
            emb2 = self.get_rep(toks2)
            if not (isinstance(emb1, np.ndarray) and
                    isinstance(emb2, np.ndarray)):
                continue  # No tokens with emb
            preds.append(cos_numpy(emb1, emb2))
            golds.append(score)
        preds = np.array(preds)
        golds = np.array(golds)
        p = pearsonr(preds, golds)[0] * 100. if len(preds) > 0 else None
        s = spearmanr(preds, golds)[0] * 100. if len(preds) > 0 else None
        return p, s, len(preds)

    def get_rep(self, toks):
        vectors = [self.wemb[tok] for tok in toks if tok in self.wemb]
        if not vectors:
            return None
        if self.hparams.pooling == 'mean':
            rep = np.mean(vectors, axis=0)
        elif self.hparams.pooling == 'max':
            rep = np.amax(np.column_stack(vectors), axis=1)
        else:
            raise ValueError
        return rep

    def load_sts_data(self):
        self.vocab = Counter()

        def load_sts_file(path):
            examples = []
            with open(path) as f:
                for line in f:
                    pieces = line.split('\t')
                    toks1 = word_tokenize(pieces[5].strip())
                    toks2 = word_tokenize(pieces[6].strip())
                    if self.hparams.lowercase:
                        toks1 = [tok.lower() for tok in toks1]
                        toks2 = [tok.lower() for tok in toks2]
                    score = float(pieces[4])
                    examples.append((toks1, toks2, score))
                    self.vocab.update(toks1)
                    self.vocab.update(toks2)
            return examples

        self.examples_train = load_sts_file('STS-B/original/sts-train.tsv')
        self.examples_val = load_sts_file('STS-B/original/sts-dev.tsv')
        self.examples_test = load_sts_file('STS-B/original/sts-test.tsv')

    def load_word_embeddings(self):
        self.wemb = {}
        with open(self.hparams.word_embeddings) as f:
            for i, line in enumerate(f):
                if self.hparams.ignore_line1 and i == 0:
                    continue

                word = line[:100].split()[0]  # Partial split to get word
                if self.hparams.lowercase:
                    word = word.lower()
                if not word in self.vocab:
                    continue

                toks = line.split()  # Full split
                try:
                    self.wemb[word] = np.array([float(tok) for tok in
                                                toks[1:]])
                except ValueError:
                    print('Skipping weird line: %s...' % line[:100])
                    continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('word_embeddings', type=str,
                        help='word embeddings file')
    parser.add_argument('--lowercase', action='store_true',
                        help='lowercase?')
    parser.add_argument('--ignore_line1', action='store_true',
                        help="ignore the first line in the embeddings file?")
    parser.add_argument('--pooling', default='mean',
                        choices=['mean', 'max'],
                        help='pooling for word embeddings [%(default)s]')
    hparams = parser.parse_args()

    evaluator = WordEmbeddingEvaluator(hparams)
    print('%d/%d words in vocab have embeddings' % (len(evaluator.wemb),
                                                    len(evaluator.vocab)))

    run = evaluator.run()
    print('  train: {:4.1f}/{:4.1f} ({:d}/{:d} evaluated)'.format(
        run.p_train, run.s_train, run.num_preds_train,
        len(evaluator.examples_train)))
    print('  val:   {:4.1f}/{:4.1f} ({:d}/{:d} evaluated)'.format(
        run.p_val, run.s_val, run.num_preds_val,
        len(evaluator.examples_val)))
    print('  test:  {:4.1f}/{:4.1f} ({:d}/{:d} evaluated)'.format(
        run.p_test, run.s_test, run.num_preds_test,
        len(evaluator.examples_test)))
