# python evaluate_word_embeddings.py ../../data/word_representations/glove.840B.300d.txt
# python evaluate_word_embeddings.py ../../data/word_representations/wiki.en.vec --lowercase --ignore_line1
# python evaluate_word_embeddings.py ../../data/word_representations/glove.840B.300d.txt --dim_subspace 17 --freq ../../data/rcv1/vocab_rcv1.txt  --pca

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

        if hparams.freq:
            self.get_word_prob()

        if hparams.dim_subspace > 0:
            self.get_projection()

    def get_word_prob(self):
        self.word_prob = Counter()
        for line in open(self.hparams.freq):
            word, count = line.split()
            if self.hparams.lowercase:
                word = word.lower()
            self.word_prob[word] += int(count)

        total = sum(list(self.word_prob.values()))
        for word in self.word_prob:
            self.word_prob[word] /= total

        self.word_prob_mean = sum(list(self.word_prob.values())) \
                              / len(self.word_prob)

    def get_projection(self):
        examples = self.examples_train if self.hparams.project_train else \
                   self.examples_val
        embs = []
        for (toks1, toks2, _) in examples:
            emb1 = self.get_rep(toks1)
            if isinstance(emb1, np.ndarray):
                embs.append(emb1)
            emb2 = self.get_rep(toks2)
            if isinstance(emb2, np.ndarray):
                embs.append(emb2)

        X = np.column_stack(embs)  # d x 2*num_pairs

        if self.hparams.pca:
            self.mu = np.mean(X, axis=1)
            X -= np.expand_dims(self.mu, axis=1)

        print('SVD on %d x %d data matrix, removing top-%d subspace from %s '
              'sentence embs' %
              (X.shape[0], X.shape[1], self.hparams.dim_subspace,
               'train' if self.hparams.project_train else 'val'))
        U, S, Vt = np.linalg.svd(X)
        U = U[:,:self.hparams.dim_subspace]
        self.P = np.matmul(U, U.transpose())

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

            if self.hparams.dim_subspace > 0:
                if self.hparams.pca:
                    emb1 -= self.mu
                    emb2 -= self.mu
                emb1 -= self.P.dot(emb1)
                emb2 -= self.P.dot(emb2)

            preds.append(cos_numpy(emb1, emb2))
            golds.append(score)
        preds = np.array(preds)
        golds = np.array(golds)
        p = pearsonr(preds, golds)[0] * 100. if len(preds) > 0 else None
        s = spearmanr(preds, golds)[0] * 100. if len(preds) > 0 else None
        return p, s, len(preds)

    def weight(self, tok):
        if not self.hparams.freq:
            w = 1
        else:
            p = self.word_prob[tok] if tok in self.word_prob else \
                self.word_prob_mean
            w = self.hparams.smoothing / (p + self.hparams.smoothing)
        return w

    def get_rep(self, toks):
        vectors = [self.weight(tok) * self.wemb[tok] for tok in toks
                   if tok in self.wemb]
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
                    self.dim = len(self.wemb[word])
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
                        help='ignore the first line in the embeddings file?')
    parser.add_argument('--project_train', action='store_true',
                        help='get projection from training data?')
    parser.add_argument('--pca', action='store_true',
                        help='do PCA instead of just best-fit subspace?')
    parser.add_argument('--dim_subspace', type=int, default=0,
                        help='best-fit subspace dimension [%(default)d]')
    parser.add_argument('--freq', default=None,
                        help='path to word frequency file')
    parser.add_argument('--smoothing', type=float, default=0.001,
                        help='smoothing value [%(default)d]')
    parser.add_argument('--pooling', default='mean',
                        choices=['mean', 'max'],
                        help='pooling for word embeddings [%(default)s]')
    hparams = parser.parse_args()

    evaluator = WordEmbeddingEvaluator(hparams)
    print('%d/%d words in vocab have embeddings (dim %d)' % (
        len(evaluator.wemb), len(evaluator.vocab), evaluator.dim))

    if hparams.freq:
        print('%d/%d words covered in word frequency file' % (
            len([word for word in evaluator.vocab if word in
                 evaluator.word_prob]), len(evaluator.vocab)))

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
