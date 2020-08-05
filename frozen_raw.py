# python frozen_raw.py --dump_path scratch/dump_word_embeddings_glove --dim_pca 17

import argparse
import numpy as np
import pickle

from scipy.stats import pearsonr, spearmanr
from util import cos_numpy


class RawFrozenEvaluator:

    def __init__(self, hparams):
        self.hparams = hparams
        self.encoding = pickle.load(open(hparams.dump_path, 'rb'))

        if hparams.dim_pca > 0:
            self.get_val_projection()

    def get_val_projection(self):
        embs = []
        for vectors1, vectors2, _ in self.encoding['val']:
            emb1 = self.get_rep(vectors1)
            emb2 = self.get_rep(vectors2)
            if isinstance(emb1, np.ndarray):
                embs.append(emb1)
            if isinstance(emb2, np.ndarray):
                embs.append(emb2)

        X = np.column_stack(embs)  # d x 2*num_pairs
        self.mu = np.mean(X, axis=1)
        X -= np.expand_dims(self.mu, axis=1)

        print('SVD on %d x %d data matrix, removing top-%d subspace from val '
              'sentence embs' % (X.shape[0], X.shape[1], self.hparams.dim_pca))
        U, S, Vt = np.linalg.svd(X)
        U = U[:,:self.hparams.dim_pca]
        self.P = np.matmul(U, U.transpose())

    def get_rep(self, vectors):
        if not vectors:
            return None
        vectors = [np.array(vector) for vector in vectors]
        if self.hparams.pooling == 'mean':
            rep = np.mean(vectors, axis=0)
        elif self.hparams.pooling == 'max':
            rep = np.amax(np.column_stack(vectors), axis=1)
        else:
            raise ValueError
        return rep

    def run(self):
        p_train, s_train, num_preds_train \
            = self.compute_correlations(self.encoding['train'])
        p_val, s_val, num_preds_val \
            = self.compute_correlations(self.encoding['val'])
        p_test, s_test, num_preds_test \
            = self.compute_correlations(self.encoding['test'])
        return argparse.Namespace(p_train=p_train, s_train=s_train,
                                  num_preds_train=num_preds_train,
                                  p_val=p_val, s_val=s_val,
                                  num_preds_val=num_preds_val,
                                  p_test=p_test, s_test=s_test,
                                  num_preds_test=num_preds_test)

    def compute_correlations(self, examples):
        preds = []
        golds = []
        for vectors1, vectors2, score in examples:
            emb1 = self.get_rep(vectors1)
            emb2 = self.get_rep(vectors2)
            if not (isinstance(emb1, np.ndarray) and
                    isinstance(emb2, np.ndarray)):
                continue  # No tokens with emb

            if self.hparams.dim_pca > 0:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_path', type=str, default='scratch/dump',
                        help='path to encoding dump [%(default)s]')
    parser.add_argument('--dim_pca', type=int, default=0,
                        help='PCA dimension [%(default)d]')
    parser.add_argument('--pooling', default='mean',
                        choices=['mean', 'max'],
                        help='pooling method [%(default)s]')
    hparams = parser.parse_args()

    evaluator = RawFrozenEvaluator(hparams)
    run = evaluator.run()
    print('  train: {:4.1f}/{:4.1f} ({:d}/{:d} evaluated)'.format(
        run.p_train, run.s_train, run.num_preds_train,
        len(evaluator.encoding['train'])))
    print('  val:   {:4.1f}/{:4.1f} ({:d}/{:d} evaluated)'.format(
        run.p_val, run.s_val, run.num_preds_val,
        len(evaluator.encoding['val'])))
    print('  test:  {:4.1f}/{:4.1f} ({:d}/{:d} evaluated)'.format(
        run.p_test, run.s_test, run.num_preds_test,
        len(evaluator.encoding['test'])))
