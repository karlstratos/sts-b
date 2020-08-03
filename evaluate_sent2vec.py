# This script assumes that sent2vec is installed, see:
# https://github.com/epfml/sent2vec/tree/74ae0313aca7145df6baa3e61f6a6314fdc72b7a
#
# python evaluate_sent2vec.py --model ../sent2vec/wiki_unigrams.bin
# python evaluate_sent2vec.py --model ../sent2vec/twitter_unigrams.bin

import argparse
import numpy as np
import os
import re
import sent2vec

from nltk import TweetTokenizer, word_tokenize
from scipy.stats import pearsonr, spearmanr
from sts_data import read_sts_original_file as read_sts
from util import cos_numpy


class Sent2vecEvaluator:

    def __init__(self, hparams):
        self.hparams = hparams
        self.load_sts_data()
        self.define_prepare_text()
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(hparams.model)

        if hparams.dim_subspace > 0:
            self.get_projection()

    def get_projection(self):
        examples = self.examples_train if self.hparams.project_train else \
                   self.examples_val
        embs = []
        for sent1, sent2, score in examples:
            text1 = self.prepare_text(sent1)
            text2 = self.prepare_text(sent2)
            emb1 = np.squeeze(self.model.embed_sentence(text1))
            emb2 = np.squeeze(self.model.embed_sentence(text2))
            embs.append(emb1)
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
        p_train, s_train = self.evaluate(self.examples_train)
        p_val, s_val = self.evaluate(self.examples_val)
        p_test, s_test = self.evaluate(self.examples_test)
        return argparse.Namespace(p_train=p_train, s_train=s_train,
                                  p_val=p_val, s_val=s_val,
                                  p_test=p_test, s_test=s_test)

    def evaluate(self, examples):
        preds = []
        golds = []
        for sent1, sent2, score in examples:
            text1 = self.prepare_text(sent1)
            text2 = self.prepare_text(sent2)
            emb1 = np.squeeze(self.model.embed_sentence(text1))
            emb2 = np.squeeze(self.model.embed_sentence(text2))

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
        p = pearsonr(preds, golds)[0] * 100.
        s = spearmanr(preds, golds)[0] * 100.
        return p, s

    def define_prepare_text(self):
        model_type = os.path.basename(self.hparams.model).split('_')[0]
        if model_type == 'twitter':
            self.tokenizer = TweetTokenizer()
            def prepare_text(text):
                tokens = self.tokenizer.tokenize(text)
                return self.preprocess_tweet(' '.join(tokens))  # lowercased
        elif model_type == 'wiki':
            def prepare_text(text):
                return self.preprocess_wiki(text)  # lowercased
        else:
            raise ValueError

        self.prepare_text = prepare_text

    def load_sts_data(self):
        self.examples_train = read_sts('STS-B/original/sts-train.tsv')
        self.examples_val = read_sts('STS-B/original/sts-dev.tsv')
        self.examples_test = read_sts('STS-B/original/sts-test.tsv')

    @staticmethod
    # Copied from sent2vec repository (tweetTokenize.py)
    def preprocess_tweet(tweet):
        tweet = tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',
                       '<url>', tweet)
        tweet = re.sub('(\@[^\s]+)','<user>',tweet)
        try:
            tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
        except:
            pass
        return tweet

    @staticmethod
    # Copied and modified from sent2vec repository (wikiTokenize.py)
    def preprocess_wiki(sentence, to_lower=True):
        """Arguments:
            - sentence: a string to be tokenized
            - to_lower: lowercasing or not
        """
        def format_token(token):
            """"""
            if token == '-LRB-':
                token = '('
            elif token == '-RRB-':
                token = ')'
            elif token == '-RSB-':
                token = ']'
            elif token == '-LSB-':
                token = '['
            elif token == '-LCB-':
                token = '{'
            elif token == '-RCB-':
                token = '}'
            return token

        sentence = sentence.strip()
        sentence = ' '.join([format_token(x) for x in word_tokenize(sentence)])
        if to_lower:
            sentence = sentence.lower()
        sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',
                          '<url>',sentence) #replace urls by <url>
        sentence = re.sub('(\@ [^\s]+)','<user>',
                          sentence) #replace @user268 by <user>
        filter(lambda word: ' ' not in word, sentence)
        return sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,  # (or wiki_unigrams.bin)
                        default='../sent2vec/twitter_unigrams.bin',
                        help='sent2vec model [%(default)s]')
    parser.add_argument('--project_train', action='store_true',
                        help='get projection from training data?')
    parser.add_argument('--pca', action='store_true',
                        help='do PCA instead of just best-fit subspace?')
    parser.add_argument('--dim_subspace', type=int, default=0,
                        help='best-fit subspace dimension [%(default)d]')
    hparams = parser.parse_args()

    evaluator = Sent2vecEvaluator(hparams)
    print('Vocab size %d, embedding dimension %d' %
          (len(evaluator.model.get_vocabulary()),
           evaluator.model.get_emb_size()))

    run = evaluator.run()
    print('  train: {:2.1f}/{:2.1f}'.format(run.p_train, run.s_train))
    print('  val:   {:2.1f}/{:2.1f}'.format(run.p_val, run.s_val))
    print('  test:  {:2.1f}/{:2.1f}'.format(run.p_test, run.s_test))
