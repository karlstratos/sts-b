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

from nltk import TweetTokenizer
from nltk import word_tokenize
from scipy.stats import pearsonr, spearmanr
from sts_data import read_sts_original_file
from util import cos_numpy


def evaluate(examples, prepare_text, model):
    preds = []
    golds = []
    for sent1, sent2, score in examples[:100]:
        text1 = prepare_text(sent1)
        text2 = prepare_text(sent2)
        emb1 = np.squeeze(model.embed_sentence(text1))
        emb2 = np.squeeze(model.embed_sentence(text2))
        pred = cos_numpy(emb1, emb2)
        preds.append(pred)
        golds.append(score)
    preds = np.array(preds)
    golds = np.array(golds)
    p = pearsonr(preds, golds)[0] * 100.
    s = spearmanr(preds, golds)[0] * 100.
    return p, s


def main(hparams):
    examples_train = read_sts_original_file('STS-B/original/sts-train.tsv')
    examples_val = read_sts_original_file('STS-B/original/sts-dev.tsv')
    examples_test = read_sts_original_file('STS-B/original/sts-test.tsv')

    model_type = os.path.basename(hparams.model).split('_')[0]
    if model_type == 'twitter':
        tokenizer = TweetTokenizer()
        def prepare_text(text):
            tokens = tokenizer.tokenize(text)
            return preprocess_tweet(' '.join(tokens))  # lowercased
    elif model_type == 'wiki':
        def prepare_text(text):
            return preprocess_wiki(text)  # lowercased
    else:
        raise ValueError

    model = sent2vec.Sent2vecModel()
    model.load_model(hparams.model)
    print('Vocab size %d, embedding dimension %d' %
          (len(model.get_vocabulary()), model.get_emb_size()))

    p_train, s_train = evaluate(examples_train, prepare_text, model)
    p_val, s_val = evaluate(examples_val, prepare_text, model)
    p_test, s_test = evaluate(examples_test, prepare_text, model)
    print('  train: {:2.1f}/{:2.1f}'.format(p_train, s_train))
    print('  val:   {:2.1f}/{:2.1f}'.format(p_val, s_val))
    print('  test:  {:2.1f}/{:2.1f}'.format(p_test, s_test))


# Copied from sent2vec repository (tweetTokenize.py)
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',
                   tweet)
    tweet = re.sub('(\@[^\s]+)','<user>',tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    return tweet


# Copied from sent2vec repository (wikiTokenize.py)
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


# Copied and modified from sent2vec repository (wikiTokenize.py)
def preprocess_wiki(sentence, to_lower=True):
    """Arguments:
        - sentence: a string to be tokenized
        - to_lower: lowercasing or not
    """
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
    hparams = parser.parse_args()

    main(hparams)
