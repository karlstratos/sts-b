# python frozen.py --model_path scratch/frozen_bert --train --dump_path scratch/dump_bert --verbose --num_runs 100 --num_workers 4 --gpu 4
# python frozen.py --model_path scratch/frozen_glove --train --dump_path scratch/dump_word_embeddings_glove --verbose --num_runs 100 --num_workers 4 --gpu 5

import argparse
import copy
import math
import os
import pickle
import torch
import torch.nn as nn

from collections import OrderedDict
from data import FrozenData
from pytorch_helper.model import Model
from pytorch_helper.util import get_init_uniform, masked_max_from_lengths, \
    masked_mean_from_lengths, FF
from scipy.stats import pearsonr, spearmanr
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FrozenModel(Model):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.padding_value = 0

    def load_data(self):
        self.data = FrozenData(self.hparams.dump_path,
                               padding_value=self.padding_value)

    def define_parameters(self):
        self.lstm1 = nn.LSTM(self.data.dim, self.hparams.dim_lstm,
                             self.hparams.num_layers_lstm,
                             dropout=self.hparams.drop if
                             self.hparams.num_layers_lstm > 1 else 0.,
                             bidirectional=True)
        self.lstm2 = copy.deepcopy(self.lstm1)

        self.scorer = FF(8 * self.hparams.dim_lstm, self.hparams.dim_hidden, 1,
                         self.hparams.num_layers_ff,
                         dropout_rate=self.hparams.drop)
        self.apply(get_init_uniform(self.hparams.init))
        self.loss = torch.nn.MSELoss()

    def forward(self, batch):
        padded_vector_sequences1, padded_vector_sequences2, lengths1, \
            lengths2, scores = [tensor.to(self.device) for tensor in batch]

        pooled1 = self.encode(padded_vector_sequences1, lengths1, self.lstm1)
        pooled2 = self.encode(padded_vector_sequences2, lengths2, self.lstm2)
        embs = torch.cat([pooled1, pooled2, torch.abs(pooled1 - pooled2),
                          pooled1 * pooled2], dim=1)  # B x 4dim_pooled

        preds = self.scorer(embs).squeeze(1)  # B
        loss = self.loss(preds, scores)

        return {'loss': loss, 'preds': preds.tolist(), 'golds': scores.tolist()}

    def encode(self, padded_vector_sequences, lengths, lstm):
        packed = pack_padded_sequence(padded_vector_sequences, lengths,
                                      batch_first=True, enforce_sorted=False)
        output = lstm(packed)[0]
        unpacked = pad_packed_sequence(output, batch_first=True)[0]
        if self.hparams.pooling == 'mean':
            pooled = masked_mean_from_lengths(unpacked, lengths)
        elif self.hparams.pooling == 'max':
            pooled = masked_max_from_lengths(unpacked, lengths)[0]
        else:
            raise ValueError
        return pooled

    def evaluate(self, loader_eval, loader_train=None):
        self.eval()
        preds = []
        golds = []
        with torch.no_grad():
            for batch_num, batch in enumerate(loader_eval):
                forward = self.forward(batch)
                preds.extend(forward['preds'])
                golds.extend(forward['golds'])
        perf = pearsonr(preds, golds)[0] * 100.
        self.train()
        return perf

    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.lr)], []

    @staticmethod
    def get_hparams_grid():
        grid = OrderedDict({
            'dim_lstm': [100, 200, 500, 1000, 1500],
            'dim_hidden': [128, 256, 512],
            'num_layers_lstm': [1, 2],
            'num_layers_ff': [0, 1],
            'pooling': ['mean', 'max'],
            'lr': [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
            'init': [0.5, 0.1, 0.05, 0.01, 0.],
            'drop': [0.5, 0.4, 0.3, 0.2, 0.1, 0.],
            'batch_size': [16, 32, 64, 128],
            'seed': list(range(100000)),
            'verbose': [True],
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Model.get_general_argparser()

        parser.add_argument('--dump_path', type=str, default='scratch/dump',
                            help='path to encoding dump [%(default)s]')
        parser.add_argument('--dim_lstm', type=int, default=100,
                            help='dim LSTM states [%(default)d]')
        parser.add_argument('--dim_hidden', type=int, default=100,
                            help='dim hidden states [%(default)d]')
        parser.add_argument('--num_layers_lstm', type=int, default=1,
                            help='num layers in LSTM[%(default)d]')
        parser.add_argument('--num_layers_ff', type=int, default=1,
                            help='num layers in feedforward [%(default)d]')
        parser.add_argument('--pooling', default='max',
                            choices=['mean', 'max'],
                            help='pooling method [%(default)s]')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='initial learning rate [%(default)g]')
        parser.add_argument('--init', type=float, default=0.1,
                            help='unif init range (default if 0) [%(default)g]')
        parser.add_argument('--drop', type=float, default=0.1,
                            help='dropout rate [%(default)g]')
        parser.add_argument('--clip', type=float, default=1,
                            help='gradient clipping [%(default)g]')

        return parser


if __name__ == '__main__':
    parser = FrozenModel.get_model_specific_argparser()
    hparams = parser.parse_args()

    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.gpu

    model = FrozenModel(hparams)
    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        val_perf, test_perf = model.final_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))
