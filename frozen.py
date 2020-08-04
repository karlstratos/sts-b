# python frozen.py --train --dump_path scratch/dump
# python frozen.py --train --model_path scratch/frozen_bert --dump_path scratch/dump_bert --num_runs 100  --gpus 6 --num_workers 8
# python frozen.py --train --model_path scratch/frozen_roberta --dump_path scratch/dump_roberta --num_runs 100  --gpus 7 --num_workers 8

import argparse
import math
import os
import pickle
import torch
import torch.nn as nn

from collections import OrderedDict
from data import FrozenData
from pytorch_helper.model import Model
from pytorch_helper.util import get_init_uniform
from scipy.stats import pearsonr, spearmanr


class FrozenModel(Model):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def load_data(self):
        self.data = FrozenData(self.hparams.dump_path)

    def define_parameters(self):
        self.linear = nn.Sequential(nn.Dropout(self.hparams.drop),
                                    nn.Linear(self.data.dim_hidden,
                                              self.data.dim_hidden))
        self.apply(get_init_uniform(self.hparams.init))
        self.loss = torch.nn.MSELoss()

    def forward(self, batch):
        X1 = batch[0].to(self.device)  # B x T x d
        X2 = batch[1].to(self.device)  # B x T' x d
        Y = batch[2].to(self.device)
        embs1 = X1.mean(dim=1)  # B x d
        embs2 = X2.mean(dim=1)  # B x d
        preds = (self.linear(embs1) * embs2).sum(dim=1)
        loss = self.loss(preds, Y)
        return {'loss': loss, 'preds': preds.tolist(), 'golds': Y.tolist()}

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
            'batch_size': [32],
            'lr': [5e-5, 4e-5, 3e-5, 2e-5],
            'drop': [0.5, 0.4, 0.3, 0.2, 0.1, 0.],
            'init': [0.5, 0.1, 0.05, 0.01, 0.],
            'seed': list(range(100000)),
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Model.get_general_argparser()

        parser.add_argument('--dump_path', type=str, default='scratch/dump',
                            help='path to encoding dump [%(default)s]')
        parser.add_argument('--lr', type=float, default=3e-5,
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
    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.gpus

    model = FrozenModel(hparams)
    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        val_perf, test_perf = model.final_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))
