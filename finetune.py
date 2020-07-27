import argparse
import math
import torch
import torch.nn as nn

from collections import OrderedDict
from copy import deepcopy
from helper.model import Model
from helper.util import get_init_transformer
from scipy.stats import pearsonr, spearmanr
from sts_data import STSData
from transformers import BertTokenizer, BertModel, RobertaTokenizer, \
    RobertaModel


class FineTuneModel(Model):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def load_data(self):
        self.tokenizer = self.get_tokenizer()
        self.data = STSData(self.hparams.data_path, self.tokenizer,
                            self.hparams.joint)

    def get_tokenizer(self):
        if self.hparams.model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.hparams.model_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        else:
            raise ValueError
        return tokenizer

    def define_parameters(self):
        if self.hparams.model_type == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        elif self.hparams.model_type == 'roberta':
            self.encoder = RobertaModel.from_pretrained('roberta-base')
        else:
            raise ValueError
        self.dim_hidden = self.encoder.config.hidden_size

        if self.hparams.joint:  # Full transformer
            if self.hparams.use_projection:  # [x1|x2] -> score
                self.projector = nn.Linear(self.dim_hidden, 1)
                self.projector.apply(get_init_transformer(self.encoder))
        else:  # Dual encoder
            self.encoder2 = deepcopy(self.encoder)
            if self.hparams.use_projection:  # [x1,x2,x1-x2,x1*x2] -> score
                self.projector = nn.Linear(4 * self.dim_hidden, 1)
                self.projector.apply(get_init_transformer(self.encoder))

        self.loss = torch.nn.MSELoss()

    def forward(self, batch):
        if self.hparams.joint:
            X = batch[0].to(self.device)
            Y = batch[1].to(self.device)
            embs = self.reduce_sequence(self.encoder(X))  # B x d
            if self.hparams.use_projection:
                preds = self.projector(embs).squeeze(1)  # B
            else:
                preds = embs.mean(dim=1)  # B
        else:
            X1 = batch[0].to(self.device)
            X2 = batch[1].to(self.device)
            Y = batch[2].to(self.device)
            embs1 = self.reduce_sequence(self.encoder(X1))  # B x d
            embs2 = self.reduce_sequence(self.encoder(X2))  # B x d
            if self.hparams.use_projection:
                embs = torch.cat([embs1, embs2, embs1 - embs2, embs1 * embs2],
                                 dim=1)  # B x 4d
                preds = self.projector(embs).squeeze(1)  # B
            else:  # Scaled dot product
                scaling = math.sqrt(self.dim_hidden)
                preds = (embs1 * embs2).sum(dim=1).div(scaling)  # B

        loss = self.loss(preds, Y)

        return {'loss': loss, 'preds': preds.tolist(), 'golds': Y.tolist()}

    def reduce_sequence(self, encoder_outputs):
        hiddens, cls = encoder_outputs  # (B x T x d), (B x d)
        if self.hparams.combine == 'cls':  # [CLS] for BERT, <s> for RoBERTa
            embs = cls
        elif self.hparams.combine == 'avg':
            embs = hiddens.mean(dim=1)
        else:
            raise ValueError
        return embs  # B x d

    def evaluate(self, loader_eval, loader_train=None):
        self.eval()
        preds = []
        golds = []
        with torch.no_grad():
            for batch_num, batch in enumerate(loader_eval):
                forward = self.forward(batch)
                preds.extend(forward['preds'])
                golds.extend(forward['golds'])
        perf = pearsonr(preds, golds)[0]
        self.train()
        return perf

    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def configure_optimizers(self):  # TODO: Adam + weight decay + scheduler?
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.lr)], []

    @staticmethod
    def get_hparams_grid():
        # [*] following https://arxiv.org/pdf/1810.04805.pdf
        grid = OrderedDict({
            'batch_size': [16, 32],  # [*]
            'lr': [5e-5, 3e-5, 2e-5],  # [*]
            'epochs': [2, 3, 4],  # [*]
            'joint': [True, False],
            'combine': ['avg', 'cls'],
            'use_projection': [True, False],
            'seed': list(range(100000)),
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Model.get_general_argparser()

        parser.add_argument('--data_path', type=str, default='STS-B',
                            help='path to STS-B folder from GLUE [%(default)s]')
        parser.add_argument('--model_type', type=str, default='bert',
                            choices=['bert', 'roberta'],
                            help='model type [%(default)s]')
        parser.add_argument('--joint', action='store_true',
                            help='joint input?')
        parser.add_argument('--combine', type=str, default='avg',
                            choices=['avg', 'cls'],
                            help='combine method [%(default)s]')
        parser.add_argument('--use_projection', action='store_true',
                            help='use projection?')
        parser.add_argument('--lr', type=float, default=3e-5,
                            help='initial learning rate [%(default)g]')
        parser.add_argument('--clip', type=float, default=1,
                            help='gradient clipping [%(default)g]')

        return parser

if __name__ == '__main__':
    argparser = FineTuneModel.get_model_specific_argparser()
    hparams = argparser.parse_args()
    model = FineTuneModel(hparams)
    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        val_perf, test_perf = model.final_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))
