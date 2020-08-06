import argparse
import math
import os
import pickle
import torch
import torch.nn as nn

from collections import OrderedDict
from copy import deepcopy
from pytorch_helper.model import Model
from pytorch_helper.util import get_init_transformer
from scipy.stats import pearsonr, spearmanr
from data import STSData
from transformers import BertTokenizer, BertModel, RobertaTokenizer, \
    RobertaModel, AlbertTokenizer, AlbertModel, DistilBertTokenizer, \
    DistilBertModel, ElectraTokenizer, ElectraModel
from util import masked_mean


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
        elif self.hparams.model_type == 'bert-cased':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        elif self.hparams.model_type == 'bert-large':
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        elif self.hparams.model_type == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased')
        elif self.hparams.model_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif self.hparams.model_type == 'roberta-large':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        elif self.hparams.model_type == 'albert':
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        elif self.hparams.model_type == 'albert-xxlarge':
            tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
        elif self.hparams.model_type == 'electra':
            tokenizer = ElectraTokenizer.from_pretrained(
                'google/electra-base-discriminator')
        elif self.hparams.model_type == 'electra-large':
            tokenizer = ElectraTokenizer.from_pretrained(
                'google/electra-large-discriminator')
        else:
            raise ValueError
        return tokenizer

    def get_encoder(self):
        if self.hparams.model_type == 'bert':
            encoder = BertModel.from_pretrained('bert-base-uncased')
        elif self.hparams.model_type == 'bert-cased':
            encoder = BertModel.from_pretrained('bert-base-cased')
        elif self.hparams.model_type == 'bert-large':
            encoder = BertModel.from_pretrained('bert-large-uncased')
        elif self.hparams.model_type == 'distilbert':
            encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        elif self.hparams.model_type == 'roberta':
            encoder = RobertaModel.from_pretrained('roberta-base')
        elif self.hparams.model_type == 'roberta-large':
            encoder = RobertaModel.from_pretrained('roberta-large')
        elif self.hparams.model_type == 'albert':
            encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif self.hparams.model_type == 'albert-xxlarge':
            encoder = AlbertModel.from_pretrained('albert-xxlarge-v2')
        elif self.hparams.model_type == 'electra':
            encoder = ElectraModel.from_pretrained(
                'google/electra-base-discriminator')
        elif self.hparams.model_type == 'electra-large':
            encoder = ElectraModel.from_pretrained(
                'google/electra-large-discriminator')
        else:
            raise ValueError
        return encoder

    def define_parameters(self):
        self.encoder = self.get_encoder()
        self.dim_hidden = self.encoder.config.hidden_size

        if self.hparams.joint:  # Full transformer
            if self.hparams.use_projection:  # [x1|x2] -> score
                self.projector = nn.Sequential(nn.Dropout(self.hparams.drop),
                                               nn.Linear(self.dim_hidden, 1))
                self.projector.apply(get_init_transformer(self.encoder))
        else:  # Dual encoder
            self.encoder2 = deepcopy(self.encoder)
            if self.hparams.use_projection:  # [x1,x2,x1-x2,x1*x2] -> score
                self.projector = nn.Sequential(
                    nn.Dropout(self.hparams.drop),
                    nn.Linear(4 * self.dim_hidden, 1))
                self.projector.apply(get_init_transformer(self.encoder))

        self.loss = torch.nn.MSELoss()

    def forward(self, batch):
        batch = [tensor.to(self.device) for tensor in batch]
        if self.hparams.joint:
            X, L, T, A, Y = batch
            H = self.encoder(X, token_type_ids=T,  # B x T x d
                             attention_mask=A)[0]
            embs = self.reduce_sequence(H, L)  # B x d
            if self.hparams.use_projection:
                preds = self.projector(embs).squeeze(1)  # B
            else:
                preds = embs.mean(dim=1)  # B
        else:
            X1, X2, L1, L2, A1, A2, Y = batch
            H1 = self.encoder(X1, attention_mask=A1)[0]  # B x T x d
            H2 = self.encoder(X2, attention_mask=A2)[0]  # B x T' x d
            embs1 = self.reduce_sequence(H1, L1)  # B x d
            embs2 = self.reduce_sequence(H2, L2)  # B x d
            if self.hparams.use_projection:
                embs = torch.cat([embs1, embs2, embs1 - embs2, embs1 * embs2],
                                 dim=1)  # B x 4d
                preds = self.projector(embs).squeeze(1)  # B
            else:  # Scaled dot product
                scaling = math.sqrt(self.dim_hidden)
                preds = (embs1 * embs2).sum(dim=1).div(scaling)  # B

        loss = self.loss(preds, Y)

        return {'loss': loss, 'preds': preds.tolist(), 'golds': Y.tolist()}


    # TODO: I can just pass attention mask...
    def reduce_sequence(self, hiddens, lengths):  # (B x T x d), (B)
        if self.hparams.pooling == 'cls':  # [CLS] for BERT, <s> for RoBERTa
            embs = hiddens[:,0,:]  # (B x d)
        elif self.hparams.pooling == 'avg':
            embs = masked_mean(hiddens, lengths) # (B x d)
        else:
            raise ValueError
        return embs

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

    def configure_optimizers(self):  # TODO: Adam + weight decay + scheduler?
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.lr)], []

    @staticmethod
    def get_hparams_grid():
        # [*] mostly following https://arxiv.org/pdf/1810.04805.pdf
        grid = OrderedDict({
            'lr': [5e-5, 4e-5, 3e-5, 2e-5],  # [*]
            'epochs': [4],  # [*]
            'joint': [True],
            'pooling': ['avg', 'cls'],
            'use_projection': [True],
            'seed': list(range(100000)),
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Model.get_general_argparser()

        parser.add_argument('--data_path', type=str, default='STS-B',
                            help='path to STS-B folder from GLUE [%(default)s]')
        parser.add_argument('--model_type', type=str, default='bert',
                            choices=['bert', 'bert-cased', 'bert-large',
                                     'distilbert', 'roberta', 'roberta-large',
                                     'albert', 'albert-xxlarge', 'electra',
                                     'electra-large'],
                            help='model type [%(default)s]')
        parser.add_argument('--dump_path', type=str, default='',
                            help='dump encoder output here and exit if '
                            'specified [%(default)s]')
        parser.add_argument('--joint', action='store_true',
                            help='joint input?')
        parser.add_argument('--pooling', type=str, default='avg',
                            choices=['avg', 'cls'],
                            help='pooling method [%(default)s]')
        parser.add_argument('--use_projection', action='store_true',
                            help='use projection?')
        parser.add_argument('--lr', type=float, default=3e-5,
                            help='initial learning rate [%(default)g]')
        parser.add_argument('--drop', type=float, default=0.1,
                            help='dropout rate [%(default)g]')
        parser.add_argument('--clip', type=float, default=1,
                            help='gradient clipping [%(default)g]')

        return parser

    def dump(self, dump_path):
        self.eval()
        encoder = self.get_encoder()
        encoder.to(self.device)

        assert not self.hparams.joint  # Encode sentences separately.
        self.load_data()
        loader_train, loader_val, loader_test \
            = self.data.get_loaders(self.hparams.batch_size,
                                    shuffle_train=False,
                                    num_workers=self.hparams.num_workers,
                                    get_test=True)
        def encode_data(loader):
            with torch.no_grad():
                hiddens1 = []
                hiddens2 = []
                scores = []
                for X1, X2, L1, L2, A1, A2, Y in loader:
                    X1 = X1.to(self.device)
                    X2 = X2.to(self.device)
                    A1 = A1.to(self.device)
                    A2 = A2.to(self.device)

                    # Concern: padding makes output values very slightly
                    # different even with attention masking - probably due to
                    # numerical precision.
                    vectors1 = encoder(X1, attention_mask=A1)[0].tolist()
                    vectors2 = encoder(X2, attention_mask=A2)[0].tolist()

                    for i in range(len(vectors1)):  # Skip padding!
                        hiddens1.append(vectors1[i][:L1[i]])
                        hiddens2.append(vectors2[i][:L2[i]])
                        scores.append(Y[i].item())

            return list(zip(hiddens1, hiddens2, scores))

        encoding = {'train': encode_data(loader_train),
                    'val': encode_data(loader_val),
                    'test': encode_data(loader_test)}
        pickle.dump(encoding, open(dump_path, 'wb'))


if __name__ == '__main__':
    parser = FineTuneModel.get_model_specific_argparser()
    hparams = parser.parse_args()

    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.gpus

    model = FineTuneModel(hparams)
    if hparams.dump_path:
        model.dump(hparams.dump_path)
        exit()

    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        val_perf, test_perf = model.final_test()
        print('Val:  {:8.1f}'.format(val_perf))
        print('Test: {:8.1f}'.format(test_perf))
