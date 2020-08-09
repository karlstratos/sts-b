import numpy as np
import os
import pickle
import torch

from pytorch_helper.data import Data
from pytorch_helper.util import get_length_mask
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset


class STSData(Data):

    def __init__(self, sts_path, tokenizer, joint):
        super().__init__()
        self.sts_path = sts_path
        self.tokenizer = tokenizer
        self.joint = joint
        self.load_datasets()

    def load_datasets(self):
        self.dataset_train = STSDataset(os.path.join(self.sts_path,
                                                     'original/sts-train.tsv'))
        self.dataset_val = STSDataset(os.path.join(self.sts_path,
                                                   'original/sts-dev.tsv'))
        self.dataset_test = STSDataset(os.path.join(self.sts_path,
                                                    'original/sts-test.tsv'))

    def custom_collate_fn(self, batch):
        sent1s, sent2s, scores = zip(*batch)
        scores = torch.tensor(scores)
        if self.joint:
            encoded = self.tokenizer(sent1s, sent2s, padding=True,
                                     return_tensors='pt',
                                     return_token_type_ids=True)
            return encoded['input_ids'], encoded['token_type_ids'], \
                encoded['attention_mask'], scores
        else:
            encoded1 = self.tokenizer(sent1s, padding=True,
                                      return_tensors='pt')
            encoded2 = self.tokenizer(sent2s, padding=True,
                                      return_tensors='pt')
            return encoded1['input_ids'], encoded2['input_ids'], \
                encoded1['attention_mask'], encoded2['attention_mask'], scores


class STSDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.examples = read_sts_original_file(path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]



def read_sts_original_file(path):  # Ex. 'STS-B/original/sts-dev.tsv'
    examples = []
    with open(path) as f:
        for line in f:
            pieces = line.split('\t')
            sent1 = pieces[5].strip()
            sent2 = pieces[6].strip()
            score = float(pieces[4]) / 5.  # Normalize scores
            assert score >= 0. and score <= 1.
            examples.append((sent1, sent2, score))
    return examples


class FrozenData(Data):

    def __init__(self, dump_path, padding_value=0):
        super().__init__()
        self.dump_path = dump_path
        self.padding_value = padding_value
        self.load_datasets()

    def load_datasets(self):
        encoding = pickle.load(open(self.dump_path, 'rb'))
        self.dim = len(encoding['train'][0][0][0])

        self.dataset_train = FrozenDataset(encoding['train'])
        self.dataset_val = FrozenDataset(encoding['val'])
        self.dataset_test = FrozenDataset(encoding['test'])

    def custom_collate_fn(self, batch):
        vector_sequences1, vector_sequences2, scores = zip(*batch)
        scores = torch.tensor(scores)

        padded_vector_sequences1 = pad_sequence(  # B x T x d
            vector_sequences1, batch_first=True,
            padding_value=self.padding_value)
        padded_vector_sequences2 = pad_sequence(  # B x T' x d
            vector_sequences2, batch_first=True,
            padding_value=self.padding_value)

        lengths1 = torch.LongTensor([len(v) for v in vector_sequences1])  # B
        lengths2 = torch.LongTensor([len(v) for v in vector_sequences2])  # B

        return padded_vector_sequences1, padded_vector_sequences2, lengths1, \
            lengths2, scores


class FrozenDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return torch.tensor(self.examples[index][0]), \
            torch.tensor(self.examples[index][1]), self.examples[index][2]
