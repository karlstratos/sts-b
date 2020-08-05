import numpy as np
import os
import pickle
import torch

from pytorch_helper.data import Data
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
        # Get original files so that we can have test labels.
        path_train = os.path.join(self.sts_path, 'original/sts-train.tsv')
        path_val = os.path.join(self.sts_path, 'original/sts-dev.tsv')
        path_test = os.path.join(self.sts_path, 'original/sts-test.tsv')

        self.dataset_train = STSDataset(path_train, self.tokenizer, self.joint)
        self.dataset_val = STSDataset(path_val, self.tokenizer, self.joint)
        self.dataset_test = STSDataset(path_test, self.tokenizer, self.joint)

    def custom_collate_fn(self, batch):
        if self.joint:
            xs, ys = zip(*batch)
            return (pad_sequence(xs, batch_first=True,  # B x T
                                 padding_value=self.tokenizer.pad_token_id),
                    torch.cat(ys, dim=0))  # B
        else:
            x1s, x2s, ys = zip(*batch)
            return (pad_sequence(x1s, batch_first=True,  # B x T
                                 padding_value=self.tokenizer.pad_token_id),
                    pad_sequence(x2s, batch_first=True,  # B x T'
                                 padding_value=self.tokenizer.pad_token_id),
                    torch.cat(ys, dim=0))  # B


class STSDataset(Dataset):
    def __init__(self, path, tokenizer, joint):
        self.path = path
        self.tokenizer = tokenizer
        self.joint = joint
        self.examples = read_sts_original_file(path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        sent1, sent2, score = self.examples[index]
        if self.joint:
            x = self.tokenizer.encode_plus(sent1, sent2)['input_ids']
            return torch.tensor(x), torch.tensor([score])
        else:
            x1 = self.tokenizer.encode_plus(sent1)['input_ids']
            x2 = self.tokenizer.encode_plus(sent2)['input_ids']
            return torch.tensor(x1), torch.tensor(x2), torch.tensor([score])


class FrozenData(Data):

    def __init__(self, dump_path, padding_value=0):
        super().__init__()
        self.dump_path = dump_path
        self.padding_value = padding_value
        self.load_datasets()

    def load_datasets(self):
        encoding = pickle.load(open(self.dump_path, 'rb'))
        self.dim_hidden = len(encoding['train'][0][0][0])

        self.dataset_train = FrozenDataset(encoding['train'])
        self.dataset_val = FrozenDataset(encoding['val'])
        self.dataset_test = FrozenDataset(encoding['test'])

    def custom_collate_fn(self, batch):
        x1s, x2s, ys = zip(*batch)
        return (pad_sequence(x1s, batch_first=True,  # B x T
                             padding_value=self.padding_value),
                pad_sequence(x2s, batch_first=True,  # B x T'
                             padding_value=self.padding_value),
                torch.cat(ys, dim=0))  # B


class FrozenDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples[0])

    def __getitem__(self, index):
        x1, x2, score = self.examples[index]
        return torch.tensor(x1), torch.tensor(x2), torch.tensor([score])


def read_sts_original_file(path):  # Ex. 'STS-B/original/sts-dev.tsv'
    examples = []
    with open(path) as f:
        for line in f:
            pieces = line.split('\t')
            sent1 = pieces[5].strip()
            sent2 = pieces[6].strip()
            score = float(pieces[4])
            examples.append((sent1, sent2, score))
    return examples
