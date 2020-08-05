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
            xs, ts, ys = zip(*batch)
            X = pad_sequence(xs, batch_first=True,  # B x T
                             padding_value=self.tokenizer.pad_token_id)
            L = torch.LongTensor([len(x) for x in xs])  # B
            T = torch.zeros(X.size())  # B x T
            A = torch.zeros(X.size())  # B x T
            for i in range(X.size(0)):
                T[i][:L[i]] = ts[i]  # Type ID of trailing padding assumed zero
                A[i][:L[i]] = torch.ones(L[i])
            Y = torch.cat(ys, dim=0)  # B
            return X, L, T.long(), A, Y

        else:
            x1s, x2s, ys = zip(*batch)
            X1 = pad_sequence(x1s, batch_first=True,  # B x T
                              padding_value=self.tokenizer.pad_token_id)
            X2 = pad_sequence(x2s, batch_first=True,  # B x T'
                              padding_value=self.tokenizer.pad_token_id)
            L1 = torch.LongTensor([len(x) for x in x1s])  # B
            L2 = torch.LongTensor([len(x) for x in x2s])  # B
            A1 = torch.zeros(X1.size())  # B x T
            A2 = torch.zeros(X2.size())  # B x T'
            for i in range(X1.size(0)):
                A1[i][:L1[i]] = torch.ones(L1[i])
                A2[i][:L2[i]] = torch.ones(L2[i])
            Y = torch.cat(ys, dim=0)  # B
            return X1, X2, L1, L2, A1, A2, Y


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
        y = torch.tensor([score])
        if self.joint:
            encoded_dict = self.tokenizer.encode_plus(
                sent1, sent2, return_token_type_ids=True)
            input_ids = torch.tensor(encoded_dict['input_ids'])
            token_type_ids = torch.tensor(encoded_dict['token_type_ids'])
            return input_ids, token_type_ids, y
        else:
            encoded_dict1 = self.tokenizer.encode_plus(sent1)
            encoded_dict2 = self.tokenizer.encode_plus(sent2)
            input_ids1 = torch.tensor(encoded_dict1['input_ids'])
            input_ids2 = torch.tensor(encoded_dict2['input_ids'])
            return input_ids1, input_ids2, y


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
        h1s, h2s, ys = zip(*batch)
        H1 = pad_sequence(h1s, batch_first=True,  # B x T x d
                          padding_value=self.padding_value)
        H2 = pad_sequence(h2s, batch_first=True,  # B x T' x d
                          padding_value=self.padding_value)
        L1 = torch.LongTensor([len(h) for h in h1s])  # B
        L2 = torch.LongTensor([len(h) for h in h2s])  # B
        A1 = torch.zeros((H1.size(0), H1.size(1)))  # B x T
        A2 = torch.zeros((H2.size(0), H2.size(1)))  # B x T'
        for i in range(H1.size(0)):
            A1[i][:L1[i]] = torch.ones(L1[i])
            A2[i][:L2[i]] = torch.ones(L2[i])
        Y = torch.cat(ys, dim=0)  # B
        return H1, H2, L1, L2, A1, A2, Y


class FrozenDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        h1, h2, score = self.examples[index]
        h1 = torch.tensor(h1) # d x T
        h2 = torch.tensor(h2) # d x T'
        y = torch.tensor([score])
        return h1, h2, y


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
