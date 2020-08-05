import numpy as np
import torch


def cos_numpy(u, v):
    return np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)


def get_length_mask(batch_size, max_length, lengths):
    indices = torch.arange(max_length).expand(batch_size, -1)
    lengths = lengths.unsqueeze(1).expand(batch_size, max_length)
    mask = indices < lengths
    return mask  # B x T


# TODO: attention mask version
def masked_mean(hiddens, lengths):  # (B x T x d), (B)
    B, T, d = hiddens.size()
    mask = get_length_mask(B, T, lengths)
    mask = mask.unsqueeze(2).expand(B, T, d)
    mean = (mask * hiddens).sum(dim=1).true_divide(lengths.unsqueeze(1))
    return mean  # B x T


# TODO: move to pytorch_helper/util.py and unittest
#H = torch.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
#                  [[100, 100, 100, 100], [9, 9, 9, 9], [11, 11, 11, 11]]])
#lengths = torch.LongTensor([2, 3])

#expect = torch.tensor([[1.5, 1.5, 1.5, 1.5],
#                       [40, 40, 40, 40]])

#A = masked_mean(H, lengths)
#print(H)
#print(H.size())
#print(expect)
#print(A)
#exit()


#mask = get_length_mask(3, 10, torch.LongTensor([7, 10, 3]))
#print(mask)
