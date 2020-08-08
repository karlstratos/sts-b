import util as util
import torch
import unittest


class TestUtil(unittest.TestCase):

    def test_masked_max(self):
        hiddens = torch.tensor([[[3, 3, 3, 3], [2, 2, 2, 7], [8, 8, 8, 8]],
                                [[1, 1, 17, 1], [9, 9, 9, 9], [11, 1, 11, 11]]])
        lengths = torch.LongTensor([2, 3])
        output = util.masked_max_from_lengths(hiddens, lengths)[0]
        self.assertListEqual(output.tolist(), [[3, 3, 3, 7],
                                               [11, 9, 17, 11]])

    def test_masked_mean(self):
        hiddens = torch.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                                [[100, 100, 100, 100], [9, 9, 9, 9],
                                 [11, 11, 11, 11]]])
        lengths = torch.LongTensor([2, 3])
        output = util.masked_mean_from_lengths(hiddens, lengths)
        self.assertListEqual(output.tolist(), [[1.5, 1.5, 1.5, 1.5],
                                               [40, 40, 40, 40]])

    def test_get_length_mask(self):
        mask = util.get_length_mask(torch.LongTensor([3, 1, 4]), 6)
        self.assertListEqual(mask.tolist(), [[1, 1, 1, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0],
                                             [1, 1, 1, 1, 0, 0]])

        mask = util.get_length_mask(torch.LongTensor([3, 1, 4]), 6, flip=True)
        self.assertListEqual(mask.tolist(), [[0, 0, 0, 1, 1, 1],
                                             [0, 1, 1, 1, 1, 1],
                                             [0, 0, 0, 0, 1, 1]])


if __name__ == '__main__':
    unittest.main()
