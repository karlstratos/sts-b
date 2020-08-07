import util as util
import torch
import unittest


class TestUtil(unittest.TestCase):

    def test_masked_mean(self):
        hiddens = torch.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                                [[100, 100, 100, 100], [9, 9, 9, 9],
                                 [11, 11, 11, 11]]])
        mask = torch.LongTensor([[1, 1, 0],
                                 [1, 1, 1]])
        output = util.masked_mean(hiddens, mask)
        self.assertListEqual(output.tolist(), [[1.5, 1.5, 1.5, 1.5],
                                               [40, 40, 40, 40]])

    def test_get_length_mask(self):
        mask = util.get_length_mask(torch.LongTensor([3, 1, 4]), 6)
        self.assertListEqual(mask.tolist(), [[1, 1, 1, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0],
                                             [1, 1, 1, 1, 0, 0]])


if __name__ == '__main__':
    unittest.main()
