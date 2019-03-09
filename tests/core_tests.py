import os
import sys
sys.path.append('..')
import unittest

from itertools import product

from nn.layers import Linear


class Test(unittest.TestCase):

    def test_linear(self):
        for in_dim, out_dim, n_batch in product(range(1, 10), range(1, 10), range(1, 10)):
            linear = Linear(in_dim, out_dim)
            x = np.random.rand(n_batch, in_dim)
            y = linear(x)
            self.assertEqual(y.shape, (in_dim, out_dim))
