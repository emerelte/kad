import unittest
from numpy.testing import assert_array_equal
import datetime
import pandas as pd
import numpy as np
import kad_utils

class TestKadUtils(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_embed_data(self):
        input_array = np.array([1, 2, 3, 4, 5])
        steps = 2

        expected_data = np.array([[1., 2.],
                                  [2., 3.],
                                  [3., 4.]])
        expected_labels = np.array([3., 4., 5.])

        data, labels = kad_utils.embed_data(input_array, steps)

        assert_array_equal(expected_data, data)
        assert_array_equal(expected_labels, labels)
