import numpy as np
from unittest import TestCase

from PlayerInterlayer import get_table_roll_array


class TestGet_table_roll_array(TestCase):

    def test_0(self):
        truth = np.arange(8)
        calc = get_table_roll_array(0)
        self.assertTrue(np.array_equal(truth, calc))

    def test_1(self):
        truth = [6, 7, 0, 1, 2, 3, 4, 5]
        calc = get_table_roll_array(1)
        self.assertTrue(np.array_equal(truth, calc))

    def test_2(self):
        truth = [4, 5, 6, 7, 0, 1, 2, 3]
        calc = get_table_roll_array(2)
        self.assertTrue(np.array_equal(truth, calc))

    def test_3(self):
        truth = [2, 3, 4, 5, 6, 7, 0, 1]
        calc = get_table_roll_array(3)
        self.assertTrue(np.array_equal(truth, calc))
