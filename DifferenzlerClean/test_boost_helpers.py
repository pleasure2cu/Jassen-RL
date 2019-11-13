from unittest import TestCase

import numpy as np

from sample_boosting import boost_list_of_tnr, boost_rnn_part, boost_36_entry_vector, boost_4_players_by_4_suits, \
    boost_basic_prediction_vector


class TestBoost_list_of_tnr(TestCase):
    def test_empty_list(self):
        self.assertTrue(np.array_equal(boost_list_of_tnr(np.array([])), np.array([[], [], [], [], [], []])))

    def test_boost_single_card(self):
        in_vector = np.array([7, 2])
        out_vector = np.array([
            [7, 2],
            [7, 3],
            [7, 1],
            [7, 3],
            [7, 1],
            [7, 2]
        ])
        self.assertTrue(np.array_equal(boost_list_of_tnr(in_vector), out_vector))

    def test_boost_every_suit(self):
        in_vector = np.array([2, 0, 4, 1, 7, 2, 4, 3])
        out_vector = np.array([
            [2, 0, 4, 1, 7, 2, 4, 3],
            [2, 0, 4, 1, 7, 3, 4, 2],
            [2, 0, 4, 2, 7, 1, 4, 3],
            [2, 0, 4, 2, 7, 3, 4, 1],
            [2, 0, 4, 3, 7, 1, 4, 2],
            [2, 0, 4, 3, 7, 2, 4, 1]
        ])
        self.assertTrue(np.array_equal(boost_list_of_tnr(in_vector), out_vector))

    def test_boost_every_suit_with_empty_slots(self):
        in_vector = np.array([2, 0, 4, 2, -1, -1, 7, 1, 4, 3, -1, -1])
        out_vector = np.array([
            [2, 0, 4, 2, -1, -1, 7, 1, 4, 3, -1, -1],
            [2, 0, 4, 3, -1, -1, 7, 1, 4, 2, -1, -1],
            [2, 0, 4, 1, -1, -1, 7, 2, 4, 3, -1, -1],
            [2, 0, 4, 3, -1, -1, 7, 2, 4, 1, -1, -1],
            [2, 0, 4, 1, -1, -1, 7, 3, 4, 2, -1, -1],
            [2, 0, 4, 2, -1, -1, 7, 3, 4, 1, -1, -1]
        ])
        self.assertTrue(np.array_equal(boost_list_of_tnr(in_vector), out_vector))


class TestRnnBoosting(TestCase):
    def test_one(self):
        blie_history = np.array([
            [2, 3, 1, 0, 7, 3, 4, 2, 0],
            [-1, -1, -1, -1, 3, 3, 3, 2, 2]
        ])
        truth = np.array([
            [
                [2, 3, 1, 0, 7, 3, 4, 2, 0],
                [-1, -1, -1, -1, 3, 3, 3, 2, 2]
            ],
            [
                [2, 2, 1, 0, 7, 2, 4, 3, 0],
                [-1, -1, -1, -1, 3, 2, 3, 3, 2]
            ],
            [
                [2, 3, 1, 0, 7, 3, 4, 1, 0],
                [-1, -1, -1, -1, 3, 3, 3, 1, 2]
            ],
            [
                [2, 1, 1, 0, 7, 1, 4, 3, 0],
                [-1, -1, -1, -1, 3, 1, 3, 3, 2]
            ],
            [
                [2, 2, 1, 0, 7, 2, 4, 1, 0],
                [-1, -1, -1, -1, 3, 2, 3, 1, 2]
            ],
            [
                [2, 1, 1, 0, 7, 1, 4, 2, 0],
                [-1, -1, -1, -1, 3, 1, 3, 2, 2]
            ]
        ])
        actual = boost_rnn_part(blie_history)
        self.assertTrue(np.array_equal(truth, actual))


class Test36VectorBooster(TestCase):
    def test_one(self):
        vector = np.arange(36)
        truth = np.array([
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                27, 28, 29, 30, 31, 32, 33, 34, 35
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
            ],
        ])
        actual = boost_36_entry_vector(vector)
        self.assertTrue(np.array_equal(truth, actual))


class Test4By4Tests(TestCase):
    def test_one(self):
        vector = np.arange(16)
        truth = np.array([
            [
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 10, 11,
                12, 13, 14, 15,
            ],
            [
                0, 1, 3, 2,
                4, 5, 7, 6,
                8, 9, 11, 10,
                12, 13, 15, 14,
            ],
            [
                0, 2, 1, 3,
                4, 6, 5, 7,
                8, 10, 9, 11,
                12, 14, 13, 15,
            ],
            [
                0, 2, 3, 1,
                4, 6, 7, 5,
                8, 10, 11, 9,
                12, 14, 15, 13,
            ],
            [
                0, 3, 1, 2,
                4, 7, 5, 6,
                8, 11, 9, 10,
                12, 15, 13, 14,
            ],
            [
                0, 3, 2, 1,
                4, 7, 6, 5,
                8, 11, 10, 9,
                12, 15, 14, 13,
            ],
        ])
        actual = boost_4_players_by_4_suits(vector)
        self.assertTrue(np.array_equal(truth, actual))


class TestPredictionSample(TestCase):
    def test_one(self):
        vector = np.arange(37)
        truth = np.array([
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                36
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                36
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                36
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                36
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                36
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                18, 19, 20, 21, 22, 23, 24, 25, 26,
                9, 10, 11, 12, 13, 14, 15, 16, 17,
                36
            ]
        ])
        actual = boost_basic_prediction_vector(vector)
        self.assertTrue(np.array_equal(truth, actual))
