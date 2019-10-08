from unittest import TestCase

import numpy as np

from player import boost_color_strat_sample, boost_color_pred_sample


class TestBoosting(TestCase):
    def test_boost_color_sample(self):
        blie_history = np.array([
            [2, 3, 1, 0, 7, 3, 4, 2, 0],
            [-1, -1, -1, -1, 3, 3, 3, 2, 2]
        ])
        aux_history = np.array([
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1,
            -1, -1, -1, -1, 3, 3, 3, 2,
            35, 8, 3
        ])

        blie_history_1_3_2 = np.array([
            [2, 2, 1, 0, 7, 2, 4, 3, 0],
            [-1, -1, -1, -1, 3, 2, 3, 3, 2]
        ])
        aux_history_1_3_2 = np.array([
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 1, 0, 1, 1, 0, 0, 0,
            -1, -1, -1, -1, 3, 2, 3, 3,
            35, 8, 2
        ])

        blie_history_2_1_3 = np.array([
            [2, 3, 1, 0, 7, 3, 4, 1, 0],
            [-1, -1, -1, -1, 3, 3, 3, 1, 2]
        ])
        aux_history_2_1_3 = np.array([
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 1,
            -1, -1, -1, -1, 3, 3, 3, 1,
            35, 8, 3
        ])

        blie_history_2_3_1 = np.array([
            [2, 1, 1, 0, 7, 1, 4, 3, 0],
            [-1, -1, -1, -1, 3, 1, 3, 3, 2]
        ])
        aux_history_2_3_1 = np.array([
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 1, 0, 0, 0, 1,
            -1, -1, -1, -1, 3, 1, 3, 3,
            35, 8, 1
        ])

        blie_history_3_1_2 = np.array([
            [2, 2, 1, 0, 7, 2, 4, 1, 0],
            [-1, -1, -1, -1, 3, 2, 3, 1, 2]
        ])
        aux_history_3_1_2 = np.array([
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 1, 0, 1, 1, 0, 0, 0,
            -1, -1, -1, -1, 3, 2, 3, 1,
            35, 8, 2
        ])

        blie_history_3_2_1 = np.array([
            [2, 1, 1, 0, 7, 1, 4, 2, 0],
            [-1, -1, -1, -1, 3, 1, 3, 2, 2]
        ])
        aux_history_3_2_1 = np.array([
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1,
            -1, -1, -1, -1, 3, 1, 3, 2,
            35, 8, 1
        ])

        blie = [
            blie_history, blie_history_1_3_2, blie_history_2_1_3, blie_history_2_3_1, blie_history_3_1_2,
            blie_history_3_2_1
        ]
        aux = [
            aux_history, aux_history_1_3_2, aux_history_2_1_3, aux_history_2_3_1, aux_history_3_1_2, aux_history_3_2_1
        ]
        truth = list(zip(blie, aux))
        actual = boost_color_strat_sample(blie_history, aux_history)
        for i in range(6):
            truth_rnn, truth_aux = truth[i]
            actual_rnn, actual_aux = actual[i]
            self.assertTrue(np.array_equal(truth_rnn, actual_rnn), msg="rnn, i = {}".format(i))
            self.assertTrue(np.array_equal(truth_aux, actual_aux), msg="aux, i = {}".format(i))

    def test_boost_color_pred_sample(self):

        ground_vector = np.array([
            1, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            3
        ])

        ground_vector_1_3_2 = np.array([
            1, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            3
        ])

        ground_vector_2_1_3 = np.array([
            1, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            3
        ])

        ground_vector_2_3_1 = np.array([
            1, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0,
            3
        ])

        ground_vector_3_1_2 = np.array([
            1, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            3
        ])

        ground_vector_3_2_1 = np.array([
            1, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0,
            3
        ])

        truth = [
            ground_vector, ground_vector_1_3_2, ground_vector_2_1_3, ground_vector_2_3_1, ground_vector_3_1_2, ground_vector_3_2_1
        ]
        actual = boost_color_pred_sample(ground_vector)
        for i in range(6):
            self.assertTrue(np.array_equal(truth[i], actual[i]), msg="i = {}".format(i))
