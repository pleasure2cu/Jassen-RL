from unittest import TestCase

import numpy as np

from helper_functions import translate_vector_to_two_number_representation, get_all_possible_actions, \
    get_winning_card_index, get_points_from_table


class TestTranslate_vector_to_two_number_representation(TestCase):
    def test_all_trumps(self):
        vector = np.zeros(36)
        vector[:9] = np.ones(9)
        truth = np.zeros((9, 2))
        truth[:, 0] = np.arange(9)
        self.assertTrue(np.array_equal(truth, translate_vector_to_two_number_representation(vector)))

    def test_all_eight_edge_cases(self):
        vector = np.zeros(36)
        intr_indices = np.array([0, 8, 9, 17, 18, 26, 27, 35])
        vector[intr_indices] = 1
        truth = np.array([
            [0, 0],
            [8, 0],
            [0, 1],
            [8, 1],
            [0, 2],
            [8, 2],
            [0, 3],
            [8, 3]
        ])
        self.assertTrue(np.array_equal(truth, translate_vector_to_two_number_representation(vector)))

    def test_random_test(self):
        for _ in range(50000):
            indices = np.ones(4)
            while len(np.unique(indices)) != 9:
                indices = np.random.randint(36, size=9)
            vector = np.zeros(36)
            vector[indices] = 1
            assert np.sum(vector) == 9
            tn = translate_vector_to_two_number_representation(vector)
            for t in tn:
                vector[t[0] + t[1] * 9] = 0
            self.assertFalse(np.any(vector))


class TestGet_all_possible_actions(TestCase):
    def test_buur_only(self):
        indices = np.array([
            0, 10, 11,
            15, 20, 21,
            25, 30, 31
        ])
        hand = np.zeros(36)
        hand[indices] = 1
        truth = translate_vector_to_two_number_representation(hand)
        self.assertTrue(np.array_equal(truth, get_all_possible_actions(hand, 0)))

    def test_buur_and_one(self):
        indices = np.array([
            0, 8, 11,
            15, 20, 21,
            25, 30, 31
        ])
        hand = np.zeros(36)
        hand[indices[:2]] = 1
        truth = translate_vector_to_two_number_representation(hand)
        self.assertTrue(np.array_equal(truth, get_all_possible_actions(hand, 0)))

    def test_normal_color(self):
        indices = np.array([
            2, 4, 17, 24, 25
        ])
        hand = np.zeros(36)
        hand[indices] = 1
        truth = np.array([
            [2, 0],
            [4, 0],
            [8, 1]
        ])
        test_answer = get_all_possible_actions(hand, 1)
        self.assertTrue(np.array_equal(truth, test_answer))

    def test_normal_color2(self):
        indices = np.array([
            2, 4, 17, 24, 25
        ])
        hand = np.zeros(36)
        hand[indices] = 1
        truth = np.array([
            [2, 0],
            [4, 0],
            [6, 2],
            [7, 2]
        ])
        test_answer = get_all_possible_actions(hand, 2)
        self.assertTrue(np.array_equal(truth, test_answer))

    def test_dont_have_color(self):
        indices = np.array([
            2, 4, 17, 24, 25
        ])
        hand = np.zeros(36)
        hand[indices] = 1
        truth = translate_vector_to_two_number_representation(hand)
        test_answer = get_all_possible_actions(hand, 3)
        self.assertTrue(np.array_equal(truth, test_answer))

    def test_no_suit_has_been_played_yet(self):
        indices = np.array([
            2, 24, 25
        ])
        hand = np.zeros(36)
        hand[indices] = 1
        truth = translate_vector_to_two_number_representation(hand)
        test_answer = get_all_possible_actions(hand, -1)
        self.assertTrue(np.array_equal(truth, test_answer))


class TestGet_winning_card_index(TestCase):
    def test_normal_test(self):
        table = np.array([
            [8, 3],
            [5, 2],
            [7, 2],
            [1, 2]
        ])
        self.assertTrue(get_winning_card_index(table, 0) == 0)
        for i in range(1, 4):
            self.assertTrue(get_winning_card_index(table, i) == 3)

    def test_trump_contained(self):
        table = np.array([
            [8, 3],
            [5, 0],
            [7, 2],
            [1, 2]
        ])
        for i in range(4):
            self.assertTrue(get_winning_card_index(table, i) == 1)

    def test_under_trumped(self):
        table = np.array([
            [8, 3],
            [7, 0],
            [1, 2],
            [5, 0]
        ])
        for i in range(4):
            self.assertTrue(get_winning_card_index(table, i) == 3)


class TestGet_points_from_table(TestCase):
    def test_normal_round(self):
        table = np.array([
            [8, 3],
            [5, 2],
            [7, 2],
            [1, 2]
        ])
        self.assertTrue(get_points_from_table(table, False) == 4)
        self.assertTrue(get_points_from_table(table, True) == 9)

    def test_random(self):
        points = np.array([
            20, 14, 11, 4, 3, 10, 0, 0, 0,
            11, 4, 3, 2, 10, 0, 0, 0, 0,
            11, 4, 3, 2, 10, 0, 0, 0, 0,
            11, 4, 3, 2, 10, 0, 0, 0, 0
        ])
        for i in range(50000):
            indices = np.ones(4)
            while len(np.unique(indices)) != 4:
                indices = np.random.randint(36, size=4)
            vector = np.zeros(36)
            vector[indices] = 1
            table = translate_vector_to_two_number_representation(vector)
            points_on_table = np.sum(points[indices])
            self.assertTrue(get_points_from_table(table, False) == points_on_table)
            self.assertTrue(get_points_from_table(table, True) == points_on_table + 5)
