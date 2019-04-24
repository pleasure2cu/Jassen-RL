from unittest import TestCase

import numpy as np

from Sample import RnnSample
from helper_functions import translate_vector_to_two_number_representation, get_all_possible_actions, \
    get_winning_card_index, get_points_from_table, two_nbr_rep_swap_suit, two_nbr_rep_table_booster, vector_rep_booster, \
    state_action_83_booster, prediction_state_37_booster, time_series_booster, rnn_sample_booster


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


class TestTwo_nbr_rep_swap_suit(TestCase):
    def test_first(self):
        card = np.array([3, 1])
        calc = two_nbr_rep_swap_suit(card, 1, 3)
        truth = np.array([3, 3])
        self.assertTrue(np.array_equal(calc, truth))

    def test_random(self):
        for _ in range(10000):
            rank = np.random.randint(-1, 9)
            old_suit = np.random.randint(-1, 4)
            card = np.array([rank, old_suit])
            security_copy = np.copy(card)
            s1 = np.random.randint(4)
            s2 = np.random.randint(4)
            if card[1] == s1:
                truth = [rank, s2]
            elif card[1] == s2:
                truth = [rank, s1]
            else:
                truth = [rank, old_suit]
            truth = np.array(truth)
            calc = two_nbr_rep_swap_suit(card, s1, s2)
            self.assertTrue(np.array_equal(calc, truth))  # check for correct behaviour
            self.assertTrue(np.array_equal(card, security_copy))  # check that the input isn't changed


class TestTwo_nbr_rep_table_booster(TestCase):
    def test_random(self):
        for _ in range(10000):
            ranks = np.random.randint(-1, 9, size=4)
            suits = np.random.randint(-1, 4, size=4)
            table = np.zeros(8)
            table[[2 * i for i in range(4)]] = ranks
            table[[2 * i + 1 for i in range(4)]] = suits
            s1 = np.random.randint(4)
            s2 = np.random.randint(4)
            table_copy = np.copy(table)
            truth = []
            for i in range(8):
                if i % 2 == 0:
                    truth.append(table[i])
                else:
                    if table[i] == s1:
                        truth.append(s2)
                    elif table[i] == s2:
                        truth.append(s1)
                    else:
                        truth.append(table[i])
            truth = np.array(truth)
            calc = two_nbr_rep_table_booster(table, s1, s2)
            self.assertTrue(np.array_equal(truth, calc))
            self.assertTrue(np.array_equal(table, table_copy))


class TestVector_rep_booster(TestCase):
    def test_simple(self):
        vector = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1
        ])
        s1 = 1
        s2 = 3
        truth = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0
        ])
        copy = np.copy(vector)
        calc = vector_rep_booster(vector, s1, s2)
        self.assertTrue(np.array_equal(truth, calc))
        self.assertTrue(np.array_equal(vector, copy))

    def test_same_indices(self):
        vector = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1
        ])
        calc = vector_rep_booster(vector, 3, 3)
        self.assertTrue(np.array_equal(vector, calc))


class TestState_action_83_booster(TestCase):
    def test_1(self):
        table = np.array([
            [3, 1],
            [8, 2],
            [-1, -1],
            [4, 0]
        ])
        hand = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1
        ])
        gone_cards = np.array([
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0
        ])
        diff = 18
        action = np.array([2, 2])
        sav1 = np.concatenate([np.reshape(table, -1), hand, gone_cards, np.array([diff]), action])
        sav2 = np.concatenate([
            two_nbr_rep_table_booster(np.reshape(table, -1), 1, 2),
            vector_rep_booster(hand, 1, 2),
            vector_rep_booster(gone_cards, 1, 2),
            np.array([diff]),
            two_nbr_rep_swap_suit(action, 1, 2)
        ])
        sav3 = np.concatenate([
            two_nbr_rep_table_booster(np.reshape(table, -1), 1, 3),
            vector_rep_booster(hand, 1, 3),
            vector_rep_booster(gone_cards, 1, 3),
            np.array([diff]),
            two_nbr_rep_swap_suit(action, 1, 3)
        ])
        sav4 = np.concatenate([
            two_nbr_rep_table_booster(np.reshape(table, -1), 2, 3),
            vector_rep_booster(hand, 2, 3),
            vector_rep_booster(gone_cards, 2, 3),
            np.array([diff]),
            two_nbr_rep_swap_suit(action, 2, 3)
        ])

        table2 = np.array([
            [3, 2],
            [8, 3],
            [-1, -1],
            [4, 0]
        ])
        hand2 = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 1, 1, 0, 0, 0, 0, 0
        ])
        gone_cards2 = np.array([
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 1, 1, 1, 0, 0, 0,
        ])
        action2 = np.array([2, 3])
        sav5 = np.concatenate([np.reshape(table2, -1), hand2, gone_cards2, np.array([diff]), action2])

        table3 = np.array([
            [3, 3],
            [8, 1],
            [-1, -1],
            [4, 0]
        ])
        hand3 = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0
        ])
        gone_cards3 = np.array([
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0
        ])
        action3 = np.array([2, 1])
        sav6 = np.concatenate([np.reshape(table3, -1), hand3, gone_cards3, np.array([diff]), action3])
        calc = state_action_83_booster(sav1)
        truth = [sav1, sav2, sav3, sav4, sav5, sav6]
        self.assertTrue(len(calc) == len(truth))
        for i in range(4):
            self.assertTrue(calc[i].shape == (83,))
            self.assertTrue(np.array_equal(calc[i], truth[i]))
        pass


class TestPrediction_state_37_booster(TestCase):
    def test_1(self):
        original = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            3
        ])
        state1 = np.copy(original)
        state2 = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            3
        ])
        state3 = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            3
        ])
        state4 = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            3
        ])
        state5 = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            3
        ])
        state6 = np.array([
            1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            3
        ])
        truth = [state1, state2, state3, state4, state5, state6]
        calc = prediction_state_37_booster(original)
        self.assertTrue(len(truth) == len(calc), msg=len(calc))
        for i in range(6):
            self.assertTrue(np.array_equal(truth[i], calc[i]), msg=i)


class TestTime_series_booster(TestCase):
    def test_1(self):
        time_series = np.array([[
            [3, 1],
            [8, 2],
            [1, 0],
            [4, 0]
        ],
            [
                [4, 3],
                [7, 2],
                [-1, -1],
                [2, 3]
            ]
        ])
        t1 = np.array([[
            [3, 2],
            [8, 1],
            [1, 0],
            [4, 0]
        ],
            [
                [4, 3],
                [7, 1],
                [-1, -1],
                [2, 3]
            ]
        ])
        t2 = np.array([[
            [3, 3],
            [8, 2],
            [1, 0],
            [4, 0]
        ],
            [
                [4, 1],
                [7, 2],
                [-1, -1],
                [2, 1]
            ]
        ])
        t3 = np.array([[
            [3, 1],
            [8, 3],
            [1, 0],
            [4, 0]
        ],
            [
                [4, 2],
                [7, 3],
                [-1, -1],
                [2, 2]
            ]
        ])
        t5 = np.array([[
            [3, 2],
            [8, 3],
            [1, 0],
            [4, 0]
        ],
            [
                [4, 1],
                [7, 3],
                [-1, -1],
                [2, 1]
            ]
        ])
        t4 = np.array([[
            [3, 3],
            [8, 1],
            [1, 0],
            [4, 0]
        ],
            [
                [4, 2],
                [7, 1],
                [-1, -1],
                [2, 2]
            ]
        ])
        truth = [time_series, t1, t2, t3, t4, t5]
        truth = map(lambda x: np.reshape(x, (2, 8)), truth)
        truth = list(map(lambda x: np.concatenate([x, np.array([[1], [2]])], axis=1), truth))
        calc = time_series_booster(truth[0])
        assert len(calc) == 6, len(calc)
        for i in range(6):
            assert np.array_equal(truth[i], calc[i]), (i, truth[i], calc[i])


class TestRnn_sample_booster(TestCase):
    def test_1(self):
        rnn_part = np.array([
            [3, 2, 8, 3, 1, 0, 4, 0, 1],
            [4, 1, 7, 3, -1, -1, 2, 1, 3]
        ])
        aux_part = np.array([
            0, 1, 0, 0,
            4, 1, 7, 3, -1, -1, 2, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
            -14,
            2, 3
        ])
        seed = RnnSample(rnn_part, aux_part, 75)
        calc = rnn_sample_booster(seed)

        # all parts themselves are tested. The only really plausible issue that could arise, is that the boosted
        # rnn_parts aren't paired up with the correct aux_parts
        for c in calc:
            assert np.array_equal(c.rnn_input[-1][:8], c.aux_input[4:12]), (c.rnn_input[-1][:8], c.aux_input[4:12])
