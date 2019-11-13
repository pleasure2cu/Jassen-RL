from unittest import TestCase

import numpy as np

from helpers import which_are_bock
from state import GameState

complete_card_history = [
    np.array([7, 2]), np.array([6, 2]), np.array([8, 2]), np.array([8, 3]),
    np.array([5, 1]), np.array([8, 1]), np.array([6, 1]), np.array([2, 1]),
    np.array([2, 2]), np.array([3, 2]), np.array([5, 2]), np.array([8, 0]),
    np.array([6, 0]), np.array([4, 0]), np.array([3, 0]), np.array([1, 0]),
    np.array([7, 3]), np.array([2, 3]), np.array([7, 0]), np.array([3, 3]),
    np.array([1, 2]), np.array([3, 1]), np.array([0, 2]), np.array([7, 1]),
    np.array([6, 3]), np.array([1, 3]), np.array([4, 3]), np.array([0, 3]),
    np.array([4, 1]), np.array([5, 3]), np.array([1, 1]), np.array([4, 2]),
    np.array([0, 1]), np.array([0, 0]), np.array([5, 0]), np.array([2, 0]),
]

blie_starters = [0, 1, 0, 3, 2, 0, 2, 1, 3]


def play_further_cards(state: GameState, offset: int, goal: int):
    for i in range(offset, goal):
        if i % 4 == 0:
            state.current_blie_index = i // 4
            state.set_starting_player_of_blie(blie_starters[i // 4])
        state.add_card(complete_card_history[i], (blie_starters[i // 4] + i % 4) % 4)


class TestGoneCards(TestCase):
    def test_all_cards_are_gone(self):
        state = GameState()
        play_further_cards(state, 0, 36)
        self.assertTrue(np.all(state.gone_cards == 1), msg=state.gone_cards)

    def test_partial_cards_are_gone(self):
        state = GameState()
        play_further_cards(state, 0, 13)
        gone_cards = np.array([
            0, 0, 0, 0, 0, 0, 1, 0, 1,
            0, 0, 1, 0, 0, 1, 1, 0, 1,
            0, 0, 1, 1, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 1
        ], dtype=np.bool)
        self.assertTrue(np.array_equal(gone_cards, state.gone_cards))


class TestWhichAreBock(TestCase):
    def test_no_bock_trivial(self):
        state = GameState()
        hand = np.array([
            0, 0, 1, 0, 0, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 1, 0, 0, 1,
            0, 0, 1, 1, 0, 0, 0, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ])
        bocks = which_are_bock(state.gone_cards, hand)
        self.assertTrue(not np.any(bocks))

    def test_one_single_bock(self):
        state = GameState()
        play_further_cards(state, 0, 29)
        hand = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0,
        ])
        real_blocks = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0,
        ])
        bocks = which_are_bock(state.gone_cards, hand)
        self.assertTrue(np.array_equal(real_blocks, bocks), msg=bocks)

    def test_multiple_single_bocks(self):
        state = GameState()
        play_further_cards(state, 0, 29)
        hand = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 0, 0
        ])
        real_blocks = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 0, 0,
        ])
        bocks = which_are_bock(state.gone_cards, hand)
        self.assertTrue(np.array_equal(real_blocks, bocks), msg=bocks)


class TestWhoCouldFollow(TestCase):
    def test_could_follow_all(self):
        state = GameState()
        play_further_cards(state, 0, 3)
        could_follow_truth = np.ones(16)
        actual = state.get_could_follow_vector()
        self.assertTrue(np.array_equal(could_follow_truth, actual), msg=actual)

    def test_could_follow_not_all(self):
        state = GameState()
        play_further_cards(state, 0, 4)
        could_follow_truth = np.array([
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 0,
            1, 1, 1, 1,
        ])
        actual = state.get_could_follow_vector()
        self.assertTrue(np.array_equal(could_follow_truth, actual), msg=actual)

    def test_trump_not_fire(self):
        state = GameState()
        play_further_cards(state, 0, 20)
        could_follow_truth = np.array([
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 0,
            1, 1, 1, 1,
        ])
        actual = state.get_could_follow_vector()
        self.assertTrue(np.array_equal(could_follow_truth, actual), msg=actual)

    def test_after_32(self):
        state = GameState()
        play_further_cards(state, 0, 32)
        could_follow_truth = np.array([
            1, 1, 1, 1,
            0, 1, 0, 1,
            1, 0, 1, 0,
            1, 1, 1, 1,
        ])
        actual = state.get_could_follow_vector()
        self.assertTrue(np.array_equal(could_follow_truth, actual), msg=actual)

    def test_after_32_one_by_one(self):
        state = GameState()
        for i in range(31):
            play_further_cards(state, i, i+1)
        could_follow_truth = np.array([
            1, 1, 1, 1,
            0, 1, 0, 1,
            1, 0, 1, 0,
            1, 1, 1, 1,
        ])
        actual = state.get_could_follow_vector()
        self.assertTrue(np.array_equal(could_follow_truth, actual), msg=actual)
