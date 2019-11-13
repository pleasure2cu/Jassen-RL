from typing import Tuple, List

import numpy as np

swaps = [
    [0, 1, 2, 3, -1],
    [0, 1, 3, 2, -1],
    [0, 2, 1, 3, -1],
    [0, 2, 3, 1, -1],
    [0, 3, 1, 2, -1],
    [0, 3, 2, 1, -1],
]

swaps_np = np.array(swaps)


def boost_list_of_tnr(vector: np.ndarray) -> np.ndarray:
    out = np.tile(vector, (6, 1))
    for i, swap in enumerate(swaps_np):
        for j in range(1, len(vector), 2):
            out[i, j] = swap[int(out[i, j])]
    return out


def boost_rnn_part(rnn_tensor: np.ndarray) -> np.ndarray:
    parts = np.array([boost_list_of_tnr(blie[:8]) for blie in rnn_tensor])
    cards_boosted = np.transpose(parts, (1, 0, 2))
    out = np.concatenate(
        [cards_boosted, np.tile(rnn_tensor[:, 8:].reshape((-1, 1)), (6, 1, 1))],
        axis=2
    )
    return out


def boost_36_entry_vector(vector: np.ndarray) -> np.ndarray:
    snippets_matrix = vector.reshape((4, 9))
    out = np.tile(snippets_matrix, (6, 1, 1))
    for i, swap in enumerate(swaps_np[:, :4]):
        out[i] = out[i][swap]
    return out.reshape((6, 36))


def boost_4_players_by_4_suits(vector: np.ndarray) -> np.ndarray:
    sqr_matrix = vector.reshape((4, 4))
    out = np.tile(sqr_matrix, (6, 1, 1))
    for i, swap in enumerate(swaps_np[:, :4]):
        for j in range(4):
            out[i, j] = out[i, j][swap]
    return out.reshape((6, 16))


def boost_basic_prediction_vector(state_vector: np.ndarray) -> np.ndarray:
    """ boosts a state vector by its suits. The vector should be a 36-entry hand vector and the players position """
    boosted_hand_vector = boost_36_entry_vector(state_vector[:36])
    return np.concatenate(
        [boosted_hand_vector, np.tile(state_vector[36:], (6, 1))],
        axis=1
    )


def boost_hand_crafted_strategy_vector(rnn_input: np.ndarray, aux_input: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    boosted_rnn_part = boost_rnn_part(rnn_input)
    boosted_hand_vector = boost_36_entry_vector(aux_input[:36])
    boosted_rel_table = boost_list_of_tnr(aux_input[36:44])
    boosted_current_diff = np.tile(aux_input[44: 45], (6, 1))
    boosted_gone_cards = boost_36_entry_vector(aux_input[45: 81])
    boosted_bocks = boost_36_entry_vector(aux_input[81: 117])
    boosted_could_follow = boost_4_players_by_4_suits(aux_input[117: 133])
    boosted_points_table = np.tile(aux_input[133: 134], (6, 1))
    boosted_made_points = np.tile(aux_input[134: 138], (6, 1))
    boosted_action = boost_list_of_tnr(aux_input[138: 140])
    boosted_aux_part = np.concatenate(
        [boosted_hand_vector, boosted_rel_table, boosted_current_diff, boosted_gone_cards, boosted_bocks,
         boosted_could_follow, boosted_points_table, boosted_made_points, boosted_action],
        axis=1
    )
    return [(l, r) for l, r in zip(list(boosted_rnn_part), list(boosted_aux_part))]
