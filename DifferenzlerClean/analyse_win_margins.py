import numpy as np


def main():
    a = np.load('win_margins_hinton_net_0_discount_30_dropout_player_100000.npy')
    v, c = np.unique(a, return_counts=True)
    c_norm = c / np.sum(c)
    for i, value in enumerate(c_norm[:20]):
        print(i, '\t', round(value, 3))
    print("hello")


if __name__ == '__main__':
    main()
