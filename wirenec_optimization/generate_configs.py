import itertools
import numpy as np
from numpy.random import shuffle, seed


if __name__ == '__main__':
    bandwidth_list = np.arange(500, 1500, 100)
    number_of_freqs = np.arange(3, 15, 2)
    seeds = np.arange(0, 10, 1)

    arrays = [
        bandwidth_list,
        number_of_freqs,
        seeds,
    ]

    seed(42)
    combinations = list(itertools.product(*arrays))
    shuffle(combinations)

    for combination in combinations:
        print(combination)

    print(len(combinations))
