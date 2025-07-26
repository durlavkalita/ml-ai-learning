import numpy as np

def load_data():
    X = np.array([
        [1, 5],
        [2, 6],
        [3, 5],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 8],
        [9, 9],
        [10, 9]
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y
