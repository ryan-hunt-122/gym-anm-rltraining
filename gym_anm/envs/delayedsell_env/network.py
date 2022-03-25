"""The input dictionary for a 2-bus, 4-device distribution network."""

import numpy as np

network = {"baseMVA": 100.0}

network["bus"] = np.array([
    [0, 0, 132, 1., 1.],
    [1, 1, 33, 1.1, 0.9]
])

network["device"] = np.array([
    [0, 0, 4, None, 200, -200, 200, -200, None, None, None, None, None, None, None],
    [1, 1, 2, None, 50, 0, 50, -50, 35, None, 20, -20, None, None, None],
    [2, 1, 3, None, 50, -50, 50, -50, 30, -30, 25, -25, 1000, 0, 0.9]
])

network["branch"] = np.array([
    [0, 1, 0.03, 0.022, 0., 25, 1, 0]
])