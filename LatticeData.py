"""
This module provides the lattice information of the triangular lattice


AS : 2-D array
    The real space translation vectors
BS : 2-D array
    The reciprocal space (k space) translation vectors
RS : 2-D array
    The displaces to three nonequivalent nearest neighbors on triangular lattice
"""


import numpy as np


__all__ = [
    "AS",
    "BS",
    "RS",
]


AS = np.array([[1, 0], [0.5, np.sqrt(3) / 2]], dtype=np.float64)
RS = np.array(
    [[1, 0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]], dtype=np.float64
)
BS = 2 * np.pi * np.array(
    [[1, -1 / np.sqrt(3)], [0, 2 / np.sqrt(3)]], dtype=np.float64
)
