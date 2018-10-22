"""
This module defines the commonly used sigma matrices
"""


import numpy as np


# sigma matrices
SIGMA0 = np.array([[1, 0], [0, 1]], dtype=np.float64)
SIGMAX = np.array([[0, 1], [1, 0]], dtype=np.float64)
SIGMAY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMAZ = np.array([[1, 0], [0, -1]], dtype=np.float64)
SIGMAS = np.array([SIGMA0, SIGMAX, SIGMAY, SIGMAZ])

# sigmay * 1j
ISIGMAY = np.array([[0, 1], [-1, 0]], dtype=np.float64)

# pairing matrices
PMS = np.matmul(ISIGMAY, SIGMAS) / np.sqrt(2)
