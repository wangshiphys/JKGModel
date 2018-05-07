"""
This module defines the commonly used sigma matrices
"""

import numpy as np


# sigma matrices
SIGMA0 = np.array([[1, 0], [0, 1]], dtype=np.int64)
SIGMAX = np.array([[0, 1], [1, 0]], dtype=np.int64)
SIGMAY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMAZ = np.array([[1, 0], [0, -1]], dtype=np.int64)
SIGMAS = np.array([SIGMA0, SIGMAX, SIGMAY, SIGMAZ])

SIGMA0.setflags(write=False)
SIGMAX.setflags(write=False)
SIGMAY.setflags(write=False)
SIGMAZ.setflags(write=False)
SIGMAS.setflags(write=False)

# sigmay * 1j
ISIGMAY = np.array([[0, 1], [-1, 0]], dtype=np.int64)
ISIGMAY.setflags(write=False)

# pairing matrices
PMS = np.matmul(ISIGMAY, SIGMAS) / np.sqrt(2)
PMS.setflags(write=False)


if __name__ == "__main__":
    print("SIGMA0:")
    print(SIGMA0)
    print()

    print("SIGMAX:")
    print(SIGMAX)
    print()

    print("SIGMAY:")
    print(SIGMAY)
    print()

    print("SIGMAZ:")
    print(SIGMAZ)
    print()

    print("SIGMAS:")
    print(SIGMAS)
    print()

    print("ISIGMAY:")
    print(ISIGMAY)
    print()

    print("PMS:")
    print(PMS)
    print()
