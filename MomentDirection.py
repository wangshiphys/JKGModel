import logging
from time import time

import numpy as np
from HamiltonianPy import Lattice
from numba import complex128, jit


@jit(complex128[:](complex128[:], complex128[:]), nopython=True, cache=True)
def mykron(a, b):
    """
    Kronecker product of two 1D arrays.

    Parameters
    ----------
    a, b : 1D np.ndarray

    Returns
    -------
    out : 1D np.ndarray
    """

    num_a = a.shape[0]
    num_b = b.shape[0]
    out = np.empty(num_a * num_b, dtype=np.complex128)
    for i in range(num_a):
        ai = a[i]
        index = i * num_b
        for j in range(num_b):
            out[index + j] = ai * b[j]
    return out


def multikron(vectors):
    """
    Kronecker product of the given 1D arrays.

    Parameters
    ----------
    vectors : A collection of 1D arrays.

    Returns
    -------
    out : The resulting Kronecker product.
    """

    num = len(vectors)
    if num == 1:
        return vectors[0]
    else:
        mid = num // 2
        left = multikron(vectors[0:mid])
        right = multikron(vectors[mid:])
        return mykron(left, right)


def ClusterGenerator(num1, num2=None, which="xy"):
    assert isinstance(num1, int) and num1 > 0
    assert (num2 is None) or (isinstance(num2, int) and num2 > 0)
    assert which in ("xy", "yz", "zx", "yx", "zy", "xz")
    if num2 is None:
        num2 = num1

    RX = np.array([1.0, 0.0], dtype=np.float64)
    RY = np.array([-0.5, np.sqrt(3) / 2], dtype=np.float64)
    RZ = np.array([0.5, np.sqrt(3) / 2], dtype=np.float64)
    AS = {
        "xy": np.array([RX, RY], dtype=np.float64),
        "yz": np.array([RY, RZ], dtype=np.float64),
        "zx": np.array([RZ, RX], dtype=np.float64),
        "yx": np.array([RY, RX], dtype=np.float64),
        "zy": np.array([RZ, RY], dtype=np.float64),
        "xz": np.array([RX, RZ], dtype=np.float64),
    }

    As = AS[which]
    vectors = As * np.array([[num1], [num2]])
    points = np.dot([[i, j] for i in range(num1) for j in range(num2)], As)
    return Lattice(points=points, vectors=vectors, name=which)


def StripeGenerator(cluster, config="StripeX"):
    if config == "StripeX":
        points = np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2]])
        vectors = np.array([[1.0, 0.0], [1.0, np.sqrt(3)]])
    elif config == "StripeY":
        points = np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2]])
        vectors = np.array([[1.5, np.sqrt(3) / 2], [1.0, np.sqrt(3)]])
    elif config == "StripeZ":
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        vectors = np.array([[2.0, 0.0], [0.5, np.sqrt(3) / 2]])
    else:
        raise ValueError("Invalid config: {0}".format(config))
    cell = Lattice(points, vectors)

    up_indices = []
    down_indices = []
    for point in cluster.points:
        index_cell = cell.getIndex(point, fold=True)
        index_cluster = cluster.getIndex(point, fold=False)
        if index_cell == 0:
            up_indices.append(index_cluster)
        else:
            down_indices.append(index_cluster)
    return up_indices, down_indices


def FMDirection(thetas, phis, ket, site_num):
    num_phis = phis.shape[0]
    num_thetas = thetas.shape[0]
    sin_half_thetas = np.sin(thetas / 2)
    cos_half_thetas = np.cos(thetas / 2)
    exp_1j_half_phis = np.exp(1j * phis / 2)
    exp_1j_half_phis_conj = exp_1j_half_phis.conjugate()

    spinors = np.empty((site_num, 2), dtype=np.complex128)
    probabilities = np.empty((num_thetas, num_phis), dtype=np.float64)
    for i in range(num_thetas):
        t0 = time()
        for j in range(num_phis):
            spinors[:, 0] = cos_half_thetas[i] * exp_1j_half_phis_conj[j]
            spinors[:, 1] = sin_half_thetas[i] * exp_1j_half_phis[j]

            # cluster_spin_coherent_state = multikron(spinors)
            inner = np.vdot(multikron(spinors), ket)
            probabilities[i, j] = (inner * inner.conjugate()).real
        t1 = time()
        logging.info("%03dth theta, dt=%.4fs", i, t1 - t0)
    return probabilities


def StripeDirection(
        thetas, phis, ket, num1, num2=None, which="xy", config="StripeX"
):
    cluster = ClusterGenerator(num1, num2, which)
    up_indices, down_indices = StripeGenerator(cluster, config=config)

    # The opposite direction to (theta, phi) is (pi - theta, phi + pi)
    num_phis = phis.shape[0]
    num_thetas = thetas.shape[0]
    sin_half_thetas_up = np.sin(thetas / 2)
    cos_half_thetas_up = np.cos(thetas / 2)
    exp_1j_half_phis_up = np.exp(1j * phis / 2)
    exp_1j_half_phis_conj_up = exp_1j_half_phis_up.conjugate()
    sin_half_thetas_down = np.sin((np.pi - thetas) / 2)
    cos_half_thetas_down = np.cos((np.pi - thetas) / 2)
    exp_1j_half_phis_down = np.exp(1j * (phis + np.pi) / 2)
    exp_1j_half_phis_conj_down = exp_1j_half_phis_down.conjugate()

    spinors = np.empty((cluster.point_num, 2), dtype=np.complex128)
    probabilities = np.empty((num_thetas, num_phis), dtype=np.float64)
    for i in range(num_thetas):
        t0 = time()
        for j in range(num_phis):
            tmp0 = cos_half_thetas_up[i] * exp_1j_half_phis_conj_up[j]
            tmp1 = sin_half_thetas_up[i] * exp_1j_half_phis_up[j]
            tmp2 = cos_half_thetas_down[i] * exp_1j_half_phis_conj_down[j]
            tmp3 = sin_half_thetas_down[i] * exp_1j_half_phis_down[j]
            spinors[up_indices, 0] = tmp0
            spinors[up_indices, 1] = tmp1
            spinors[down_indices, 0] = tmp2
            spinors[down_indices, 1] = tmp3

            # cluster_spin_coherent_state = multikron(spinors)
            inner = np.vdot(multikron(spinors), ket)
            probabilities[i, j] = (inner * inner.conjugate()).real
        t1 = time()
        logging.info("%03dth theta, dt=%.4fs", i, t1 - t0)
    return probabilities
