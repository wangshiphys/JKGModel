"""
Test the moment direction of a magnetic ordered state.
"""

__all__ = ["FMDirection", "StripeDirection"]


import logging
from time import time

import numpy as np
from HamiltonianPy import Lattice
from numba import complex128, jit

from utilities import TriangularLattice


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


def StripeGenerator(cluster, config="StripeX"):
    """
    Generate stripe ordered state on the given cluster.

    Parameters
    ----------
    cluster : Lattice
        The cluster on which to generate the stripe ordered state.
    config : ["StripeX" | "StripeY", | "StripeZ"], str, optional
        The type of the stripe order to generate.
        If set to "StripeX", the spins along the x-bond direction are parallel;
        If set to "StripeY", the spins along the y-bond direction are parallel;
        If set to "StripeZ", the spins along the z-bond direction are parallel.
        Default: "StripeX".

    Returns
    -------
    spin_up_indices : list of int
        The indices of the lattice sites which are in spin-up state.
    spin_down_indices : list of int
        The indices of the lattice sites which are in spin-down state.
    """

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
        raise ValueError("Invalid `config`: {0}".format(config))
    cell = Lattice(points, vectors)

    spin_up_indices = []
    spin_down_indices = []
    for point in cluster.points:
        index_cell = cell.getIndex(point, fold=True)
        index_cluster = cluster.getIndex(point, fold=False)
        if index_cell == 0:
            spin_up_indices.append(index_cluster)
        else:
            spin_down_indices.append(index_cluster)
    return spin_up_indices, spin_down_indices


def FMDirection(thetas, phis, ket, site_num):
    """
    The probabilities of the given `ket` in different polarized FM states.

    thetas : 1D array
        A collection of radial angles.
        The radial angle is the angle between the moment direction and the
        z-axis. It should be in the range [0, pi].
    phis : 1D array
        A collection of azimuth angles.
        The azimuth angle is the angle between the projection of the moment
        direction on the xy-plane and the x-axis. It should be in the
        range [0, 2*pi). `theta` and `phi` specify the direction of the
        ordered moment. The given `thetas` and `phis` specify a mesh of
        different directions.
    site_num : int
        The number of lattice sites of the system.
    ket : 1D array with shape (2 ** site_num, )
        Matrix representation of the quantum state.

    Return
    ------
    probabilities : 2D array
        The probabilities of the `ket` in these polarized FM states.
    """

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
        logging.info("%03dth theta, dt=%.3fs", i, t1 - t0)
    return probabilities


def StripeDirection(
        thetas, phis, ket, *, num1=4, num2=6, direction="xy", config="StripeX"
):
    """
    The probabilities of the given `ket` in different polarized stripe states.

    thetas : 1D array
        A collection of radial angles.
        The radial angle is the angle between the moment direction and the
        z-axis. It should be in the range [0, pi].
    phis : 1D array
        A collection of azimuth angles.
        The azimuth angle is the angle between the projection of the moment
        direction on the xy-plane and the x-axis. It should be in the
        range [0, 2*pi). `theta` and `phi` specify the direction of the
        ordered moment. The given `thetas` and `phis` specify a mesh of
        different directions.
    ket : 1D array with shape (2 ** site_num, )
        Matrix representation of the quantum state.
    config : ["StripeX" | "StripeY", | "StripeZ"], str, optional
        Determine the type of the stripe order.
        If set to "StripeX", the spins along the x-bond direction are parallel;
        If set to "StripeY", the spins along the y-bond direction are parallel;
        If set to "StripeZ", the spins along the z-bond direction are parallel.
        Default: "StripeX".
    num1, num2, direction
        They are passed to the constructor of `TriangularLattice` defined in
        `utilities` module. See also the document of `TriangularLattice`.

    Return
    ------
    probabilities : 2D array
        The probabilities of the `ket` in these polarized stripe states.
    """

    cluster = TriangularLattice(num1, num2, direction).cluster
    spin_up_indices, spin_down_indices = StripeGenerator(cluster, config=config)

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
            spinors[spin_up_indices, 0] = tmp0
            spinors[spin_up_indices, 1] = tmp1
            spinors[spin_down_indices, 0] = tmp2
            spinors[spin_down_indices, 1] = tmp3

            # cluster_spin_coherent_state = multikron(spinors)
            inner = np.vdot(multikron(spinors), ket)
            probabilities[i, j] = (inner * inner.conjugate()).real
        t1 = time()
        logging.info("%03dth theta, dt=%.3fs", i, t1 - t0)
    return probabilities
