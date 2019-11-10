"""
Calculate the static structure factor of J-K-Gamma-Gamma' model on
triangular lattice.
"""


__all__ = [
    "QuantumSpinStructureFactor",
    "ClassicalSpinStructureFactor",
]


from itertools import combinations
from numba import complex128, float64, jit, prange

import numpy as np
from HamiltonianPy import SpinInteraction


# Core function for calculating the static structure factors
@jit(
    complex128[:, :](float64[:, :, :], float64[:, :], float64[:]),
    nopython=True, cache=True, parallel=True,
)
def _StructureFactorCore(kpoints, dRs, correlations):
    numkx, numky, space_dim = kpoints.shape
    factors = np.zeros((numkx, numky), np.complex128)
    for i in prange(numkx):
        for j in range(numky):
            factors[i, j] = np.sum(
                np.exp(1j * np.dot(dRs, kpoints[i, j])) * correlations
            )
    return factors


def QuantumSpinStructureFactor(kpoints, points, ket):
    """
    Calculate static structure factors over the given `ket`.

    Parameters
    ----------
    kpoints : 3D array with shape (nkx, nky, 2)
        A collection of k-points.
    points : 2D array with shape (site_num, 2)
        A collection of coordinates of lattice sites.
    ket : 2D array with shape (2**site_num, 1)
        Matrix representation of a quantum state.
        `ket` should be normalized.

    Returns
    -------
    factors : 2D array with shape (nkx, nky)
        The corresponding static structure factors.
    """

    site_num, space_dim = points.shape
    correlations = 0.75 * np.identity(site_num, dtype=np.float64)
    dRs = np.zeros((site_num, site_num, space_dim), dtype=np.float64)
    mfunc = SpinInteraction.matrix_function
    for i, j in combinations(range(site_num), r=2):
        Siz_dot_Sjz = mfunc([(i, "z"), (j, "z")], site_num, coeff=0.5)
        Sip_dot_Sjm = mfunc([(i, "p"), (j, "m")], site_num, coeff=0.5)
        avg_ij = np.vdot(ket, Siz_dot_Sjz.dot(ket))
        avg_ij += np.vdot(ket, Sip_dot_Sjm.dot(ket))
        avg_ij += avg_ij.conj()
        del Siz_dot_Sjz, Sip_dot_Sjm

        correlations[i, j] = correlations[j, i] = avg_ij.real
        dRs[i, j] = points[i] - points[j]
        dRs[j, i] = points[j] - points[i]
    dRs = np.reshape(dRs, newshape=(-1, space_dim))
    correlations = np.reshape(correlations, newshape=(-1, ))
    return _StructureFactorCore(kpoints, dRs, correlations) / site_num


def ClassicalSpinStructureFactor(kpoints, points, vectors):
    """
    Calculate static structure factors over a specific vectors configuration.

    Parameters
    ----------
    kpoints : 3D array with shape (nkx, nky, 2)
        A collection of k-points.
    points : 2D array with shape (site_num, 2)
        A collection of coordinates of lattice sites.
    vectors : 2D array with shape (site_num, 3)
        A collection of 3D unit vectors defined on `points`.

    Returns
    -------
    factors : 2D array with shape (nkx, nky)
        The corresponding static structure factors.
    """

    site_num, space_dim = points.shape
    correlations = np.identity(site_num, dtype=np.float64)
    dRs = np.zeros((site_num, site_num, space_dim), dtype=np.float64)
    for i, j in combinations(range(site_num), r=2):
        vi_dot_vj = np.matmul(vectors[i], vectors[j])
        correlations[i, j] = correlations[j, i] = vi_dot_vj
        dRs[i, j] = points[i] - points[j]
        dRs[j, i] = points[j] - points[i]
    dRs = np.reshape(dRs, newshape=(-1, space_dim))
    correlations = np.reshape(correlations, newshape=(-1, ))
    return _StructureFactorCore(kpoints, dRs, correlations) / site_num
