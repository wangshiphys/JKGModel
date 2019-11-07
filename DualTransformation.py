"""
This module provide functions for visualizing the FM and 120-degree Neel
order as well as the results of Four-Sublattice transformation.
"""


__all__ = [
    "T1", "T1T4",
    "GenerateFMOrder", "GenerateNeelOrder",
    "Show120Neel", "ShowFMT1T4", "ShowNeelT1T4",
]


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy import Lattice, lattice_generator
from HamiltonianPy.rotation3d import E, RX180, RY180, RZ180, R111_180, RotationGeneral

from utilities import *


FOUR_SUBLATTICE_TRANSFORMATION_CELL = Lattice(
    points=np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]]
    ),
    vectors=np.array([[2.0, 0.0], [1.0, np.sqrt(3)]])
)
FOUR_SUBLATTICE_ROTATIONS = {0: E, 1: RX180, 2:RY180, 3: RZ180}

NEEL_ORDER_MAGNETIC_CELL = Lattice(
    points=np.array([[0.0, 0.0], [0.5, np.sqrt(3)/2], [1.0, 0.0]]),
    vectors=np.array([[1.5, -np.sqrt(3)/2], [1.5, np.sqrt(3)/2]]),
)


def GenerateFMOrder(points, cell_vectors=None):
    """
    Generate FM ordered spin vectors.

    Parameters
    ----------
    points : np.ndarray
        The coordinates of the points on which the spin vectors are defined.
    cell_vectors : np.ndarray with shape (3, ) or None
        The spin vector for the magnetic unit cell.
        If None, a random 3D vector is used.

    Returns
    -------
    spin_vectors : np.ndarray
        The spin vectors on the given `points`.
    """

    if cell_vectors is None:
        cell_vectors = 2 * np.random.random(3) - 1

    cell_vectors = cell_vectors / np.linalg.norm(cell_vectors)
    spin_vectors = np.zeros((points.shape[0], 3), dtype=np.float64)
    spin_vectors[:] = cell_vectors
    return spin_vectors


def GenerateNeelOrder(points, cell_vectors=None):
    """
    Generate 120 degree Neel ordered spin vectors.

    Parameters
    ----------
    points : np.ndarray
        The coordinates of the points on which the spin vectors are defined.
    cell_vectors : np.ndarray with shape (3, 3) or None
        The spin vectors for the magnetic unit cell, every row represent a
        spin vector. If None, random 120 degree Neel ordered unit-cell spin
        vectors are generated.

    Returns
    -------
    spin_vectors : np.ndarray
        The spin vectors on the given `points`.
    """

    if cell_vectors is None:
        vx, vy, vz = 2 * np.random.random(3) - 1
        R120 = RotationGeneral((0.0, -vz, vy), theta=120, deg=True)
        R240 = RotationGeneral((0.0, -vz, vy), theta=240, deg=True)
        cell_vectors = np.array(
            [
                np.array([vx, vy, vz]),
                np.dot(R120, [vx, vy, vz]),
                np.dot(R240, [vx, vy, vz]),
            ], dtype=np.float64
        )
    cell_vectors = cell_vectors / np.linalg.norm(
        cell_vectors, axis=1, keepdims=True
    )

    spin_vectors = []
    for point in points:
        index = NEEL_ORDER_MAGNETIC_CELL.getIndex(point, fold=True)
        spin_vectors.append(cell_vectors[index])
    return np.array(spin_vectors, dtype=np.float64)


def T1(points, spin_vectors):
    """
    Rotate all spin vectors about the (111)-axis by 180 degree.
    """

    return np.dot(R111_180, spin_vectors.T).T


def T4(points, spin_vectors):
    """
    Perform Four-Sublattice transformation for the given spin vectors.
    """

    rotated_vectors = []
    for point, vector in zip(points, spin_vectors):
        index = FOUR_SUBLATTICE_TRANSFORMATION_CELL.getIndex(point, fold=True)
        rotation = FOUR_SUBLATTICE_ROTATIONS[index]
        rotated_vectors.append(np.dot(rotation, vector))
    return np.array(rotated_vectors, dtype=np.float64)


def T1T4(points, spin_vectors):
    """
    Combination of T1 and T4 (T4 followed by T1)
    """

    return T1(points, T4(points, spin_vectors))


def Show120Neel(points, cell_vectors=None, *, markersize=10, **kwargs):
    spin_vectors = GenerateNeelOrder(points, cell_vectors)
    fig = plt.figure("120-Neel-Order")
    ax = fig.add_subplot(111, projection="3d")
    ShowVectorField3D(
        ax, points, spin_vectors, markersize=markersize, **kwargs
    )
    ax.set_zlim(-0.5, 0.5)
    plt.show()
    plt.close("all")


def ShowFMT1T4(points, cell_vectors=None, *, markersize=10, **kwargs):
    original_vectors = GenerateFMOrder(points, cell_vectors)
    T4_rotated_vectors = T4(points, original_vectors)
    T1T4_rotated_vectors = T1T4(points, original_vectors)

    fig_original = plt.figure("original")
    ax_original = fig_original.add_subplot(111, projection="3d")
    fig_T4_rotated = plt.figure("T4-rotated")
    ax_T4_rotated = fig_T4_rotated.add_subplot(111, projection="3d")
    fig_T1T4_rotated = plt.figure("T1T4-rotated")
    ax_T1T4_rotated = fig_T1T4_rotated.add_subplot(111, projection="3d")
    ShowVectorField3D(
        ax_original, points, original_vectors,
        markersize=markersize, **kwargs
    )
    ShowVectorField3D(
        ax_T4_rotated, points, T4_rotated_vectors,
        markersize=markersize, **kwargs
    )
    ShowVectorField3D(
        ax_T1T4_rotated, points, T1T4_rotated_vectors,
        markersize=markersize, **kwargs
    )
    ax_original.set_zlim(-0.5, 0.5)
    ax_T4_rotated.set_zlim(-0.5, 0.5)
    ax_T1T4_rotated.set_zlim(-0.5, 0.5)
    plt.show()
    plt.close("all")


def ShowNeelT1T4(points, cell_vectors=None, *, markersize=10, **kwargs):
    original_vectors = GenerateNeelOrder(points, cell_vectors)
    T4_rotated_vectors = T4(points, original_vectors)
    T1T4_rotated_vectors = T1T4(points, original_vectors)

    fig_original = plt.figure("original")
    ax_original = fig_original.add_subplot(111, projection="3d")
    fig_T4_rotated = plt.figure("T4-rotated")
    ax_T4_rotated = fig_T4_rotated.add_subplot(111, projection="3d")
    fig_T1T4_rotated = plt.figure("T1T4-rotated")
    ax_T1T4_rotated = fig_T1T4_rotated.add_subplot(111, projection="3d")
    ShowVectorField3D(
        ax_original, points, original_vectors,
        markersize=markersize, **kwargs
    )
    ShowVectorField3D(
        ax_T4_rotated, points, T4_rotated_vectors,
        markersize=markersize, **kwargs
    )
    ShowVectorField3D(
        ax_T1T4_rotated, points, T1T4_rotated_vectors,
        markersize=markersize, **kwargs
    )
    ax_original.set_zlim(-0.5, 0.5)
    ax_T4_rotated.set_zlim(-0.5, 0.5)
    ax_T1T4_rotated.set_zlim(-0.5, 0.5)
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    numx = 12
    numy = 12
    site_num = numx * numy
    cluster = lattice_generator("triangle", num0=numx, num1=numy)

    Show120Neel(cluster.points)
    ShowFMT1T4(cluster.points)
    ShowNeelT1T4(cluster.points)
