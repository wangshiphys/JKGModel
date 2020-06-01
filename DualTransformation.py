"""
This module provide functions for visualizing the FM and 120-degree Neel
order as well as the results of Four-Sublattice transformation.
"""


__all__ = [
    "T1", "T4", "T1T4",
    "GenerateFMOrder", "GenerateNeelOrder",
    "Show120Neel", "ShowFMT1T4", "ShowNeelT1T4",
]

import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

FOUR_SUBLATTICE_TRANSFORMATION_CELL = HP.Lattice(
    points=np.array(
        [[0.0, 0.0], [1.0, 0.0], [-0.5, np.sqrt(3) / 2], [0.5, np.sqrt(3) / 2]]
    ),
    vectors=np.array([[2.0, 0.0], [1.0, np.sqrt(3)]])
)
FOUR_SUBLATTICE_ROTATIONS = {
    0: HP.rotation3d.E,
    1: HP.rotation3d.RX180,
    2: HP.rotation3d.RY180,
    3: HP.rotation3d.RZ180,
}

NEEL_ORDER_MAGNETIC_CELL = HP.Lattice(
    points=np.array([[0.0, 0.0], [0.5, np.sqrt(3)/2], [1.0, 0.0]]),
    vectors=np.array([[1.5, -np.sqrt(3)/2], [1.5, np.sqrt(3)/2]]),
)


def Vectors2Colors(vectors):
    """
    Convert 2D/3D vectors to colors.

    Parameters
    ----------
    vectors : np.ndarray with shape (N, 2) or (N, 3)

    Returns
    -------
    colors : np.ndarray with shape (N, 3)
        The corresponding RGBA colors.
    """

    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2 and vectors.shape[1] in (2, 3)

    normalized_vectors = vectors / np.linalg.norm(
        vectors, axis=1, keepdims=True
    )

    N, D = vectors.shape
    colors = np.zeros((N, 3), dtype=np.float64)
    colors[:, 0:D] = 0.5 * normalized_vectors + 0.5
    return colors


def GenerateFMOrder(points, cell_vectors=None):
    """
    Generate FM ordered spin vectors.

    Parameters
    ----------
    points : np.ndarray
        The coordinates of the points on which the spin vectors are defined.
    cell_vectors : np.ndarray with shape (3, ) or None, optional
        The spin vector for the magnetic unit cell.
        If None, a random 3D vector is used.
        Default: None.

    Returns
    -------
    spin_vectors : np.ndarray
        The spin vectors on the given `points`.
    """

    if cell_vectors is None:
        cell_vectors = 2 * np.random.random(3) - 1

    cell_vectors = cell_vectors / np.linalg.norm(cell_vectors)
    spin_vectors = np.empty((points.shape[0], 3), dtype=np.float64)
    spin_vectors[:] = cell_vectors
    return spin_vectors


def GenerateNeelOrder(points, cell_vectors=None):
    """
    Generate 120-degree Neel ordered spin vectors.

    Parameters
    ----------
    points : np.ndarray
        The coordinates of the points on which the spin vectors are defined.
    cell_vectors : np.ndarray with shape (3, 3) or None, optional
        The spin vectors for the magnetic unit cell, every row represent a
        spin vector. If None, random 120-degree Neel ordered unit-cell spin
        vectors are generated.
        Default: None.

    Returns
    -------
    spin_vectors : np.ndarray
        The spin vectors on the given `points`.
    """

    if cell_vectors is None:
        angle = 360 * np.random.random()
        nx, ny, nz = 2 * np.random.random(3) - 1
        cell_vectors = np.empty((3, 3), dtype=np.float64)
        R0 = HP.RotationGeneral((nx, ny, nz), theta=angle, deg=True)
        R1 = HP.RotationGeneral((nx, ny, nz), theta=angle + 120, deg=True)
        R2 = HP.RotationGeneral((nx, ny, nz), theta=angle + 240, deg=True)
        cell_vectors[0] = np.dot(R0, [0.0, -nz, ny])
        cell_vectors[1] = np.dot(R1, [0.0, -nz, ny])
        cell_vectors[2] = np.dot(R2, [0.0, -nz, ny])
    cell_vectors = cell_vectors / np.linalg.norm(
        cell_vectors, axis=1, keepdims=True
    )

    point_num = points.shape[0]
    spin_vectors = np.empty((point_num, 3), dtype=np.float64)
    for i in range(point_num):
        index = NEEL_ORDER_MAGNETIC_CELL.getIndex(points[i], fold=True)
        spin_vectors[i] = cell_vectors[index]
    return spin_vectors


def T1(points, spin_vectors):
    """
    Rotate all spin vectors about the (111)-axis by 180 degree.

    Parameters
    ----------
    points : np.ndarray with shape (N, M)
        The coordinates of the points on which the spin vectors are defined.
    spin_vectors : np.ndarray with shape (N, 3)
        The spin vectors defined on these points.
        Every row represent a spin vector.

    Returns
    -------
    rotated_spin_vectors : np.ndarray
        The rotated spin vectors.
        Every row represent a spin vector.
    """

    return np.dot(HP.rotation3d.R111_180, spin_vectors.T).T


def T4(points, spin_vectors):
    """
    Perform Four-Sublattice Transformation for the given spin vectors.

    Parameters
    ----------
    points : np.ndarray with shape (N, 2)
        The coordinates of the points on which the spin vectors are defined.
    spin_vectors : np.ndarray with shape (N, 3)
        The spin vectors defined on these points.
        Every row represent a spin vector.

    Returns
    -------
    rotated_spin_vectors : np.ndarray
        The rotated spin vectors.
        Every row represent a spin vector.
    """

    rotated_spin_vectors = np.empty_like(spin_vectors, dtype=np.float64)
    for i in range(points.shape[0]):
        point = points[i]
        index = FOUR_SUBLATTICE_TRANSFORMATION_CELL.getIndex(point, fold=True)
        rotation = FOUR_SUBLATTICE_ROTATIONS[index]
        rotated_spin_vectors[i] = np.dot(rotation, spin_vectors[i])
    return rotated_spin_vectors


def T1T4(points, spin_vectors):
    """
    Combination of T1 and T4 (T4 followed by T1).

    Parameters
    ----------
    points : np.ndarray with shape (N, 2)
        The coordinates of the points on which the spin vectors are defined.
    spin_vectors : np.ndarray with shape (N, 3)
        The spin vectors defined on these points.
        Every row represent a spin vector.

    Returns
    -------
    rotated_spin_vectors : np.ndarray
        The rotated spin vectors.
        Every row represent a spin vector.
    """

    return T1(points, T4(points, spin_vectors))


def Show120Neel(points, cell_vectors=None, *, markersize=5.0, **kwargs):
    """
    Show the 120-degree Neel order.

    Parameters
    ----------
    points, cell_vectors
        See the document of `GenerateNeelOrder`.
    markersize : float, optional
        The size of these points.
        Default: 5.0.
    kwargs
        Other keyword arguments are passed to `Axes.quiver`.
    """

    spin_vectors = GenerateNeelOrder(points, cell_vectors)

    fig, ax = plt.subplots(num="120-Neel-Order")
    ax.plot(
        points[:, 0], points[:, 1], color="k", ls="", marker="o", ms=markersize
    )
    ax.quiver(
        points[:, 0], points[:, 1], spin_vectors[:, 0], spin_vectors[:, 1],
        pivot="middle", color=Vectors2Colors(spin_vectors), **kwargs,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except Exception:
        pass
    plt.show()
    plt.close("all")


def ShowFMT1T4(points, cell_vectors=None, *, markersize=5.0, **kwargs):
    """
    Show Four-Sublattice Transformation of FM order state.

    Parameters
    ----------
    points, cell_vectors
        See the document of `GenerateFMOrder`.
    markersize : float, optional
        The size of these points.
        Default: 5.0.
    kwargs
        Other keyword arguments are passed to `Axes.quiver`.
    """

    original_vectors = GenerateFMOrder(points, cell_vectors)
    T4_rotated_vectors = T4(points, original_vectors)
    T1T4_rotated_vectors = T1T4(points, original_vectors)

    fig_original, ax_original = plt.subplots(num="original")
    fig_T4_rotated, ax_T4_rotated = plt.subplots(num="T4_rotated")
    fig_T1T4_rotated, ax_T1T4_rotated = plt.subplots(num="T1T4_rotated")

    for ax, vectors in zip(
        [ax_original, ax_T4_rotated, ax_T1T4_rotated],
        [original_vectors, T4_rotated_vectors, T1T4_rotated_vectors]
    ):
        ax.plot(
            points[:, 0], points[:, 1],
            color="k", ls="", marker="o", ms=markersize
        )
        ax.quiver(
            points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1],
            pivot="middle", color=Vectors2Colors(vectors), **kwargs,
        )
        ax.set_axis_off()
        ax.set_aspect("equal")
    plt.show()
    plt.close("all")


def ShowNeelT1T4(points, cell_vectors=None, *, markersize=5.0, **kwargs):
    """
    Show Four-Sublattice Transformation of 120-degree Neel order state.

    Parameters
    ----------
    points, cell_vectors
        See the document of `GenerateNeelOrder`.
    markersize : float, optional
        The size of these points.
        Default: 5.0.
    kwargs
        Other keyword arguments are passed to `Axes.quiver`.
    """

    original_vectors = GenerateNeelOrder(points, cell_vectors)
    T4_rotated_vectors = T4(points, original_vectors)
    T1T4_rotated_vectors = T1T4(points, original_vectors)

    fig_original, ax_original = plt.subplots(num="original")
    fig_T4_rotated, ax_T4_rotated = plt.subplots(num="T4_rotated")
    fig_T1T4_rotated, ax_T1T4_rotated = plt.subplots(num="T1T4_rotated")

    for ax, vectors in zip(
            [ax_original, ax_T4_rotated, ax_T1T4_rotated],
            [original_vectors, T4_rotated_vectors, T1T4_rotated_vectors]
    ):
        ax.plot(
            points[:, 0], points[:, 1],
            color="k", ls="", marker="o", ms=markersize
        )
        ax.quiver(
            points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1],
            pivot="middle", color=Vectors2Colors(vectors), **kwargs,
        )
        ax.set_axis_off()
        ax.set_aspect("equal")
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    num0 = num1 = 12
    vectors = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], dtype=np.float64)
    points = np.dot([[i, j] for i in range(num0) for j in range(num1)], vectors)

    FM_CELL_VECTORS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    NEEL_CELL_VECTORS = np.array(
        [
            [-np.sqrt(3) / 2, -0.5, 0.0],
            [0.0, 1.0, 0.0],
            [np.sqrt(3) / 2, -0.5, 0.0]
        ], dtype=np.float64
    )

    Show120Neel(points)
    Show120Neel(points, cell_vectors=NEEL_CELL_VECTORS)

    ShowFMT1T4(points)
    ShowFMT1T4(points, cell_vectors=FM_CELL_VECTORS)

    ShowNeelT1T4(points)
    ShowNeelT1T4(points, cell_vectors=NEEL_CELL_VECTORS)
