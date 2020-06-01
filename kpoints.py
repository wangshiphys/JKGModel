"""
k-points in the reciprocal space correspond to several 4*6 clusters with
periodical boundary conditions (PBC).
"""


__all__ = ["KPoints", "ShowKPoints"]


import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy import TRIANGLE_CELL_KS

BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]


def KPoints(direction="xy"):
    """
    Calculate k-points in the reciprocal space correspond to specific `4 * 6`
    cluster with PBC.

    In general, the k-points in the reciprocal space correspond to the specific
    `4 * 6` cluster with PBC can be given by the following formula:
        `x, y = np.dot(M_INV, [i, j])`
        `kx, ky = np.dot([x, y], cell_bs)`
    where `i`, `j` are integer numbers and `(i, j)` can be used as the
    identity of the k-point.
    As for all non-equivalent k-points, both `x` and `y` should be in the
    range [-0.5, 0.5).

    Parameters
    ----------
    direction : ["xy" | "xz" | "yx" | "yz" | "zx" | "zy"], str, optional
        Define the direction of the `4 * 6` cluster.
        For example, if `direction` is set to "xy", then there are 4 lattice
        sites along the x-bond direction and 6 lattice sites along the
        y-bond direction.
        Default: "xy".

    Returns
    -------
    M_INV : np.ndarray with shape (2, 2)
    cell_bs : np.ndarray with shape (2, 2)
    ids : tuple of length-2 tuples
        A collection of the identities of the k-points in the first
        Brillouin zone.
    """

    if direction == "xy":
        cell_vectors = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
        ids = (
            (-2, 0), (-2, 1), (-2, 2), (-2, 3),
            (-1, 3), (-1, 2), (-1, 1), (-1, 0), (-1, -1), (-1, -2),
            (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 1), (1, 0), (1, -1), (1, -2), (1, -3),
            (2, -3), (2, -2), (2, -1), (2, 0),
        )
    elif direction == "xz":
        cell_vectors = np.array([[1.0, 0.0], [0.5, np.sqrt(3)/2]])
        ids = (
            (-2, -3), (-2, -2), (-2, -1), (-2, 0),
            (-1, 2), (-1, 1), (-1, 0), (-1, -1), (-1, -2), (-1, -3),
            (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (1, 2), (1, 1), (1, 0), (1, -1), (1, -2),
            (2, 0), (2, 1), (2, 2), (2, 3),
        )
    elif direction == "yx":
        cell_vectors = np.array([[-0.5, np.sqrt(3)/2], [1.0, 0.0]])
        ids = (
            (-2, 0), (-2, 1), (-2, 2), (-2, 3),
            (-1, 3), (-1, 2), (-1, 1), (-1, 0), (-1, -1), (-1, -2),
            (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 1), (1, 0), (1, -1), (1, -2), (1, -3),
            (2, -3), (2, -2), (2, -1), (2, 0),
        )
    elif direction == "yz":
        cell_vectors = np.array([[-0.5, np.sqrt(3)/2], [0.5, np.sqrt(3)/2]])
        ids = (
            (-2, -3), (-2, -2), (-2, -1), (-2, 0),
            (-1, 2), (-1, 1), (-1, 0), (-1, -1), (-1, -2), (-1, -3),
            (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (1, 2), (1, 1), (1, 0), (1, -1), (1, -2),
            (2, 0), (2, 1), (2, 2), (2, 3),
        )
    elif direction == "zx":
        cell_vectors = np.array([[0.5, np.sqrt(3)/2], [1.0, 0.0]])
        ids = (
            (-2, -3), (-2, -2), (-2, -1), (-2, 0),
            (-1, 2), (-1, 1), (-1, 0), (-1, -1), (-1, -2), (-1, -3),
            (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (1, 2), (1, 1), (1, 0), (1, -1), (1, -2),
            (2, 0), (2, 1), (2, 2), (2, 3),
        )
    elif direction == "zy":
        cell_vectors = np.array([[0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2]])
        ids = (
            (-2, -3), (-2, -2), (-2, -1), (-2, 0),
            (-1, 2), (-1, 1), (-1, 0), (-1, -1), (-1, -2), (-1, -3),
            (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (1, 2), (1, 1), (1, 0), (1, -1), (1, -2),
            (2, 0), (2, 1), (2, 2), (2, 3),
        )
    else:
        raise ValueError("Invalid `direction`: {0}.".format(direction))

    M = np.array([[4, 0], [0, 6]])
    M_INV = np.linalg.inv(M)
    cell_bs = 2 * np.pi * np.linalg.inv(cell_vectors.T)
    return M_INV, cell_bs, ids


def ShowKPoints(
        direction="xy", background=True, show_id=True,
        lw=6, ms=10, fontsize="xx-small", save=False,
):
    M_INV, cell_bs, ids = KPoints(direction)

    fig, ax = plt.subplots(num="direction={0}".format(direction))
    if background:
        for i in range(-10, 11):
            for j in range(-10, 11):
                kx, ky = np.dot(np.dot(M_INV, [i, j]), cell_bs)
                ax.plot(kx, ky, marker="o", ms=ms, color="tab:green", zorder=2)

    kpoints = []
    for i, j in ids:
        kpoint = np.dot(np.dot(M_INV, [i, j]), cell_bs)
        kpoints.append(kpoint)
        ax.plot(
            kpoint[0], kpoint[1],
            marker="o", ms=0.6*ms, color="tab:red", zorder=3,
        )
        if show_id:
            ax.text(
                kpoint[0], kpoint[1], "({0},{1})".format(i, j),
                fontsize=fontsize, ha="center", va="center",
                clip_on=True, zorder=4,
            )

    kpoints = np.array(kpoints)
    ax.plot(
        kpoints[:, 0], kpoints[:, 1],
        ls="solid", lw=lw/3, color="tab:orange", zorder=1,
    )
    ax.plot(
        BZBoundary[:, 0], BZBoundary[:, 1],
        ls="solid", lw=lw, color="tab:blue", zorder=0,
    )
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-6.0, 6.0)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.set_size_inches(4, 4)
    plt.show()

    if save:
        fig_name = "KPoints_direction={0}.png".format(direction)
        fig.savefig(fig_name, dpi=300, transparent=True)
    plt.close("all")


if __name__ == "__main__":
    for direction in ["xy", "xz", "yx", "yz", "zx", "zy"]:
        ShowKPoints(direction, background=False, show_id=True)
