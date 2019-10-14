"""
This module provides utility programs used in this project.
"""


__all__ = [
    "TriangularLattice",
    "ShowVectorField2D",
    "ShowVectorField3D",
]


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy import lattice_generator


class TriangularLattice:
    """
    Base class of triangular lattice on which the J-K-Gamma-Gamma' (J-K-G-GP)
    model is defined.
    """

    # Default model parameters
    # "alpha" and "beta" for the J-K-Gamma (J-K-G) model
    # J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
    # K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
    # G = np.cos(alpha * np.pi)
    # "J", "K", "G" and "GP" for the J-K-Gamma-Gamma' (J-K-G-GP) model
    # The default model parameters correspond to ferromagnetic Heisenberg model
    DEFAULT_MODEL_PARAMETERS = {
        "alpha": 0.5, "beta": 1.5,
        "J": -1.0, "K": 0.0, "G": 0.0, "GP": 0.0,
    }

    def __init__(self, numx, numy=None):
        """
        Customize the newly created instance.

        Parameters
        ----------
        numx : int
            The number of lattice site along the 1st translation vector.
        numy : int or None, optional
            The number of lattice site along the 2nd translation vector.
            The default value `None` implies that `numy` takes the same value
            as `numx`.
        """

        assert isinstance(numx, int) and numx > 0
        assert (numy is None) or (isinstance(numy, int) and numy > 0)
        if numy is None:
            numy = numx

        cluster = lattice_generator("triangle", num0=numx, num1=numy)
        intra, inter = cluster.bonds(nth=1)
        x_bonds = []
        y_bonds = []
        z_bonds = []
        for bond in intra + inter:
            p0, p1 = bond.endpoints
            index0 = cluster.getIndex(site=p0, fold=True)
            index1 = cluster.getIndex(site=p1, fold=True)
            bond_index = (index0, index1)

            azimuth = bond.getAzimuth(ndigits=0)
            if azimuth in (-180, 0, 180):
                x_bonds.append(bond_index)
            elif azimuth in (-120, 60):
                y_bonds.append(bond_index)
            elif azimuth in (-60, 120):
                z_bonds.append(bond_index)
            else:
                raise ValueError("Invalid azimuth: {0}".format(azimuth))

        self._numx = numx
        self._numy = numy
        self._site_num = numx * numy
        self._cluster = cluster
        self._x_bonds = tuple(x_bonds)
        self._y_bonds = tuple(y_bonds)
        self._z_bonds = tuple(z_bonds)

    @property
    def numx(self):
        """
        The `numx` attribute.
        """

        return self._numx

    @property
    def numy(self):
        """
        The `numy` attribute.
        """

        return self._numy

    @property
    def site_num(self):
        """
        The `site_num` attribute.
        """

        return self._site_num

    @property
    def cluster(self):
        """
        The `cluster` attribute.
        """

        return self._cluster

    @property
    def x_bonds(self):
        """
        The `x_bonds` attribute.
        """

        return self._x_bonds

    @property
    def y_bonds(self):
        """
        The `y_bonds` attribute.
        """

        return self._y_bonds

    @property
    def z_bonds(self):
        """
        The `z_bonds` attribute.
        """

        return self._z_bonds

    @property
    def x_bond_num(self):
        """
        The number of x-type bonds.
        """

        return len(self._x_bonds)

    @property
    def y_bond_num(self):
        """
        The number of y-type bonds.
        """

        return len(self._y_bonds)

    @property
    def z_bond_num(self):
        """
        The number of z-type bonds.
        """

        return len(self._z_bonds)

    @property
    def all_bonds(self):
        """
        The `all_bonds` attribute.
        """

        return self._x_bonds, self._y_bonds, self._z_bonds


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

    assert isinstance(vectors, np.ndarray) and vectors.ndim == 2 and \
           vectors.shape[1] in (2, 3)

    normalized_vectors = vectors / np.linalg.norm(
        vectors, axis=1, keepdims=True
    )

    N, D = vectors.shape
    colors = np.zeros((N, 3), dtype=np.float64)
    colors[:, 0:D] = 0.5 * normalized_vectors + 0.5
    return colors


def ShowVectorField2D(ax, points, vectors, title="", markersize=10, **kwargs):
    """
    Show 2D vector field.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes instance on which to draw the 2D vector field.
    points : np.ndarray with shape (N, 2)
        The coordinates of the vector field.
    vectors : np.ndarray with shape (N, 2)
        The x and y direction components of the vectors.
    title : str, optional
        The title of the axes.
        Default: "" (empty string).
    markersize : float, optional
        The size of the marker.
        Default: 10.
    **kwargs :
        Other keyword arguments are passed to the `Axes.quiver` method.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The passed in Axes instance.
    """

    ax.plot(
        points[:, 0], points[:, 1],
        color="k", ls="", marker="o", ms=markersize
    )
    ax.quiver(
        points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1], vectors,
        pivot="middle",
    )
    ax.set_title(title, fontsize="xx-large")
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def ShowVectorField3D(ax, points, vectors, title="", markersize=10, **kwargs):
    """
    Show 3D vector field.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Axes instance on which to draw the 3D vector field.
    points : np.ndarray with shape (N, 2) or (N, 3)
        The coordinates of the vector field.
    vectors : np.ndarray with shape (N, 3)
        The x, y and z direction components of the vectors.
    title : str, optional
        The title of the axes.
        Default: "" (empty string).
    markersize : float, optional
        The size of the marker.
        Default: 10.
    **kwargs :
        Other keyword arguments are passed to the `Axes.quiver` method.

    Returns
    -------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The passed in Axes instance.
    """

    N, D = points.shape
    xs = points[:, 0]
    ys = points[:, 1]
    if D == 2:
        zs = 0.0
    elif D == 3:
        zs = points[:, 2]
    else:
        raise ValueError("Invalid `points` parameter!")

    tmp = Vectors2Colors(vectors)
    colors = np.zeros((N * 3, 3), dtype=np.float64)
    colors[0:N, :] = tmp
    colors[N::2, :] = tmp
    colors[(N+1)::2, :] = tmp

    ax.plot(xs, ys, zs, color="k", ls="", marker="o", ms=markersize)
    ax.quiver(
        xs, ys, zs, vectors[:, 0], vectors[:, 1], vectors[:, 2],
        length=1.0, pivot="middle", colors=colors,
    )
    ax.set_title(title, fontsize="xx-large")
    return ax


if __name__ == "__main__":
    numx = numy = 10
    site_num = numx * numy
    cluster = TriangularLattice(numx, numy)

    vectors2D = np.random.random((site_num, 2))
    vectors3D = np.random.random((site_num, 3))
    vectors2D = vectors2D / np.linalg.norm(vectors2D, axis=1, keepdims=True)
    vectors3D = vectors3D / np.linalg.norm(vectors3D, axis=1, keepdims=True)

    fig = plt.figure("VectorField2D")
    ax = fig.add_subplot(111)
    ShowVectorField2D(
        ax, cluster.cluster.points, vectors2D,
        markersize=8, title="VectorField2D"
    )
    ax.set_axis_off()

    fig = plt.figure("VectorField3D")
    ax = fig.add_subplot(111, projection="3d")
    ShowVectorField3D(
        ax, cluster.cluster.points, vectors3D,
        markersize=8, title="VectorField3D"
    )
    ax.set_zlim(-0.5, 0.5)

    plt.show()
    plt.close("all")
