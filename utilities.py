"""
This module provides utility programs used in this project.
"""


__all__ = [
    "TriangularLattice",
    "derivation",
    "ShowVectorField2D",
    "ShowVectorField3D",
    "mykron0", "mykron1",
]


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from numba import complex128, jit, void
from HamiltonianPy import Lattice


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

    def __init__(self, num1, num2=None, which="xz"):
        """
        Customize the newly created instance.

        On triangular lattice, the nearest-neighbor (NN) bonds along the
        zero-degree direction is defined as the x-type bond (x-bond); NN bonds
        along the 120-degree direction is defined as the y-type bond (y-bond);
        NN bonds along the 60-degree direction is defined as the z-type bond
        (z-bond). The definition of the x, y, z bond is counterclockwise.

        Parameters
        ----------
        num1 : int
            The number of lattice site along the 1st translation vector.
        num2 : int or None, optional
            The number of lattice site along the 2nd translation vector.
            The default value `None` implies that `num2` takes the same value
            as `num1`.
        which : ["xy" | "yz" | "zx" | "yx" | "zy" | "xz"], optional
            Define the direction of the cluster. This parameter determine the
            interpretation of the `num1` and `num2` parameters. For example,
            if `which` is set to "xy", then there are `num1` lattice sites
            along the x-bond direction and `num2` lattice sites along the
            y-bond direction.
            Default: "xz".
        """

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
        cluster = Lattice(points=points, vectors=vectors, name=which)
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
            # The definition of x, y, z bond in a trio is counterclockwise.
            if azimuth in (-180, 0, 180):
                x_bonds.append(bond_index)
            elif azimuth in (-120, 60):
                z_bonds.append(bond_index)
            elif azimuth in (-60, 120):
                y_bonds.append(bond_index)
            else:
                raise ValueError("Invalid bond azimuth: {0}".format(azimuth))

        self._num1 = num1
        self._num2 = num2
        self._which = which
        self._cluster = cluster
        self._x_bonds = tuple(x_bonds)
        self._y_bonds = tuple(y_bonds)
        self._z_bonds = tuple(z_bonds)

    @property
    def num1(self):
        """
        The `num1` attribute.
        """

        return self._num1

    @property
    def num2(self):
        """
        The `num2` attribute.
        """

        return self._num2

    @property
    def which(self):
        """
        The `which` attribute.
        """

        return self._which

    @property
    def identity(self):
        """
        A identity of the cluster.
        """

        return "num1={0}_num2={1}_which={2}".format(
            self._num1, self._num2, self._which
        )

    @property
    def site_num(self):
        """
        The `site_num` attribute.
        """

        return self._num1 * self._num2

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

    def ShowNNBonds(self):
        """
        Show nearest neighbor bonds.
        """

        fig, ax = plt.subplots()
        intra, inter = self.cluster.bonds(nth=1)
        for ls, bonds in [("solid", intra), ("dashed", inter)]:
            for bond in bonds:
                (x0, y0), (x1, y1) = bond.endpoints
                azimuth = bond.getAzimuth(ndigits=0)
                if azimuth in (-180, 0, 180):
                    color = "tab:red"
                elif azimuth in (-120, 60):
                    color = "tab:green"
                elif azimuth in (-60, 120):
                    color = "tab:blue"
                else:
                    raise ValueError("Invalid bond azimuth: {0}".format(azimuth))
                ax.plot([x0, x1], [y0, y1], ls=ls, color=color)

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                dR = np.dot([i, j], self._cluster.vectors)
                points = self.cluster.points + dR
                ax.plot(points[:, 0], points[:, 1], ls="", marker="o")

        ax.set_axis_off()
        ax.set_aspect("equal")
        try:
            plt.get_current_fig_manager().window.showMaximized()
        except Exception:
            pass
        plt.show()
        plt.close("all")


def derivation(xs, ys, nth=1):
    """
    Calculate the nth derivatives of `ys` versus `xs` discretely.

    The derivatives are calculated using the following formula:
        dy / dx = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])

    Parameters
    ----------
    xs : 1-D array
        The independent variables.
        `xs` is assumed to be sorted in ascending order and there are no
        identical values in `xs`.
    ys : 1-D array
        The dependent variables.
        `ys` should be of the same length as `xs`.
    nth : int, optional
        The nth derivatives.
        Default: 1.

    Returns
    -------
    xs : 1-D array
        The independent variables.
    ys : 1-D array
        The nth derivatives corresponding to the returned `xs`.
    """

    assert isinstance(nth, int) and nth >= 0
    assert isinstance(xs, np.ndarray) and xs.ndim == 1
    assert isinstance(ys, np.ndarray) and ys.shape == xs.shape

    for i in range(nth):
        ys = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        xs = (xs[1:] + xs[:-1]) / 2
    return xs, ys


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


@jit(
    void(complex128[:], complex128[:], complex128[:]),
    nopython=True, cache=True
)
def mykron0(a, b, out):
    num_a = a.shape[0]
    num_b = b.shape[0]
    for i in range(num_a):
        ai = a[i]
        index = i * num_b
        for j in range(num_b):
            out[index + j] = ai * b[j]


@jit(complex128[:](complex128[:], complex128[:]), nopython=True, cache=True)
def mykron1(a, b):
    num_a = a.shape[0]
    num_b = b.shape[0]
    out = np.zeros(num_a * num_b, dtype=np.complex128)
    for i in range(num_a):
        ai = a[i]
        index = i * num_b
        for j in range(num_b):
            out[index + j] = ai * b[j]
    return out


if __name__ == "__main__":
    num1 = num2 = 10
    site_num = num1 * num2
    lattice = TriangularLattice(num1, num2)
    lattice.ShowNNBonds()

    vectors2D = np.random.random((site_num, 2))
    vectors3D = np.random.random((site_num, 3))
    vectors2D = vectors2D / np.linalg.norm(vectors2D, axis=1, keepdims=True)
    vectors3D = vectors3D / np.linalg.norm(vectors3D, axis=1, keepdims=True)

    fig = plt.figure("VectorField2D")
    ax = fig.add_subplot(111)
    ShowVectorField2D(
        ax, lattice.cluster.points, vectors2D,
        markersize=8, title="VectorField2D"
    )
    ax.set_axis_off()

    fig = plt.figure("VectorField3D")
    ax = fig.add_subplot(111, projection="3d")
    ShowVectorField3D(
        ax, lattice.cluster.points, vectors3D,
        markersize=8, title="VectorField3D"
    )
    ax.set_zlim(-0.5, 0.5)

    plt.show()
    plt.close("all")
