"""
This module provides utility programs used in this project.
"""


__all__ = ["TriangularLattice"]

import matplotlib.pyplot as plt
import numpy as np

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

    def __init__(self, num1=4, num2=6, direction="xy"):
        """
        On triangular lattice, the nearest-neighbor (NN) bonds along the
        zero-degree direction is defined as the x-type bond (x-bond); NN bonds
        along the 120-degree direction is defined as the y-type bond (y-bond);
        NN bonds along the 60-degree direction is defined as the z-type bond
        (z-bond). The definition of the x, y, z bond is counterclockwise.

        Parameters
        ----------
        num1, num2 : int, optional
            The number of lattice site along the 1st and 2nd translation vector.
            The default values for `num1` and `num2` are 4 and 6 respectively.
        direction : ["xy" | "xz" | "yx" | "yz" | "zx" | "zy"], optional
            Define the direction of the cluster. This parameter determine the
            interpretation of the `num1` and `num2` parameters. For example,
            if `direction` is set to "xy", then there are `num1` lattice sites
            along the x-bond direction and `num2` lattice sites along the
            y-bond direction.
            Default: "xy".
        """

        assert isinstance(num1, int) and num1 > 0
        assert isinstance(num2, int) and num2 > 0
        assert direction in ("xy", "xz", "yx", "yz", "zx", "zy")

        identity = "num1={0}_num2={1}_direction={2}".format(
            num1, num2, direction
        )

        RX = np.array([1.0, 0.0], dtype=np.float64)
        RY = np.array([-0.5, np.sqrt(3) / 2], dtype=np.float64)
        RZ = np.array([0.5, np.sqrt(3) / 2], dtype=np.float64)
        AS = {
            "xy": np.array([RX, RY], dtype=np.float64),
            "xz": np.array([RX, RZ], dtype=np.float64),
            "yx": np.array([RY, RX], dtype=np.float64),
            "yz": np.array([RY, RZ], dtype=np.float64),
            "zx": np.array([RZ, RX], dtype=np.float64),
            "zy": np.array([RZ, RY], dtype=np.float64),
        }

        As = AS[direction]
        vectors = As * np.array([[num1], [num2]])
        points = np.dot([[i, j] for i in range(num1) for j in range(num2)], As)
        cluster = Lattice(points=points, vectors=vectors, name=identity)
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
                raise RuntimeError("Invalid bond azimuth: {0}".format(azimuth))

        self._num1 = num1
        self._num2 = num2
        self._direction = direction
        self._identity = identity
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
    def direction(self):
        """
        The `direction` attribute.
        """

        return self._direction

    @property
    def identity(self):
        """
        The identity of the cluster.
        """

        return self._identity

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

    def ShowNNBonds(self, lw=2.0, ms=6.0):
        """
        Show nearest neighbor bonds.

        Parameters
        ----------
        lw : float, optional
            The line width of the bond.
            Default: 2.0.
        ms : float, optional
            The size of the point.
            Default: 6.0.
        """

        fig, ax = plt.subplots(num="NNBonds")
        intra, inter = self._cluster.bonds(nth=1)
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
                    raise RuntimeError(
                        "Invalid bond azimuth: {0}".format(azimuth)
                    )
                ax.plot([x0, x1], [y0, y1], ls=ls, lw=lw, color=color)

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                dR = np.dot([i, j], self._cluster.vectors)
                points = self._cluster.points + dR
                ax.plot(points[:, 0], points[:, 1], ls="", marker="o", ms=ms)

        ax.set_axis_off()
        ax.set_aspect("equal")
        try:
            plt.get_current_fig_manager().window.showMaximized()
        except Exception:
            pass
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    directions = ("xy", "xz", "yx", "yz", "zx", "zy")
    for direction in directions:
        lattice = TriangularLattice(direction=direction)
        lattice.ShowNNBonds()
