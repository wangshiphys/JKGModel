"""
This module provides a base class which describe a triangular cluster.
"""


from HamiltonianPy import lattice_generator


class TriangularCluster:
    """
    Base class of triangular cluster on which the J-K-Gamma-Gamma' model is
    defined.
    """

    # Default model parameters
    # "alpha" and "beta" for the J-K-Gamma model
    # J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
    # K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
    # G = np.cos(alpha * np.pi)
    # "J", "K", "G" and "GP" for the J-K-Gamma-Gamma' model.
    # The default model parameters correspond to ferromagnetic Heisenberg model
    DEFAULT_MODEL_PARAMETERS = {
        "alpha": 0.5, "beta": -0.5,
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
            Default: numy = numx.
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


if __name__ == "__main__":
    cluster = TriangularCluster(6, 6)
    print("numx = {0}".format(cluster.numx))
    print("numy = {0}".format(cluster.numy))
    print("site_num = {0}".format(cluster.site_num))
    print("x_bond_num = {0}".format(cluster.x_bond_num))
    print("y_bond_num = {0}".format(cluster.y_bond_num))
    print("z_bond_num = {0}".format(cluster.z_bond_num))
