"""
Exact diagonalization of the J-K-Gamma model on triangular lattice
"""


from pathlib import Path
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigsh
from time import time

import numpy as np

from HamiltonianPy.lattice import Lattice
from HamiltonianPy.termofH import SpinInteraction
from LatticeData import AS


class JKGModelSolver:
    """
    Exact diagonalization of the J-K-Gamma spin model on triangular lattice

    Attributes
    ----------
    numx: int
        The number of lattice site along the first translation vector
    numy: int
        The number of lattice site along the second translation vector
    site_num: int
        The number of lattice site of the system
    """

    def __init__(self, numx, numy=None):
        """
        Customize the newly created instance

        Parameters
        ----------
        numx : int
            The number of lattice site along the first translation vector
        numy : int, optional
            The number of lattice site along the second translation vector
            default: numy = numx
        """

        assert isinstance(numx, int) and numx > 0
        assert (numy is None) or (isinstance(numy, int) and numy > 0)

        if numy is None:
            numy = numx
        self._numx = numx
        self._numy = numy
        self._site_num = numx * numy

    @property
    def numx(self):
        """
        The `numx` attribute
        """

        return self._numx

    @property
    def numy(self):
        """
        The `numy` attribute
        """

        return self._numy

    @property
    def site_num(self):
        """
        The `site_num` attribute
        """

        return self._site_num

    def _bonds(self):
        # Generate all nearest neighbor bonds
        # Categorize the nearest neighbor bonds according to their direction

        points = np.matmul(
            [[x, y] for x in range(self._numx) for y in range(self._numy)], AS
        )
        vectors = AS * np.array([[self._numx], [self._numy]])

        cluster = Lattice(points=points, vectors=vectors)
        intra, inter = cluster.bonds(nth=1)

        x_bonds = []
        y_bonds = []
        z_bonds = []
        for bond in intra + inter:
            p0, p1 = bond.getEndpoints()
            index0 = cluster.getIndex(site=p0, fold=True)
            index1 = cluster.getIndex(site=p1, fold=True)
            bond_index = (index0, index1)

            azimuth = int(bond.getAzimuth(ndigits=0))
            if azimuth in (-180, 0):
                x_bonds.append(bond_index)
            elif azimuth in (-120, 60):
                y_bonds.append(bond_index)
            elif azimuth in (-60, 120):
                z_bonds.append(bond_index)
            else:
                raise ValueError("Invalid bond direction!")

        return x_bonds, y_bonds, z_bonds

    def _TermMatrix(self):
        # Calculate the matrix representation of the J, K, Gamma term
        # Save these matrices on the file system
        # If the the matrix already exists on the file system, then load them
        # instead of calculating them

        directory = Path("tmp")
        name_template = "TRIANGLE_NUMX={0}_NUMY={1}_H{{0}}.npz".format(
            self._numx, self._numy
        )
        # print(name_template)
        # The slash operator helps create child paths
        file_HJ = directory / name_template.format("J")
        file_HK = directory / name_template.format("K")
        file_HG = directory / name_template.format("G")

        if file_HJ.exists() and file_HK.exists() and file_HG.exists():
            HJ = load_npz(file_HJ)
            HK = load_npz(file_HK)
            HG = load_npz(file_HG)
        else:
            site_num = self._site_num
            all_bonds = self._bonds()
            configs = [("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y")]

            HJ = HK = HG = 0.0
            for (gamma, alpha, beta), bonds in zip(configs, all_bonds):
                for index0, index1 in bonds:
                    SKM = SpinInteraction.matrix_function(
                        [(index0, gamma), (index1, gamma)], site_num
                    )
                    HK += SKM
                    HJ += SKM
                    HJ += SpinInteraction.matrix_function(
                        [(index0, alpha), (index1, alpha)], site_num
                    )
                    HJ += SpinInteraction.matrix_function(
                        [(index0, beta), (index1, beta)], site_num
                    )
                    HG += SpinInteraction.matrix_function(
                        [(index0, alpha), (index1, beta)], site_num
                    )
                    HG += SpinInteraction.matrix_function(
                        [(index0, beta), (index1, alpha)], site_num
                    )
            directory.mkdir(parents=True, exist_ok=True)
            save_npz(file_HJ, HJ, compressed=False)
            save_npz(file_HK, HK, compressed=False)
            save_npz(file_HG, HG, compressed=False)

        # Caching these matrices for reuse
        self._cache = (HJ, HK, HG)

    def GS(self, alpha=0.5, beta=-0.5, tol=0, path=None, save_data=True):
        """
        Calculate the ground state energy and vector of the model Hamiltonian

        Parameters
        ----------
        alpha, beta : float, optional
            Model parameters
            The default values for alpha and beta are 0.5 and -0.5 respectively.

            J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
            K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
            G = np.cos(alpha * np.pi)

            J is the coefficient of the Heisenberg term
            K is the coefficient of the Kitaev term
            G is the coefficient of the Gamma term
        tol : float, optional
            Relative accuracy for eigenvalues (stop criterion)
            The default value 0 implies machine precision.
        path : str, optional
            If the ground state data for the given `alpha` and `beta` is
            already exist, then `path` specify where to load the data;
            If the ground state data does not exist, then `path` specify
            where to save the result.
            The default value `None` implies 'data/SpinModel/' relative to
            the working directory.
        save_data : boolean, optional
            Whether to save the ground state energy and vector to file system
            default: True

        Returns
        -------
        alpha : float
            The current alpha parameter
        beta : float
            The current beta parameter
        gse : float
            The ground state energy
        ket : array
            The ground state vector
        """

        data_dir = Path("data/SpinModel/") if path is None else Path(path)

        data_name = "GS_numx={0}_numy={1}_alpha={2:.3f}_beta={3:.3f}.npz"
        full_name = data_dir / data_name.format(
            self._numx, self._numy, alpha, beta
        )

        if full_name.exists():
            with np.load(full_name) as fp:
                alpha, beta = fp["parameters"]
                gse = fp["gse"]
                ket = fp["ket"]
        else:
            J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
            K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
            G = np.cos(alpha * np.pi)

            if not hasattr(self, "_cache"):
                self._TermMatrix()
            HJ, HK, HG = self._cache

            gse, ket = eigsh(
                J * HJ + K * HK + G * HG, k=1, which="SA", tol=tol
            )

            if save_data:
                data_dir.mkdir(parents=True, exist_ok=True)
                np.savez(full_name, parameters=[alpha, beta], gse=gse, ket=ket)
        return alpha, beta, gse[0], ket