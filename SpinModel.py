"""
Exact diagonalization of the J-K-Gamma model on triangular lattice
"""


from pathlib import Path
from time import time

from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigsh

import argparse
import logging

import numpy as np

from HamiltonianPy.lattice import Lattice
from HamiltonianPy.termofH import SpinInteraction
from ProjectDataBase.LatticeData import AS


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

        directory = Path("tmp/")
        name_template = "TRIANGLE_NUMX={0}_NUMY={1}_H{{0}}.npz".format(
            self._numx, self._numy
        )
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

    def GS(self, alpha=0.5, beta=-0.5, tol=0, gs_dir=None):
        """
        Calculate the ground state energy and vector of the model Hamiltonian

        This method first check whether the ground state data for the given
        `alpha` and `beta` exists in the given `gs_dir`:
            1. If exist, then stop;
            2. If the requested data does not exist in the given `gs_dir`:
                2.1 Calculate the ground state data for the given `alpha`
                and `beta`;
                2.2 Save the ground state data to the given `gs_dir` with
                name:
                "GS_numx={0}_numy={1}_alpha={2:.3f}_beta={3:.3f}.npz".format(
                    self.numx, self.numy, alpha, beta
                )

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
        gs_dir : str, optional
            If the ground state data for the given `alpha` and `beta` is
            already exist, then `gs_dir` specify where to load the data;
            If the ground state data does not exist, then `gs_dir` specify
            where to save the result.
            The default value `None` implies
                'data/SpinModel/GS/alpha={0:.3f}/'.format(alpha)
             relative to the current working directory.
        """

        if gs_dir is None:
            gs_dir = "data/SpinModel/GS/alpha={0:.3f}/".format(alpha)
        gs_dir = Path(gs_dir)
        gs_name_template = "GS_numx={0}_numy={1}_alpha={2:.3f}_beta={3:.3f}.npz"
        gs_full_name = gs_dir / gs_name_template.format(
            self._numx, self._numy, alpha, beta
        )

        if not gs_full_name.exists():
            alpha_pi, beta_pi = np.pi * alpha, np.pi * beta
            J = np.sin(alpha_pi) * np.sin(beta_pi)
            K = np.sin(alpha_pi) * np.cos(beta_pi)
            G = np.cos(alpha_pi)

            if not hasattr(self, "_cache"):
                self._TermMatrix()
            HJ, HK, HG = self._cache

            gse, ket = eigsh(
                J * HJ + K * HK + G * HG, k=1, which="SA", tol=tol
            )

            gs_dir.mkdir(parents=True, exist_ok=True)
            np.savez(gs_full_name, parameters=[alpha, beta], gse=gse, ket=ket)


def main(fixed_param, fixed_which="alpha", step=0.01, numx=3, numy=4,
         log_dir=None, **kwargs):
    """
    The entrance for calculating the ground state versus model parameters

    Parameters
    ----------
    fixed_which : str, optional
        Which model parameter is fixed
        Valid value are "alpha" or "beta"
        default : "alpha"
    fixed_param: float
        The value of the fixed model parameter
    step : float, optional
        The step of the non-fixed model parameter
        If fixed_which is "alpha", betas = np.arange(0, 2, step);
        If fixed_which is "beta", alphas = np.arange(0, 1+step/2, step).
        default: 0.01
    numx: int, optional
        The number of lattice site along the first translation vector.
        default: 3
    numy: int
        The number of lattice site along the second translation vector
        `numx` and `numy` are used to construct the JKGModelSolver instance
        default: 4
    log_dir : str, optional
        Where to save the log information
        The log information will be saved to `gs_dir` with name:
            "Log_numx={0}_numy={1}_{2}={3:.3f}.log".format(
                numx, numy, fixed_which, fixed_param
            )
        The default value `None` implies `log/SpinModel/` relative to the
        current working directory.
    **kwargs
        Other keyword arguments are passed to the `GS` method of JKGModelSolver
    """

    assert fixed_which in ("alpha", "beta")

    if log_dir is None:
        log_dir = "log/SpinModel/"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name_template = "Log_numx={0}_numy={1}_{2}={3:.3f}.log"
    log_full_name = log_dir / log_name_template.format(
        numx, numy, fixed_which, fixed_param
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.FileHandler(log_full_name)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Program start running")
    solver = JKGModelSolver(numx=numx, numy=numy)
    if fixed_which == "alpha":
        betas = np.arange(0, 2, step)
        alphas = np.zeros(betas.shape, dtype=np.float64) + fixed_param
    else:
        alphas = np.arange(0, 1 + step/2, step)
        betas = np.zeros(alphas.shape, dtype=np.float64) + fixed_param

    msg_template = "alpha={0:.3f}, beta={1:.3f}, dt={2:.3f}s"
    for alpha, beta in zip(alphas, betas):
        t0 = time()
        solver.GS(alpha=alpha, beta=beta, **kwargs)
        t1 = time()
        msg = msg_template.format(alpha, beta, t1 - t0)
        logger.info(msg)
    logger.info("Program stop running")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command line arguments."
    )

    parser.add_argument(
        "--fixed_which", type=str, default="alpha",
        help="Set which model parameter to be fixed (Default: %(default)s).",
        choices=("alpha", "beta")
    )
    parser.add_argument(
        "--fixed_param", type=float, default=0.5,
        help="The value of the fixed model parameter (Default: %(default)s)."
    )
    parser.add_argument(
        "--numx", type=int, default=3,
        help="The number of lattice site along the 1st translation vector "
             "(Default: %(default)s)."
    )
    parser.add_argument(
        "--numy", type=int, default=4,
        help="The number of lattice site along the 2nd translation vector "
             "(Default: %(default)s)."
    )
    parser.add_argument(
        "--step", type=float, default=0.01,
        help="The step of variable model parameter (Default: %(default)s)."
    )
    parser.add_argument(
        "--tol", type=float, default=1e-8,
        help="The relative accuracy for eigenvalues (stop criterion) "
             "(Default: %(default)s)."
    )
    args = parser.parse_args()
    main(
        fixed_param=args.fixed_param,
        fixed_which=args.fixed_which,
        numx=args.numx,
        numy=args.numy,
        step=args.step,
        tol=args.tol,
    )
