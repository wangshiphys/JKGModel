from datetime import datetime
from pathlib import Path
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigsh
from time import time

import argparse
import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy.lattice import Lattice
from HamiltonianPy.termofH import SpinInteraction
from LatticeInfo import AS


class SpinModelSolver:
    """
    Exact diagonalization of the J-K-Gamma model on triangular lattice
    """

    def __init__(self, numx=4, numy=None):
        """
        Customize the newly created instance

        Parameters
        ----------
        numx : int, optional
            The number of lattice sites along the first translation vector
            default: 4
        numy : int, optional
            The number of lattice sites along the second translation vector
            default: numx = numy
        """

        if not isinstance(numx, int) or numx < 1:
            raise ValueError("The `numx` parameter should be positive integer!")
        if not isinstance(numy, int):
            if numy is None:
                numy = numx
            else:
                raise ValueError(
                    "The `numy` parameter should be positive integer or None!"
                )
        elif numy < 1:
            raise ValueError(
                "The `numy` parameter should be positive integer or None!"
            )

        self.numx = numx
        self.numy = numy
        self.site_num = numx * numy

    def _bonds(self):
        # Generate all nearest neighbor bonds
        # Categorize the nearest neighbor bonds according to their direction

        sites = np.matmul(
            [[x, y] for x in range(self.numx) for y in range(self.numy)], AS
        )
        tvs = AS * np.array([[self.numx], [self.numy]])

        cluster = Lattice(points=sites, tvs=tvs)
        intra, inter = cluster.bonds(nth=1)

        atol = 1e-3
        xbonds = []
        ybonds = []
        zbonds = []
        for bond in intra+inter:
            p0, p1 = bond.getEndpoints()
            azimuth = bond.getAzimuth()

            index0 = cluster.getIndex(site=p0, fold=True)
            index1 = cluster.getIndex(site=p1, fold=True)
            bond_index = (index0, index1)

            if np.abs(azimuth) < atol or np.abs(azimuth - 180) < atol:
                xbonds.append(bond_index)
            elif np.abs(azimuth - 60) < atol or np.abs(azimuth + 120) < atol:
                ybonds.append(bond_index)
            elif np.abs(azimuth - 120) < atol or np.abs(azimuth + 60) < atol:
                zbonds.append(bond_index)
            else:
                raise ValueError("Invalid bond direction!")

        return tuple(xbonds), tuple(ybonds), tuple(zbonds)

    def _TermMatrix(self):
        # Calculate the matrix representation of the J, K, Gamma term
        # respectively. Save the matrix to temp file.
        # If the the matrix already exists on the file system, then load them
        # instead of calculating them.

        tmp_dir = "tmp/"
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        name_template = tmp_dir + "H{0}_numx_{1}_numy_{2}.npz"
        name_HJ = name_template.format('J', self.numx, self.numy)
        name_HK = name_template.format('K', self.numx, self.numy)
        name_HG = name_template.format('G', self.numx, self.numy)

        if Path(name_HJ).exists() and Path(name_HK).exists() and Path(
                name_HG).exists():
            HJ = load_npz(name_HJ)
            HK = load_npz(name_HK)
            HG = load_npz(name_HG)
        else:
            all_bonds = self._bonds()
            configs = [('x', 'y', 'z'), ('y', 'z', 'x'), ('z', 'x', 'y')]
            site_num = self.site_num

            HJ = HK = HG = 0.0
            for (K, G0, G1), bonds in zip(configs, all_bonds):
                for index0, index1 in bonds:
                    SKM = SpinInteraction.matrixFunc(
                        [(index0, K), (index1, K)], site_num
                    )
                    HK += SKM
                    HJ += SKM
                    HJ += SpinInteraction.matrixFunc(
                        [(index0, G0), (index1, G0)], site_num
                    )
                    HJ += SpinInteraction.matrixFunc(
                        [(index0, G1), (index1, G1)], site_num
                    )
                    HG += SpinInteraction.matrixFunc(
                        [(index0, G0), (index1, G1)], site_num
                    )
                    HG += SpinInteraction.matrixFunc(
                        [(index0, G1), (index1, G0)], site_num
                    )
            save_npz(name_HJ, HJ, compressed=False)
            save_npz(name_HK, HK, compressed=False)
            save_npz(name_HG, HG, compressed=False)

        # Caching these matrix in the SpinModelSolver instance
        self._HJ = HJ
        self._HK = HK
        self._HG = HG

    def GSE(self, alpha=0.0, beta=0.0, tol=0.0):
        """
        Calculate the ground state energy of the model Hamiltonian

        Parameter
        ---------
        alpha : float, optional
            Parameter that determines the value the J, K, G coefficients
            default: 0.0
        beta : float, optional
            Parameter that determines the value the J, K, G coefficients
            default: 0.0
        tol : float, optional
            Relative accuracy for eigenvalues (stop criterion). The default
            value 0 implies machine precision
            default: 0.0

        J = np.sin(beta) * np.sin(alpha)
        K = np.sin(beta) * np.cos(alpha)
        G = np.cos(beta)

        alpha in the range [-pi, pi)
        beta in the range[0, pi)

        Returns
        -------
        dt : float
            The time spend on calculating the ground state energy
        gse : float
            The ground state energy
        """

        J = np.sin(beta) * np.sin(alpha)
        K = np.sin(beta) * np.cos(alpha)
        G = np.cos(beta)

        if not hasattr(self, "_HJ"):
            self._TermMatrix()
        H = J * self._HJ + K * self._HK + G * self._HG

        t0 = time()
        gse = eigsh(H, k=1, which="SA", return_eigenvectors=False, tol=tol)
        t1 = time()

        return t1 - t0, gse[0] / self.site_num


def derivation(xs, ys, nth=1):
    """
    Calculate the n-th derivatives of `ys` versus `xs` discretely

    Parameters
    ----------
    xs : 1-D array
        The independent variables
    ys : 1-D array
        The dependent variables
    nth : int, optional
        The n-th derivatives
        default: 1

    Returns
    -------
    xs : 1-D array
        The independent variables
    ys : 1-D array
        The n-th derivatives corresponding the returned xs
    """

    if not isinstance(nth, int) or nth < 0:
        raise ValueError("The `nth` parameter should be non-negative integer!")

    for i in range(nth):
        ys = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        xs = (xs[1:] + xs[:-1]) / 2
    return xs, ys


def visualization(ax, alphas, Es):
    """
    Plot Es versus alphas as well as the second derivatives of Es versus alphas.
    """

    color0 = "#3465A4"
    color1 = "#F57900"
    fontdict = {"fontsize": 12}

    ax_left = ax
    ax_right = ax.twinx()

    alphas_new, d2Es = derivation(alphas, Es, nth=2)
    d2Es = -d2Es / (np.pi ** 2)
    line_left, = ax_left.plot(alphas, Es, color=color0)
    line_right, = ax_right.plot(alphas_new, d2Es, color=color1)

    ax_left.set_ylabel('E', color=color0, fontdict=fontdict)
    ax_left.tick_params('y', colors=color0)
    ax_right.set_ylabel(r"$-d^2E/d\alpha^2$", color=color1, fontdict=fontdict)
    ax_right.tick_params('y', colors=color1)

    ax_left.set_title(
        r"$E\ and\ -\frac{d^2E}{d\alpha^2}\ vs\ \alpha$", fontdict=fontdict
    )
    ax_left.set_xlim(alphas[0], alphas[-1])
    ax_left.set_xlabel(r"$\alpha(/\pi)$")
    ax_left.legend(
        (line_left, line_right),
        ('E', r"$-\frac{d^2E}{d\alpha^2}$"),
        loc = 0
    )


def ArgParser():
    """
    Parse the command lin arguments
    """

    parser = argparse.ArgumentParser(description="Parse command line argument.")
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="The Gamma term parameter (Default: %(default)s)."
    )
    parser.add_argument(
        "--step", type=float, default=0.01,
        help="The step of alphas (Default: %(default)s)."
    )
    parser.add_argument(
        "--numx", type=int, default=4,
        help="The number of sites along a0 direction (Default: %(default)s)."
    )
    parser.add_argument(
        "--numy", type=int, default=4,
        help="The number of sites along a1 direction (Default: %(default)s)."
    )

    args = parser.parse_args()
    return args.numx, args.numy, args.step, args.beta


def FileNames(numx, numy, step, beta):
    """
    Return the file names for log file, data file and figure file
    """

    log_dir = "log/SpinModel/"
    data_dir = "data/SpinModel/"
    fig_dir = "fig/SpinModel/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    template = "{0}_numx_{1}_numy_{2}_step_{3:.3f}_beta_{4:.2f}.{5}"
    log_file = log_dir + template.format("Log", numx, numy, step, beta, "txt")
    data_file = data_dir + template.format("GSE", numx, numy, step, beta, "npy")
    fig_file = fig_dir + template.format("GSE", numx, numy, step, beta, "png")
    return log_file, data_file, fig_file


def BreakPointRecovery(data_file, step):
    """
    Recovery the program state from break point
    """

    if Path(data_file).exists():
        res = np.load(data_file)
        tmp = np.isnan(res[1])
        if np.any(tmp):
            start = list(tmp).index(True)
        else:
            start = res.shape[1]
    else:
        alphas = np.arange(0, 2.2, step=step)
        Es = np.zeros(alphas.shape) + np.nan
        res = np.array([alphas, Es])
        start = 0
    stop = res.shape[1]
    return start, stop, res


if __name__ == "__main__":
    start_time = "Program start running at: {0:%Y-%m-%d %H:%M:%S}"
    start_time = start_time.format(datetime.now())

    # numx, numy, step, beta = ArgParser()
    numx, numy, step, beta = 3, 4, 0.005, 0.5
    log_file, data_file, fig_file = FileNames(numx, numy, step, beta)

    start, stop, res = BreakPointRecovery(data_file, step)

    fp = open(log_file, mode='a', buffering=1)
    fp.write(start_time + '\n' + '#' * 80 + "\n\n")

    beta_pi = beta * np.pi
    solver = SpinModelSolver(numx=numx, numy=numy)
    for i in range(start, stop):
        dt, res[1, i] = solver.GSE(alpha=res[0, i]*np.pi, beta=beta_pi, tol=0.0)
        np.save(data_file, res)

        msg = "The {0}th alpha\n".format(i)
        msg += "The current alpha: {0:.3f}\n".format(res[0, i])
        msg += "The ground state energy: {0}\n".format(res[1, i])
        msg += "The time spend on the GSE: {0}s\n".format(dt)
        msg += '=' * 80 + '\n'
        fp.write(msg)
    fp.close()

    fig, ax = plt.subplots()
    visualization(ax, res[0], res[1])
    fig.savefig(fig_file)
    # plt.close("all")
    plt.show()