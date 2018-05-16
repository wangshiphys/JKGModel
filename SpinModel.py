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
    Exact diagonalizing of the J-K-Gamma model on triangular lattice
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
            default: numy = numx
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
        alphas = np.array([[0, 180], [-120, 60], [-60, 120]])
        for bond in intra+inter:
            p0, p1 = bond.getEndpoints()
            azimuth = bond.getAzimuth()
            judge = np.any(np.abs(azimuth - alphas) < atol, axis=1)

            index0 = cluster.getIndex(site=p0, fold=True)
            index1 = cluster.getIndex(site=p1, fold=True)
            bond_index = (index0, index1)

            if judge[0]:
                xbonds.append(bond_index)
            elif judge[1]:
                ybonds.append(bond_index)
            elif judge[2]:
                zbonds.append(bond_index)
            else:
                raise ValueError("Invalid bond direction!")

        return xbonds, ybonds, zbonds

    def _TermMatrix(self):
        # Calculate the matrix representation of the J, K, Gamma term
        # respectively. Save the matrix to temp file.
        # If the the matrix already exists on the file system, then load them
        # instead of calculating them.

        tmp_dir = "tmp/"
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        name_template = tmp_dir + "TRIANGLE_NUMX_{0}_NUMY_{1}_H{2}.npz"
        name_HJ = name_template.format(self.numx, self.numy, "J")
        name_HK = name_template.format(self.numx, self.numy, "K")
        name_HG = name_template.format(self.numx, self.numy, "G")

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
            for (gamma, alpha, beta), bonds in zip(configs, all_bonds):
                for index0, index1 in bonds:
                    SKM = SpinInteraction.matrixFunc(
                        [(index0, gamma), (index1, gamma)], site_num
                    )
                    HK += SKM
                    HJ += SKM
                    HJ += SpinInteraction.matrixFunc(
                        [(index0, alpha), (index1, alpha)], site_num
                    )
                    HJ += SpinInteraction.matrixFunc(
                        [(index0, beta), (index1, beta)], site_num
                    )
                    HG += SpinInteraction.matrixFunc(
                        [(index0, alpha), (index1, beta)], site_num
                    )
                    HG += SpinInteraction.matrixFunc(
                        [(index0, beta), (index1, alpha)], site_num
                    )
            save_npz(name_HJ, HJ, compressed=False)
            save_npz(name_HK, HK, compressed=False)
            save_npz(name_HG, HG, compressed=False)

        # Caching these matrices in the SpinModelSolver instance for reuse
        self._cache = (HJ, HK, HG)

    def GSE(self, J=-1.0, K=0.0, G=0.0, *, tol=0.0):
        """
        Calculate the ground state energy of the model Hamiltonian

        Parameter
        ---------
        J : float, optional
            The coefficient of the Heisenberg term
            default: -1.0
        K : float, optional
            The coefficient of the Kitaev term
            default: 0.0
        G : float, optional
            The coefficient of the Gamma term
            default: 0.0
        tol : float, optional
            Relative accuracy for eigenvalues (stop criterion). The default
            value 0 implies machine precision
            default: 0.0

        Returns
        -------
        dt : float
            The time spend on calculating the ground state energy
        gse : float
            The pre-site ground state energy
        """

        if not hasattr(self, "_cache"):
            self._TermMatrix()
        HJ, HK, HG = self._cache
        H = J * HJ + K * HK + G * HG

        t0 = time()
        gse = eigsh(H, k=1, which="SA", return_eigenvectors=False, tol=tol)
        t1 = time()

        return t1 - t0, gse[0] / self.site_num

    __call__ = GSE


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

    ax_Es = ax
    ax_d2Es = ax.twinx()

    alphas_new, d2Es = derivation(alphas, Es, nth=2)
    d2Es = -d2Es / (np.pi ** 2)
    line_Es, = ax_Es.plot(alphas, Es, color=color0)
    line_d2Es, = ax_d2Es.plot(alphas_new, d2Es, color=color1)

    ax_Es.set_ylabel('E', color=color0, fontdict=fontdict)
    ax_Es.tick_params('y', colors=color0)
    ax_d2Es.set_ylabel(r"$-d^2E/d\alpha^2$", color=color1, fontdict=fontdict)
    ax_d2Es.tick_params('y', colors=color1)

    ax_Es.set_title(
        r"$E\ and\ -\frac{d^2E}{d\alpha^2}\ vs\ \alpha$", fontdict=fontdict
    )
    ax_Es.set_xlim(alphas[0], alphas[-1])
    ax_Es.set_xlabel(r"$\alpha(/\pi)$")
    ax_Es.legend(
        (line_Es, line_d2Es),
        ('E', r"$-\frac{d^2E}{d\alpha^2}$"),
        loc = 0
    )


def ArgParser():
    """
    Parse the command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Parse command line arguments."
    )

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

    template = "{0}_numx_{1}_numy_{2}_step_{3:.3f}_beta_{4:.3f}.{5}"
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
    else:
        alphas = np.arange(0, 2.2, step=step)
        Es = np.zeros(alphas.shape) + np.nan
        res = np.array([alphas, Es])
    num = res.shape[1]
    return num, res


def GlobalPhaseDiagram(numx=3, numy=4, step=0.01, beta=0.5):
    now_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    header = "Program start running at: " + now_time + "\n" + "#" * 80 + "\n\n"
    log_template = "The {ith}th alpha\nThe current alpha: {alpha:.3f}\n"
    log_template += "The ground state energy: {gse}\n"
    log_template += "The time spend on the GSE: {dt}s\n" + "=" * 80 + "\n"

    log_file, data_file, fig_file = FileNames(numx, numy, step, beta)
    solver = SpinModelSolver(numx=numx, numy=numy)
    num_alpha, res = BreakPointRecovery(data_file, step)
    Js = np.sin(beta * np.pi) * np.sin(res[0] * np.pi)
    Ks = np.sin(beta * np.pi) * np.cos(res[0] * np.pi)
    G = np.cos(beta * np.pi)

    fp = open(log_file, mode="a", buffering=1)
    fp.write(header)
    for i in range(num_alpha):
        if np.isnan(res[1, i]):
            dt, res[1, i] = solver.GSE(J=Js[i], K=Ks[i], G=G, tol=1e-10)
            np.save(data_file, res)
            msg = log_template.format(
                ith=i, alpha=res[0, i], gse=res[1, i], dt=dt
            )
            fp.write(msg)
    fp.close()

    fig, ax = plt.subplots()
    visualization(ax, res[0], res[1])
    fig.savefig(fig_file)
    plt.close("all")


if __name__ == "__main__":
    # numx, numy, step, beta = ArgParser()
    numx, numy, step, beta = 3, 4, 0.005, 0.5
    GlobalPhaseDiagram(numx=numx, numy=numy, step=step, beta=beta)