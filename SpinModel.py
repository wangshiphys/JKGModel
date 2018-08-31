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


class JKGModelSolver:
    """
    Exact diagonalization of the J-K-Gamma spin model on triangular lattice

    Attributes
    ----------
    num1: int
        The number of lattice site along the first translation vector
    num2: int
        The number of lattice site along the second translation vector
    site_num: int
        The number of lattice site of the system
    """

    def __init__(self, num1, num2=None):
        """
        Customize the newly created instance

        Parameters
        ----------
        num1 : int
            The number of lattice site along the first translation vector
        num2 : int, optional
            The number of lattice site along the second translation vector
            default: num2 = num1
        """

        assert isinstance(num1, int) and num1 >= 1
        assert (num2 is None) or (isinstance(num2, int) and num2 >= 1)

        if num2 is None:
            num2 = num1
        self._num1 = num1
        self._num2 = num2
        self._site_num = num1 * num2

    @property
    def num1(self):
        """
        The `num1` attribute
        """

        return self._num1

    @property
    def num2(self):
        """
        The `num2` attribute
        """

        return self._num2

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
            [[x, y] for x in range(self._num1) for y in range(self._num2)], AS
        )
        vectors = AS * np.array([[self._num1], [self._num2]])

        cluster = Lattice(points=points, vectors=vectors)
        intra, inter = cluster.bonds(nth=1)

        bonds_x = []
        bonds_y = []
        bonds_z = []
        for bond in intra + inter:
            p0, p1 = bond.getEndpoints()
            index0 = cluster.getIndex(site=p0, fold=True)
            index1 = cluster.getIndex(site=p1, fold=True)
            bond_index = (index0, index1)

            azimuth = int(round(bond.getAzimuth()))
            if azimuth in (-180, 0):
                bonds_x.append(bond_index)
            elif azimuth in (-120, 60):
                bonds_y.append(bond_index)
            elif azimuth in (-60, 120):
                bonds_z.append(bond_index)
            else:
                raise ValueError("Invalid bond direction!")

        return bonds_x, bonds_y, bonds_z

    def _TermMatrix(self):
        # Calculate the matrix representation of the J, K, Gamma term
        # Save these matrices on the file system
        # If the the matrix already exists on the file system, then load them
        # instead of calculating them

        directory = Path("tmp")
        name_template = "TRIANGLE_NUM1={0}_NUM2={1}_H{{0}}.npz".format(
            self._num1, self._num2
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
            all_bonds = self._bonds()
            configs = [("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y")]
            site_num = self._site_num

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

    def GSE(self, J=1.0, K=0.0, G=0.0, tol=0.0):
        """
        Calculate the ground state energy of the model Hamiltonian

        Parameter
        ---------
        J : float, optional
            The coefficient of the Heisenberg term
            default: 1.0
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

        return t1 - t0, gse[0] / self._site_num

    __call__ = GSE


def derivation(xs, ys, nth=1):
    """
    Calculate the n-th derivatives of `ys` versus `xs` discretely

    Parameters
    ----------
    xs : 1-D array
        The independent variables
        `xs` is assumed to be sorted in ascending order
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

    assert isinstance(nth, int) and nth > 0

    xs = np.array(xs)
    ys = np.array(ys)
    for i in range(nth):
        # The following line calculate all the (ys[i]-ys[i-1])/(xs[i]-xs[i-1])
        # Equivalent to the following code:
        # dys = np.array([ys[i] - ys[i-1] for i in range(1, len(ys))])
        # dxs = np.array([xs[i] - xs[i-1] for i in range(1, len(xs))])
        # ys = dys / dxs
        ys = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        # Equivalent to the following code:
        # xs = np.array([(xs[i]+xs[i-1])/2 for i in range(1, len(xs))])
        xs = (xs[1:] + xs[:-1]) / 2
    return xs, ys


def ArgParser():
    """
    Parse the command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Parse command line arguments"
    )

    parser.add_argument(
        "--param", type=float,
        help="The independent model parameter"
    )
    parser.add_argument(
        "--step", type=float, default=0.01,
        help="The step of independent parameter (Default: %(default)s)."
    )
    parser.add_argument(
        "--num1", type=int, default=3,
        help="The number of site along a0 direction (Default: %(default)s)."
    )
    parser.add_argument(
        "--num2", type=int, default=4,
        help="The number of site along a1 direction (Default: %(default)s)."
    )

    args = parser.parse_args()
    return args.num1, args.num2, args.step, args.param


def PhaseDiagramVsAlphas(num1=3, num2=4, step=0.01, beta=0.5, tol=0.0):
    now_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    header = "Program start running at: " + now_time + "\n" + "#" * 80 + "\n\n"
    log_template = "The {ith}th alpha\nThe current alpha: {alpha:.3f}\n"
    log_template += "The ground state energy: {gse}\n"
    log_template += "The time spend on the GSE: {dt}s\n" + "=" * 80 + "\n"

    dir_log = Path("log/SpinModel/")
    dir_data = Path("data/SpinModel/")
    dir_fig = Path("fig/SpinModel/")
    dir_log.mkdir(parents=True, exist_ok=True)
    dir_data.mkdir(parents=True, exist_ok=True)
    dir_fig.mkdir(parents=True, exist_ok=True)

    template = "{0}_num1={1}_num2={2}_step={3:.3f}_beta={4:.3f}.{5}"
    file_log = dir_log / template.format("Log", num1, num2, step, beta, "txt")
    file_data = dir_data / template.format("GSE", num1, num2, step, beta, "npy")
    file_fig = dir_fig / template.format("GSE", num1, num2, step, beta, "png")

    # Break point recovery
    if Path(file_data).exists():
        res = np.load(file_data)
    else:
        alphas = np.arange(0, 2.1, step=step)
        Es = np.zeros(alphas.shape) + np.nan
        res = np.array([alphas, Es])
    alpha_num = res.shape[1]

    solver = JKGModelSolver(num1=num1, num2=num2)
    Js = np.sin(beta * np.pi) * np.sin(res[0] * np.pi)
    Ks = np.sin(beta * np.pi) * np.cos(res[0] * np.pi)
    G = np.cos(beta * np.pi)

    with open(file_log, mode="a", buffering=1) as fp:
        fp.write(header)
        for i in range(alpha_num):
            if np.isnan(res[1, i]):
                dt, res[1, i] = solver.GSE(J=Js[i], K=Ks[i], G=G, tol=tol)
                np.save(file_data, res)
                msg = log_template.format(
                    ith=i, alpha=res[0, i], gse=res[1, i], dt=dt
                )
                fp.write(msg)

    color0 = "#3465A4"
    color1 = "#F57900"
    fontsize = 12
    fig, ax_Es = plt.subplots()
    ax_d2Es = ax_Es.twinx()

    alphas_new, d2Es = derivation(res[0], res[1], nth=2)
    d2Es = -d2Es / (np.pi ** 2)
    line_Es, = ax_Es.plot(res[0], res[1], color=color0)
    line_d2Es, = ax_d2Es.plot(alphas_new, d2Es, color=color1)

    ax_Es.set_ylabel("E", color=color0, fontsize=fontsize)
    ax_Es.tick_params("y", colors=color0)
    ax_d2Es.set_ylabel(r"$-d^2E/d\alpha^2$", color=color1, fontsize=fontsize)
    ax_d2Es.tick_params("y", colors=color1)

    title = r"$E$ and $-\frac{d^2E}{d\alpha^2}$ vs $\alpha$" + "\n"
    title += r"At $\beta$ = {0:.3f}".format(beta)
    ax_Es.set_title(title, fontsize=fontsize)
    ax_Es.set_xlim(res[0, 0], res[0, -1])
    ax_Es.set_xlabel(r"$\alpha(/\pi)$")
    ax_Es.legend(
        (line_Es, line_d2Es), ("E", r"$-\frac{d^2E}{d\alpha^2}$"), loc = 0
    )
    fig.savefig(file_fig)
    # plt.close("all")
    plt.show()


def PhaseDiagramVsBetas(num1=3, num2=4, step=0.01, alpha=0.5, tol=0.0):
    now_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    header = "Program start running at: " + now_time + "\n" + "#" * 80 + "\n\n"
    log_template = "The {ith}th beta\nThe current beta: {beta:.3f}\n"
    log_template += "The ground state energy: {gse}\n"
    log_template += "The time spend on the GSE: {dt}s\n" + "=" * 80 + "\n"

    dir_log = Path("log/SpinModel/")
    dir_data = Path("data/SpinModel/")
    dir_fig = Path("fig/SpinModel/")
    dir_log.mkdir(parents=True, exist_ok=True)
    dir_data.mkdir(parents=True, exist_ok=True)
    dir_fig.mkdir(parents=True, exist_ok=True)

    template = "{0}_num1={1}_num2={2}_step={3:.3f}_alpha={4:.3f}.{5}"
    file_log = dir_log / template.format("Log", num1, num2, step, alpha, "txt")
    file_data = dir_data / template.format(
        "GSE", num1, num2, step, alpha, "npy"
    )
    file_fig = dir_fig / template.format("GSE", num1, num2, step, alpha, "png")

    if Path(file_data).exists():
        res = np.load(file_data)
    else:
        betas = np.arange(0, 1.0, step=step)
        Es = np.zeros(betas.shape) + np.nan
        res = np.array([betas, Es])
    beta_num = res.shape[1]

    solver = JKGModelSolver(num1=num1, num2=num2)
    Js = np.sin(res[0] * np.pi) * np.sin(alpha * np.pi)
    Ks = np.sin(res[0] * np.pi) * np.cos(alpha * np.pi)
    Gs = np.cos(res[0] * np.pi)

    with open(file_log, mode="a", buffering=1) as fp:
        fp.write(header)
        for i in range(beta_num):
            if np.isnan(res[1, i]):
                dt, res[1, i] = solver.GSE(J=Js[i], K=Ks[i], G=Gs[i], tol=tol)
                np.save(file_data, res)
                msg = log_template.format(
                    ith=i, beta=res[0, i], gse=res[1, i], dt=dt
                )
                fp.write(msg)

    color0 = "#3465A4"
    color1 = "#F57900"
    fontsize = 12
    fig, ax_Es = plt.subplots()
    ax_d2Es = ax_Es.twinx()

    betas_new, d2Es = derivation(res[0], res[1], nth=2)
    d2Es = -d2Es / (np.pi ** 2)
    line_Es, = ax_Es.plot(res[0], res[1], color=color0)
    line_d2Es, = ax_d2Es.plot(betas_new, d2Es, color=color1)

    ax_Es.set_ylabel("E", color=color0, fontsize=fontsize)
    ax_Es.tick_params("y", colors=color0)
    ax_d2Es.set_ylabel(r"$-d^2E/d\beta^2$", color=color1, fontsize=fontsize)
    ax_d2Es.tick_params("y", colors=color1)

    title = r"$E$ and $-\frac{d^2E}{d\beta^2}$ vs $\beta$" + "\n"
    title += r"At $\alpha$ = {0:.3f}".format(alpha)
    ax_Es.set_title(title, fontsize=fontsize)
    ax_Es.set_xlim(res[0, 0], res[0, -1])
    ax_Es.set_xlabel(r"$\beta(/\pi)$")
    ax_Es.legend(
        (line_Es, line_d2Es), ("E", r"$-\frac{d^2E}{d\beta^2}$"), loc = 0
    )
    fig.savefig(file_fig)
    # plt.close("all")
    plt.show()


if __name__ == "__main__":
    num1, num2, step, param = ArgParser()
    # PhaseDiagramVsAlphas(num1=num1, num2=num2, step=step, beta=param)
    PhaseDiagramVsBetas(num1=num1, num2=num2, step=step, alpha=param)
