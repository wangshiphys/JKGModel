"""
Exact diagonalization of the J-K-Gamma model on triangular lattice
"""


from pathlib import Path
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigsh

import logging
import time

import matplotlib.pyplot as plt
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

    def GS(self, alpha=0.5, beta=-0.5, tol=0, data_path=None, save_data=True):
        """
        Calculate the ground state energy and vector of the model Hamiltonian

        This method first check whether the ground state data for the given
        `alpha` and `beta` exists in the given `data_path`:
            1. If exist, then load and return the data;
            2. If the requested data does not exist in the given `data_path`:
                2.1 Calculate the ground state data for the given `alpha`
                and `beta`;
                2.2 If `save_data` is True, save the ground state data to
                the given `data_path`;
                2.3 Return the ground state data;

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
        data_path : str, optional
            If the ground state data for the given `alpha` and `beta` is
            already exist, then `data_path` specify where to load the data;
            If the ground state data does not exist, then `data_path` specify
            where to save the result.
            The default value `None` implies 'data/SpinModel/' relative to
            the working directory.
        save_data : boolean, optional
            Whether to save the ground state energy and vector to the
            given `data_path`.
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

        if data_path is None:
            data_path = "data/SpinModel/"
        data_dir = Path(data_path)
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


def derivation(xs, ys, nth=1):
    """
    Calculate the nth derivatives of `ys` versus `xs` discretely

    The derivatives are calculated using the following formula:
        dy/dx = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])

    Parameters
    ----------
    xs : 1-D array
        The independent variables
        `xs` is assumed to be sorted in ascending order and there are no
        identical values in `xs`.
    ys : 1-D array
        The dependent variables
        `ys` should be of the same length as `xs`.
    nth : int, optional
        The nth derivatives
        default: 1

    Returns
    -------
    xs : 1-D array
        The independent variables
    ys : 1-D array
        The nth derivatives corresponding to the returned `xs`
    """

    assert isinstance(nth, int) and nth >= 0
    assert isinstance(xs, np.ndarray) and xs.ndim == 1
    assert isinstance(ys, np.ndarray) and ys.shape == xs.shape

    for i in range(nth):
        ys = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        xs = (xs[1:] + xs[:-1]) / 2
    return xs, ys


# Useful data for plotting Es versus alphas or betas
fontsize = 10
linewidth = 4
spinewidth = 2
tick_params = {
    "labelsize": 9,
    "which": "both",
    "length": 6,
    "width": spinewidth,
    "direction": "in",
}

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def GSEsVsBeta(solver, alpha=0.5, step=0.01, save_fig=True, fig_path=None,
               **kwargs):
    """
    Calculate and plot the ground state energies versus betas with fixed alpha

    Parameters
    ----------
    solver : JKGModelSolver
        The J-K-Gamma spin model solver
    alpha : float, optional
        The `alpha` parameter
        default: 0.5
    step : float, optional
        The step of betas
        betas = np.arange(0, 2 + step, step)
        default: 0.01
    save_fig : boolean, optional
        Whether to save the figure
        If True, the figure will be saved to `fig_path` with name
        'GSEs_numx={0}_numy={1}_step={2:.3f}_alpha={3:.3f}.png'
        default: True
    fig_path : str, optional
        If `save_fig` is True, `fig_path` specify where to save the figure.
        The default value `None` implies 'figure/SpinModel/fixed_alpha/'
        relative to the working directory.
    **kwargs
        Other keyword arguments are passed to the `GS` method of The
        `JKGModelSolver` class.
    """

    info = "index=%d, alpha=%.3f, beta=%.3f, gse=%.8f, dt=%.3f"
    logger = logging.getLogger("GSEsVsBeta")

    betas = np.arange(0, 2 + step, step)
    Es = np.zeros(betas.shape, dtype=np.float64)
    for index, beta in enumerate(betas):
        t0 = time.time()
        alpha, beta, gse, ket = solver.GS(
            alpha=alpha, beta=beta, **kwargs
        )
        Es[index] = gse
        t1 = time.time()
        logger.info(info, index, alpha, beta, gse, t1 - t0)

    # Plot the ground state energies versus betas
    fig, ax_Es = plt.subplots()
    ax_d2Es = ax_Es.twinx()

    d2betas, d2Es = derivation(betas, Es, nth=2)
    line_Es, = ax_Es.plot(betas, Es, color=colors[0], lw=linewidth)
    line_d2Es, = ax_d2Es.plot(
        d2betas, -d2Es/(np.pi**2), color=colors[1], lw=linewidth,
        # marker="o", ms=10,
    )

    ax_d2Es.set_ylabel(
        r"$-d^2E/d\beta^2$", color=colors[1], fontsize=fontsize
    )
    ax_d2Es.tick_params("y", colors=colors[1], **tick_params)

    title = r"$E$ and -$\frac{d^2E}{d\beta^2}$ vs $\beta$" + "\n"
    title += r"At $\alpha$ = {0:.3f}$\pi$".format(alpha)
    ax_Es.set_title(title, fontsize=fontsize+2)

    ax_Es.set_ylabel("E", color=colors[0], fontsize=fontsize)
    ax_Es.tick_params("y", colors=colors[0], **tick_params)

    ax_Es.set_xlim(0, 2)
    ax_Es.set_xlabel(r"$\beta(/\pi)$", fontsize=fontsize)
    ax_Es.tick_params("x", **tick_params)

    ax_Es.legend(
        (line_Es, line_d2Es),
        ("E", r"$-\frac{d^2E}{d\beta^2}$"),
        loc=0
    )

    for which, spine in ax_Es.spines.items():
        spine.set_linewidth(spinewidth)

    if save_fig:
        if fig_path is None:
            fig_path = "figure/SpinModel/fixed_alpha/"
        fig_dir = Path(fig_path)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_name = "GSEs_numx={0}_numy={1}_step={2:.3f}_alpha={3:.3f}.png"
        full_name = fig_dir / fig_name.format(
            solver.numx, solver.numy, step, alpha
        )
        fig.savefig(full_name)
    else:
        plt.show()
    plt.close("all")


def GSEsVsAlpha(solver, beta=-0.5, step=0.01, save_fig=True, fig_path=None,
                **kwargs):
    """
    Calculate and plot the ground state energies versus alphas with fixed beta

    Parameters
    ----------
    solver : JKGModelSolver
        The J-K-Gamma spin model solver
    beta : float, optional
        The `beta` parameter
        default: -0.5
    step : float, optional
        The step of alphas
        alphas = np.arange(0, 1 + step, step)
        default: 0.01
    save_fig : boolean, optional
        Whether to save the figure to file system.
        If True, the figure will be saved to `fig_path` with name
        'GSEs_numx={0}_numy={1}_step={2:.3f}_beta={3:.3f}.png'
        default: True
    fig_path : str, optional
        If `save_fig` is True, `fig_path` specify where to save the figure.
        The default value `None` implies 'figure/SpinModel/fixed_beta/'
        relative to the working directory.
    **kwargs
        Other keyword arguments are passed to the `GS` method of The
        `JKGModelSolver` class.
    """

    info = "index=%d, alpha=%.3f, beta=%.3f, gse=%.8f, dt=%.3f"
    logger = logging.getLogger("GSEsVsAlpha")

    alphas = np.arange(0, 1 + step, step)
    Es = np.zeros(alphas.shape, dtype=np.float64)
    for index, alpha in enumerate(alphas):
        t0 = time.time()
        alpha, beta, gse, ket = solver.GS(
            alpha=alpha, beta=beta, **kwargs
        )
        Es[index] = gse
        t1 = time.time()
        logger.info(info, index, alpha, beta, gse, t1 - t0)

    # Plot the ground state energies versus alphas
    fig, ax_Es = plt.subplots()
    ax_d2Es = ax_Es.twinx()

    d2alphas, d2Es = derivation(alphas, Es, nth=2)
    line_Es, = ax_Es.plot(alphas, Es, color=colors[2], lw=linewidth)
    line_d2Es, = ax_d2Es.plot(
        d2alphas, -d2Es/(np.pi**2), color=colors[3], lw=linewidth,
        # marker="o", ms=10,
    )

    ax_d2Es.set_ylabel(
        r"$-d^2E/d\alpha^2$", color=colors[3], fontsize=fontsize
    )
    ax_d2Es.tick_params("y", colors=colors[3], **tick_params)

    title = r"$E$ and -$\frac{d^2E}{d\alpha^2}$ vs $\alpha$" + "\n"
    title += r"At $\beta$ = {0:.3f}$\pi$".format(beta)
    ax_Es.set_title(title, fontsize=fontsize+4)

    ax_Es.set_ylabel("E", color=colors[2], fontsize=fontsize)
    ax_Es.tick_params("y", colors=colors[2], **tick_params)

    ax_Es.set_xlim(0, 1)
    ax_Es.set_xlabel(r"$\alpha(/\pi)$", fontsize=fontsize)
    ax_Es.tick_params("x", **tick_params)

    ax_Es.legend(
        (line_Es, line_d2Es),
        ("E", r"$-\frac{d^2E}{d\alpha^2}$"),
        loc=0
    )

    for which, spine in ax_Es.spines.items():
        spine.set_linewidth(spinewidth)

    if save_fig:
        if fig_path is None:
            fig_path = "figure/SpinModel/fixed_beta/"
        fig_dir = Path(fig_path)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_name = "GSEs_numx={0}_numy={1}_step={2:.3f}_beta={3:.3f}.png"
        full_name = fig_dir / fig_name.format(
            solver.numx, solver.numy, step, beta
        )
        fig.savefig(full_name)
    else:
        plt.show()
    plt.close("all")


def main(params, fixed_which="alpha", numx=3, numy=4, log_path=None, **kwargs):
    if fixed_which.lower() == "alpha":
        log_name = "Log_numx={0}_numy={1}_alpha={2:.3f}.log"
    elif fixed_which.lower() == "beta":
        log_name = "Log_numx={0}_numy={1}_beta={2:.3f}.log"
    else:
        raise ValueError("Invalid `fixed_which` parameter!")

    if log_path is None:
        log_path = "log/SpinModel/"

    if log_path == "console":
        handler = logging.StreamHandler()
    else:
        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        full_name = log_dir / log_name.format(numx, numy, params)
        handler = logging.FileHandler(full_name)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s\n%(message)s"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    root.info("Program start running!")
    solver = JKGModelSolver(numx=numx, numy=numy)
    if fixed_which.lower() == "alpha":
        GSEsVsBeta(solver, alpha=params,  **kwargs)
    else:
        GSEsVsAlpha(solver, beta=params, **kwargs)
    root.info("Program stop running!")


if __name__ == "__main__":
    main(
        0.5, fixed_which="alpha", numx=3, numy=4, log_path="console",
        save_fig=True, fig_path="C:/Users/swang/Desktop/figure/",
        tol=1e-10, data_path="C:/Users/swang/Desktop/data/", save_data=True
    )