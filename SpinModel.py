"""
Exact diagonalization of the J-K-Gamma-Gamma' (J-K-G-GP) model on triangular
lattice.
"""


__all__ = [
    "JKGModelEDSolver",
    "JKGGPModelEDSolver",
]


import logging
from pathlib import Path
from time import time

import numpy as np
from HamiltonianPy import SpinInteraction, SpinOperator
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigsh

from utilities import TriangularLattice

logging.getLogger(__name__).addHandler(logging.NullHandler())


class JKGModelEDSolver(TriangularLattice):
    """
    Exact diagonalization of the J-K-Gamma (J-K-G) model on triangular lattice.
    """

    def _TermMatrix(self):
        # Check whether the matrix representation for the J, K, Gamma term
        # exist in the "tmp/" directory relative to the current working
        # directory(CWD). If exist, load these matrices; if not, calculate
        # the matrix representation and save the result to the "tmp/"
        # directory relative CWD, the "tmp/" directory is created if necessary.

        directory = Path("tmp/")
        name_template = "TRIANGLE_NUMX={0}_NUMY={1}_H{2}.npz"

        # The slash operator helps create child paths
        file_HJ = directory / name_template.format(self.numx, self.numy, "J")
        file_HK = directory / name_template.format(self.numx, self.numy, "K")
        file_HG = directory / name_template.format(self.numx, self.numy, "G")

        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "TermMatrix"])
        )
        if file_HJ.exists() and file_HK.exists() and file_HG.exists():
            t0 = time()
            HJ = load_npz(file_HJ)
            HK = load_npz(file_HK)
            HG = load_npz(file_HG)
            t1 = time()
            logger.info("Load HJ, HK, HG, dt=%.4fs", t1 - t0)
        else:
            site_num = self.site_num
            configs = (("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"))

            HJ = HK = HG = 0.0
            msg = "%s-bond: %2d/%2d, dt=%.4fs"
            m_func = SpinInteraction.matrix_function
            for (gamma, alpha, beta), bonds in zip(configs, self.all_bonds):
                bond_num = len(bonds)
                for count, (index0, index1) in enumerate(bonds, start=1):
                    t0 = time()
                    SKM = m_func([(index0, gamma), (index1, gamma)], site_num)
                    # Kitaev term
                    HK += SKM
                    # Heisenberg term
                    HJ += SKM
                    HJ += m_func([(index0, alpha), (index1, alpha)], site_num)
                    HJ += m_func([(index0, beta), (index1, beta)], site_num)
                    # Gamma term
                    HG += m_func([(index0, alpha), (index1, beta)], site_num)
                    HG += m_func([(index0, beta), (index1, alpha)], site_num)
                    t1 = time()
                    logger.info(msg, gamma, count, bond_num, t1 - t0)
            directory.mkdir(parents=True, exist_ok=True)
            save_npz(file_HJ, HJ, compressed=True)
            save_npz(file_HK, HK, compressed=True)
            save_npz(file_HG, HG, compressed=True)
        return HJ, HK, HG

    def GS(self, gs_path="data/SpinModel/", tol=0.0, **model_params):
        """
        Calculate the ground state energy and vector of the J-K-G model.

        This method saves the ground state data into a single file in
        compressed `.npz` format. The `.npz` files will be saved into the given
        `gs_path` and the names of the `.npz` files have the following pattern:
            "GS_numx={numx}_numy={numy}_alpha={alpha:.3f}_beta={beta:.3f}.npz"
        The variables in the "{}"s are replaced with the actual variables.
        The model parameters `alpha` and `beta` are stored with keyword
        name: `parameters`;
        The ground state energy is stored with keyword name: `gse`;
        The ground state vector is stored with keyword name: `ket`.
        For details about `.npz` format, see `numpy.savez_compressed`.

        This method first check whether the ground state data for the given
        `alpha` and `beta` exists in the given `gs_path`:
            1. If exist, read ground state data into memory;
            2. If the requested data does not exist in the given `gs_path`:
                2.1 Calculate the ground state data;
                2.2 Save the ground state data according to the rules just
                described.
            3. Return the ground state data and the Hamiltonian matrix.

        Parameters
        ----------
        gs_path : str, optional
            Where to save the ground state data. It can be an absolute path
            or a path relative to the current working directory(CWD). The
            specified `gs_path` will be created if necessary.
            Default: "data/SpinModel/"(Relative to CWD).
        tol : float, optional
            Relative accuracy for eigenvalues (stop criterion).
            The default value 0 implies machine precision.
        model_params : optional
            The model parameters recognized by this method are `alpha` and
            `beta`. All other keyword arguments are simply ignored. If `alpha`
            and/or `beta` are no specified, the default value defined in
            class variable `DEFAULT_MODEL_PARAMETERS` will be used.

            J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
            K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
            G = np.cos(alpha * np.pi)
            J is the coefficient of the Heisenberg term.
            K is the coefficient of the Kitaev term.
            G is the coefficient of the Gamma term.

        Returns
        -------
        gse : float
            The ground state energy of the model Hamiltonian at the given
            model parameters.
        ket : np.ndarray with shape (N, 1)
            The corresponding ground state vector.
        HM : csr_matrix
            The corresponding Hamiltonian matrix.

        See also
        --------
        numpy.savez_compressed
        """

        actual_model_params = dict(self.DEFAULT_MODEL_PARAMETERS)
        actual_model_params.update(model_params)
        alpha = actual_model_params["alpha"]
        beta = actual_model_params["beta"]
        J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
        K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
        G = np.cos(alpha * np.pi)
        HJ, HK, HG = self._TermMatrix()
        HM = K * HK
        del HK
        HM += J * HJ
        del HJ
        HM += G * HG
        del HG

        gs_path = Path(gs_path)
        gs_name_temp = "GS_numx={0}_numy={1}_alpha={2:.3f}_beta={3:.3f}.npz"
        gs_full_name = gs_path / gs_name_temp.format(
            self.numx, self.numy, alpha, beta
        )

        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "GS"])
        )
        if gs_full_name.exists():
            with np.load(gs_full_name) as ld:
                gse = ld["gse"]
                ket = ld["ket"]
            logger.info("Load GS data from %s", gs_full_name)
        else:
            t0 = time()
            gse, ket = eigsh(HM, k=1, which="SA", tol=tol)
            t1 = time()
            msg = "GS for alpha=%.3f, beta=%.3f, dt=%.4fs"
            logger.info(msg, alpha, beta, t1 - t0)

            gs_path.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                gs_full_name, parameters=[alpha, beta], gse=gse, ket=ket
            )
            logger.info("Save GS data to %s", gs_full_name)
        return gse[0], ket, HM

    def excited_states(self, gs_ket):
        """
        Calculate excited states.

        Parameters
        ----------
        gs_ket : np.ndarray with shape (N, 1)
            The ground state vector.

        Returns
        -------
        excited_states : dict
            A collection of excited states.
            The key are instances of SpinOperator, and the values are the
            corresponding vectors of excited states.
        """

        cluster = self.cluster
        total_spin = self.site_num
        excited_states = {}
        for site in cluster.points:
            site_index = cluster.getIndex(site=site, fold=False)
            for otype in ("p", "m"):
                spin_operator = SpinOperator(otype=otype, site=site)
                excited_states[spin_operator] = SpinOperator.matrix_function(
                    (site_index, otype), total_spin=total_spin
                ).dot(gs_ket)
        return excited_states


class JKGGPModelEDSolver(JKGModelEDSolver):
    """
    Exact diagonalization of the J-K-G-GP model on triangular lattice.
    """

    def _TermMatrix(self):
        # Check whether the matrix representation for the J, K, Gamma,
        # Gamma' term exist in the "tmp/" directory relative to the current
        # working directory(CWD). If exist, load these matrices; if not,
        # calculate the matrix representation and save the result to the
        # "tmp/" directory relative CWD, the "tmp/" directory is created if
        # necessary.

        # Prepare HJ, HK, HG first
        HJ, HK, HG = super()._TermMatrix()

        file_HGP = Path("tmp/") / "TRIANGLE_NUMX={0}_NUMY={1}_HGP.npz".format(
            self.numx, self.numy
        )

        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "TermMatrix"])
        )
        if file_HGP.exists():
            t0 = time()
            HGP = load_npz(file_HGP)
            t1 = time()
            logger.info("Load HGP, dt=%.4fs", t1 - t0)
        else:
            site_num = self.site_num
            configs = (("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"))

            HGP = 0.0
            msg = "%s-bond: %2d/%2d, dt=%.4fs"
            m_func = SpinInteraction.matrix_function
            for (gamma, alpha, beta), bonds in zip(configs, self.all_bonds):
                bond_num = len(bonds)
                for count, (index0, index1) in enumerate(bonds, start=1):
                    t0 = time()
                    HGP += m_func([(index0, alpha), (index1, gamma)], site_num)
                    HGP += m_func([(index0, gamma), (index1, alpha)], site_num)
                    HGP += m_func([(index0, beta), (index1, gamma)], site_num)
                    HGP += m_func([(index0, gamma), (index1, beta)], site_num)
                    t1 = time()
                    logger.info(msg, gamma, count, bond_num, t1 - t0)
            save_npz(file_HGP, HGP, compressed=True)
        return HJ, HK, HG, HGP

    def GS(self, gs_path="data/SpinModel/", tol=0.0, **model_params):
        """
        Calculate the ground state energy and vector of the J-K-G-GP model.

        This method saves the ground state data into a single file in
        compressed `.npz` format. The `.npz` files will be saved into the given
        `gs_path` and the names of the `.npz` files have the following pattern:
            "GS_numx={numx}_numy={numy}_J={J:.3f}_K={K:.3f}_G={G:.3f}_GP={GP:.3f}.npz"
        The variables in the "{}"s are replaced with the actual variables.
        The model parameters `J`, `K`, `G` and `GP` are stored with keyword
        name: `parameters`;
        The ground state energy is stored with keyword name: `gse`;
        The ground state vector is stored with keyword name: `ket`.
        For details about `.npz` format, see `numpy.savez_compressed`.

        This method first check whether the ground state data for the given
        `J`, `K`, `G` and `GP` exists in the given `gs_path`:
            1. If exist, read the ground state data into memory;
            2. If the requested data does not exist in the given `gs_path`:
                2.1 Calculate the ground state data;
                2.2 Save the ground state data according to the rules just
                described.
            3. Return the ground state data and Hamiltonian matrix.

        Parameters
        ----------
        gs_path : str, optional
            Where to save the ground state data. It can be an absolute path
            or a path relative to the current working directory(CWD). The
            specified `gs_path` will be created if necessary.
            Default: "data/SpinModel/"(Relative to CWD).
        tol : float, optional
            Relative accuracy for eigenvalues (stop criterion).
            The default value 0 implies machine precision.
        model_params : optional
            The model parameters recognized by this method are `J`, `K`,
            `G` and `GP`. All other keyword arguments are simply ignored.
            If `J`, `K` , `G`, `GP` are not specified, the default value
            defined in class variable `DEFAULT_MODEL_PARAMETERS` will be used.

            J is the coefficient of the Heisenberg term.
            K is the coefficient of the Kitaev term.
            G is the coefficient of the Gamma term.
            GP is the coefficient of the Gamma' term.
        """

        actual_model_params = dict(self.DEFAULT_MODEL_PARAMETERS)
        actual_model_params.update(model_params)
        J = actual_model_params["J"]
        K = actual_model_params["K"]
        G = actual_model_params["G"]
        GP = actual_model_params["GP"]
        HJ, HK, HG, HGP = self._TermMatrix()
        HM = K * HK
        del HK
        HM += GP * HGP
        del HGP
        HM += J * HJ
        del HJ
        HM += G * HG
        del HG

        gs_path = Path(gs_path)
        gs_full_name = gs_path / "GS_numx={numx}_numy={numy}_J={J:.3f}_" \
                       "K={K:.3f}_G={G:.3f}_GP={GP:.3f}.npz".format(
            numx=self.numx, numy=self.numy, J=J, K=K, G=G, GP=GP
        )

        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "GS"])
        )

        if gs_full_name.exists():
            with np.load(gs_full_name) as ld:
                gse = ld["gse"]
                ket = ld["ket"]
            logger.info("Load GS data from %s", gs_full_name)
        else:
            t0 = time()
            gse, ket = eigsh(HM, k=1, which="SA", tol=tol)
            t1 = time()
            msg = "GS for J=%.3f, K=%.3f, G=%.3f, GP=%.3f, dt=%.4fs"
            logger.info(msg, J, K, G, GP, t1 - t0)

            gs_path.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                gs_full_name, parameters=[J, K, G, GP], gse=gse, ket=ket
            )
            logger.info("Save GS data to %s", gs_full_name)
        return gse[0], ket, HM


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from utilities import derivation

    numx = 3
    numy = 4
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        # filename="log/Log_numx={0}_numy={1}.log".format(numx, numy),
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    solver0 = JKGModelEDSolver(numx, numy)
    solver1 = JKGGPModelEDSolver(numx, numy)

    solver0.GS()
    solver0.GS(alpha=1.2, beta=0.8)
    solver0.GS(alpha=1.2, beta=0.8)
    solver0.GS(alpha=1.0, beta=0.6)
    solver0.GS(alpha=1.0, beta=0.7)

    solver1.GS()
    solver1.GS(J=-1, K=-6, G=8, GP=-4)
    solver1.GS(J=-1, K=-6, G=8, GP=-4)
    solver1.GS(J=1, K=6, G=-8, GP=4)
    solver1.GS(J=1, K=6, G=-8, GP=5)

    alpha = 0.5
    betas = np.arange(0, 2, 0.005)
    for beta in betas:
        solver0.GS(alpha=alpha, beta=beta)

    gses = []
    gs_path = "data/SpinModel/"
    gs_name_temp = "GS_numx={0}_numy={1}_alpha={2:.3f}_beta={3:.3f}.npz"
    for beta in betas:
        gs_full_name = gs_path + gs_name_temp.format(numx, numy, alpha, beta)
        with np.load(gs_full_name) as ld:
            gse = ld["gse"][0]
        gses.append(gse)
    gses = np.array(gses, dtype=np.float64)
    d2betas, d2gses = derivation(betas, gses, nth=2)

    fig, ax_Es = plt.subplots()
    ax_d2Es = ax_Es.twinx()

    ax_Es.plot(betas, gses, color="tab:blue")
    ax_d2Es.plot(d2betas, -d2gses/(np.pi ** 2), color="tab:orange")
    ax_Es.set_xlim(0, 2)
    plt.show()
    plt.close("all")
