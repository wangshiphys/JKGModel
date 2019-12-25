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
        name_template = "TRIANGLE_" + self.identity.upper() + "_H{0}.npz"

        # The slash operator helps create child paths
        file_HJ = directory / name_template.format("J")
        file_HK = directory / name_template.format("K")
        file_HG = directory / name_template.format("G")

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

    def GS(
            self, es_path="data/QuantumSpinModel/ES/",
            k=6, v0=None, tol=0.0, **model_params,
    ):
        """
        Find the smallest `kth` eigenvalues and eigenvectors of the J-K-G
        model Hamiltonian.

        This method saves the eigenvalues and eigenvectors into a single file in
        compressed `.npz` format. The `.npz` files will be saved into the given
        `es_path` and the names of the `.npz` files have the following pattern:
            "ES_num1={num1}_num2={num2}_which={which}_
            alpha={alpha:.4f}_beta={beta:.4f}.npz"
        The variables in the "{}"s are replaced with the actual variables.
        The lattice size `num1` and `num2` are stored with keyword name: `size`;
        The lattice type `which` is stored with keyword name: `which`;
        The model parameters `alpha` and `beta` are stored with keyword
        name: `parameters`;
        The eigenvalues are stored with keyword name: `values`;
        The eigenvectors are stored with keyword name: `vectors`.
        For details about `.npz` format, see `numpy.savez_compressed`.

        This method first check whether the eigenstates data for the given
        `alpha` and `beta` exists in the given `es_path`:
            1. If exist, read eigenstates data into memory;
            2. If the requested data does not exist in the given `es_path`:
                2.1 Preform the calculation;
                2.2 Save the eigenstates data according to the rules just
                described.
            3. Return the eigenstates data and the Hamiltonian matrix.

        Parameters
        ----------
        es_path : str, optional
            Where to save the eigenstates data. It can be an absolute path
            or a path relative to the current working directory(CWD). The
            specified `es_path` will be created if necessary.
            Default: "data/QuantumSpinModel/ES/"(Relative to CWD).
        k : int, optional
            The number of eigenvalues and eigenvectors desired. `k` must be
            smaller than the dimension of the Hilbert space. It is not
            possible to compute all eigenvectors of a matrix.
            Default: 6.
        v0 : np.ndarray, optional
            Starting vector for iteration. This parameter is passed to the
            `scipy.sparse.linalg.eigsh` as the `v0` parameter.
            Default: None.
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
            J is the coefficient of the Heisenberg term;
            K is the coefficient of the Kitaev term;
            G is the coefficient of the Gamma term.

        Returns
        -------
        values : np.array
            Array of k eigenvalues.
        vectors : np.ndarray with shape (N, k)
            An array representing the k eigenvectors.
            The column `vectors[:, i]` is the eigenvector corresponding to
            the eigenvalue `values[i]`.
        HM : csr_matrix
            The corresponding Hamiltonian matrix.

        See also
        --------
        numpy.savez_compressed
        scipy.sparse.linalg.eigsh
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

        es_path = Path(es_path)
        es_name_temp = "ES_" + self.identity + "_alpha={0:.4f}_beta={1:.4f}.npz"
        es_full_name = es_path / es_name_temp.format(alpha, beta)

        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "GS"])
        )
        if es_full_name.exists():
            with np.load(es_full_name) as ld:
                values = ld["values"]
                vectors = ld["vectors"]
            logger.info("Load ES data from %s", es_full_name)
        else:
            t0 = time()
            values, vectors = eigsh(HM, k=k, which="SA", v0=v0, tol=tol)
            t1 = time()
            msg = "ES for alpha=%.4f, beta=%.4f, dt=%.4fs"
            logger.info(msg, alpha, beta, t1 - t0)

            es_path.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                es_full_name, size=[self.num1, self.num2], which=[self.which],
                parameters=[alpha, beta], values=values, vectors=vectors,
            )
            logger.info("Save ES data to %s", es_full_name)
        return values, vectors, HM

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

        file_HGP = Path("tmp/") / ("TRIANGLE_" + self.identity + "_HGP.npz")

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

    def GS(
            self, es_path="data/QuantumSpinModel/ES/",
            k=6, v0=None, tol=0.0, **model_params,
    ):
        """
        Find the smallest `kth` eigenvalues and eigenvectors of the J-K-G-GP
        model Hamiltonian.

        This method saves the eigenvalues and eigenvectors into a single file in
        compressed `.npz` format. The `.npz` files will be saved into the given
        `es_path` and the names of the `.npz` files have the following pattern:
            "ES_num1={num1}_num2={num2}_which={which}_
            J={J:.4f}_K={K:.4f}_G={G:.4f}_GP={GP:.4f}.npz"
        The variables in the "{}"s are replaced with the actual variables.
        The lattice size `num1` and `num2` are stored with keyword name: `size`;
        The lattice type `which` is stored with keyword name: `which`;
        The model parameters `J`, `K`, `G` and `GP` are stored with keyword
        name: `parameters`;
        The eigenvalues are stored with keyword name: `values`;
        The eigenvectors are stored with keyword name: `vectors`.
        For details about `.npz` format, see `numpy.savez_compressed`.

        This method first check whether the eigenstates data for the given
        `J`, `K`, `G` and `GP` exists in the given `gs_path`:
            1. If exist, read the eigenstates data into memory;
            2. If the requested data does not exist in the given `es_path`:
                2.1 Perform the calculation;
                2.2 Save the eigenstates data according to the rules just
                described.
            3. Return the eigenstates data and Hamiltonian matrix.

        Parameters
        ----------
        es_path : str, optional
            Where to save the eigenstates data. It can be an absolute path
            or a path relative to the current working directory(CWD). The
            specified `es_path` will be created if necessary.
            Default: "data/QuantumSpinModel/ES/"(Relative to CWD).
        k : int, optional
            The number of eigenvalues and eigenvectors desired. `k` must be
            smaller than the dimension of the Hilbert space. It is not
            possible to compute all eigenvectors of a matrix.
            Default: 6.
        v0 : np.ndarray, optional
            Starting vector for iteration. This parameter is passed to the
            `scipy.sparse.linalg.eigsh` as the `v0` parameter.
            Default: None.
        tol : float, optional
            Relative accuracy for eigenvalues (stop criterion).
            The default value 0 implies machine precision.
        model_params : optional
            The model parameters recognized by this method are `J`, `K`,
            `G` and `GP`. All other keyword arguments are simply ignored.
            If `J`, `K` , `G`, `GP` are not specified, the default value
            defined in class variable `DEFAULT_MODEL_PARAMETERS` will be used.

            J is the coefficient of the Heisenberg term;
            K is the coefficient of the Kitaev term;
            G is the coefficient of the Gamma term;
            GP is the coefficient of the Gamma' term.

         Returns
        -------
        values : np.array
            Array of k eigenvalues.
        vectors : np.ndarray with shape (N, k)
            An array representing the k eigenvectors.
            The column `vectors[:, i]` is the eigenvector corresponding to
            the eigenvalue `values[i]`.
        HM : csr_matrix
            The corresponding Hamiltonian matrix.

        See also
        --------
        numpy.savez_compressed
        scipy.sparse.linalg.eigsh
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

        es_path = Path(es_path)
        es_file_name = "GS_" + self.identity
        es_file_name += "_J={J:.4f}_K={K:.4f}_G={G:.4f}_GP={GP:.4f}.npz".format(
            J=J, K=K, G=G, GP=GP,
        )
        es_full_name = es_path / es_file_name

        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "GS"])
        )
        if es_full_name.exists():
            with np.load(es_full_name) as ld:
                values = ld["values"]
                vectors = ld["vectors"]
            logger.info("Load ES data from %s", es_full_name)
        else:
            t0 = time()
            values, vectors = eigsh(HM, k=k, which="SA", v0=v0, tol=tol)
            t1 = time()
            msg = "ES for J=%.4f, K=%.4f, G=%.4f, GP=%.4f, dt=%.4fs"
            logger.info(msg, J, K, G, GP, t1 - t0)

            es_path.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                es_full_name, size=[self.num1, self.num2], which=[self.which],
                parameters=[J, K, G, GP], values=values, vectors=vectors,
            )
            logger.info("Save ES data to %s", es_full_name)
        return values, vectors, HM
