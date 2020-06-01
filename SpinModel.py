"""
Exact diagonalization of the J-K-Gamma (J-K-G) and J-K-Gamma-Gamma' (J-K-G-GP)
model on triangular lattice.
"""


__all__ = ["JKGModelEDSolver", "JKGGPModelEDSolver"]


import logging
from pathlib import Path
from time import time

import HamiltonianPy as HP
import numpy as np
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
            logger.info("Load HJ, HK, HG, dt=%.3fs", t1 - t0)
        else:
            site_num = self.site_num
            configs = (("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"))

            HJ = HK = HG = 0.0
            msg = "%s-bond: %2d/%2d, dt=%.3fs"
            m_func = HP.SpinInteraction.matrix_function
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

    def EigenStates(
            self, es_path="data/QuantumSpinModel/ES/",
            k=1, v0=None, tol=0.0, **model_params,
    ):
        """
        Find the smallest `k` eigenvalues and eigenvectors of the J-K-G
        model Hamiltonian.

        This method saves the eigenvalues and eigenvectors into a single file in
        compressed `.npz` format. The `.npz` files will be saved into the given
        `es_path` and the names of the `.npz` files have the following pattern:
        `ES_num1={num1}_num2={num2}_direction={direction}_alpha={alpha:.4f}_beta={beta:.4f}.npz`
        The variables in the "{}"s are replaced with the actual variables.

        The lattice size `num1` and `num2` are stored with keyword name: `size`;
        The lattice direction is stored with keyword name: `direction`;
        The model parameters `alpha` and `beta` are stored with keyword
        name: `parameters`;
        The eigenvalues are stored with keyword name: `values`;
        The eigenvectors are stored with keyword name: `vectors`.
        For details about `.npz` format, see `numpy.savez_compressed`.

        This method first check whether the eigenstates data for the given
        `alpha` and `beta` exists in the given `es_path`. If exist,
        read eigenstates data into memory; if the requested data does not
        exist, then preform the calculation and save the eigenstates data
        according to the rules just described. Finally the eigenstates data
        and the Hamiltonian matrix are returned.

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
            possible to compute all eigenvectors of a large matrix.
            Default: 1.
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
        J = 0.0 if np.abs(J) <= 1E-10 else J
        K = 0.0 if np.abs(K) <= 1E-10 else K
        G = 0.0 if np.abs(G) <= 1E-10 else G

        HJ, HK, HG = self._TermMatrix()
        HJ *= J
        HK *= K
        HG *= G
        HM = HJ + HK
        del HJ, HK
        HM += HG
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
            msg = "ES for alpha=%.4f, beta=%.4f, dt=%.3fs"
            logger.info(msg, alpha, beta, t1 - t0)

            es_path.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                es_full_name,
                size=[self.num1, self.num2], direction=[self.direction],
                parameters=[alpha, beta], values=values, vectors=vectors,
            )
            logger.info("Save ES data to %s", es_full_name)
        return values, vectors, HM

    def excited_states(self, gs_ket, excitation="Sm"):
        """
        Calculate excited states.

        Parameters
        ----------
        gs_ket : np.ndarray with shape (N, )
            The ground state vector.
        excitation : ["Sp" | "Sm" | "Sz"], str, optional
            The type of operators to be operated on the ground state.
            "Sp": Operate all $S_i^+$ operators on the ground state;
            "Sm": Operate all $S_i^-$ operators on the ground state;
            "Sz": Operate all $S_i^z$ operators on the ground state.
            Default: "Sm".

        Returns
        -------
        excited_states : dict
            A collection of excited states.
            The key are instances of SpinOperator, and the values are the
            corresponding vectors of excited states.
        """

        if excitation == "Sp":
            otype = "p"
        elif excitation == "Sm":
            otype = "m"
        elif excitation == "Sz":
            otype = "z"
        else:
            raise ValueError("Invalid `excitation`: {0}".format(excitation))

        excited_states = dict()
        cluster = self.cluster
        total_spin = self.site_num
        mfunc = HP.SpinOperator.matrix_function
        for site in cluster.points:
            site_index = cluster.getIndex(site=site, fold=False)
            spin_operator = HP.SpinOperator(otype=otype, site=site)
            excited_states[spin_operator] = mfunc(
                (site_index, otype), total_spin=total_spin
            ).dot(gs_ket)
        return excited_states

    def LanczosProjection(self, HM, gs_ket, excitation="Sm"):
        """
        Perform Lanczos projection of the Hamiltonian matrix and excited states.

        Parameters
        ----------
        HM : csr_matrix with shape (N, N)
            The Hamiltonian matrix.
        gs_ket : np.ndarray with shape (N, )
            The ground state vector.
        excitation : ["all", "Sp" | "Sm" | "Sz"], str, optional
            The type of operators to be operated on the ground state.
            "Sp": Operate all $S_i^+$ operators on the ground state;
            "Sm": Operate all $S_i^-$ operators on the ground state;
            "Sz": Operate all $S_i^z$ operators on the ground state;
            "all": Operate all $S_i^+$, $S_i^-$, $S_i^z$ operators on the
            ground state.
            Default: "Sm".

        Returns
        -------
        projected_matrices : dict
            The representations of `HM` in these Krylov spaces.
        projected_vectors: dict
            The representations of `vectors` in these Krylov space.
        """

        assert excitation in ("all", "Sp", "Sm", "Sz")

        if excitation == "all":
            projected_vectors = dict()
            projected_matrices = dict()
            for tag in ["Sp", "Sm", "Sz"]:
                excited_states_tag = self.excited_states(gs_ket, excitation=tag)
                projected_matrices_tag, projected_vectors_tag = HP.MultiKrylov(
                    HM, excited_states_tag
                )
                projected_vectors.update(projected_vectors_tag)
                projected_matrices.update(projected_matrices_tag)
        else:
            excited_states = self.excited_states(gs_ket, excitation=excitation)
            projected_matrices, projected_vectors = HP.MultiKrylov(
                HM, excited_states
            )
        return projected_matrices, projected_vectors


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

        file_HGP = Path("tmp/TRIANGLE_" + self.identity.upper() + "_HGP.npz")

        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "TermMatrix"])
        )
        if file_HGP.exists():
            t0 = time()
            HGP = load_npz(file_HGP)
            t1 = time()
            logger.info("Load HGP, dt=%.3fs", t1 - t0)
        else:
            site_num = self.site_num
            configs = (("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"))

            HGP = 0.0
            msg = "%s-bond: %2d/%2d, dt=%.3fs"
            m_func = HP.SpinInteraction.matrix_function
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

    def EigenStates(
            self, es_path="data/QuantumSpinModel/ES/",
            k=1, v0=None, tol=0.0, **model_params,
    ):
        """
        Find the smallest `k` eigenvalues and eigenvectors of the J-K-G-GP
        model Hamiltonian.

        This method saves the eigenvalues and eigenvectors into a single file in
        compressed `.npz` format. The `.npz` files will be saved into the given
        `es_path` and the names of the `.npz` files have the following pattern:
        `ES_num1={num1}_num2={num2}_direction={direction}_J={J:.4f}_K={K:.4f}_G={G:.4f}_GP={GP:.4f}.npz`
        The variables in the "{}"s are replaced with the actual variables.

        The lattice size `num1` and `num2` are stored with keyword name: `size`;
        The lattice direction is stored with keyword name: `direction`;
        The model parameters `J`, `K`, `G` and `GP` are stored with keyword
        name: `parameters`;
        The eigenvalues are stored with keyword name: `values`;
        The eigenvectors are stored with keyword name: `vectors`.
        For details about `.npz` format, see `numpy.savez_compressed`.

        This method first check whether the eigenstates data for the given
        `J`, `K`, `G` and `GP` exists in the given `es_path`. If exist,
        read eigenstates data into memory; if the requested data does not
        exist, then preform the calculation and save the eigenstates data
        according to the rules just described. Finally the eigenstates data
        and the Hamiltonian matrix are returned.

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
            possible to compute all eigenvectors of a large matrix.
            Default: 1.
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
        HJ *= J
        HK *= K
        HG *= G
        HGP *= GP
        HM = HJ + HK
        del HJ, HK
        HM += HG
        del HG
        HM += HGP
        del HGP

        es_path = Path(es_path)
        es_file_name = "ES_" + self.identity
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
            msg = "ES for J=%.4f, K=%.4f, G=%.4f, GP=%.4f, dt=%.3fs"
            logger.info(msg, J, K, G, GP, t1 - t0)

            es_path.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                es_full_name,
                size=[self.num1, self.num2], direction=[self.direction],
                parameters=[J, K, G, GP], values=values, vectors=vectors,
            )
            logger.info("Save ES data to %s", es_full_name)
        return values, vectors, HM


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s - %(message)s",
    )

    num1 = 3
    num2 = 4
    direction = "xy"
    alpha, beta = 0.3, 0.25
    solver = JKGModelEDSolver(num1=num1, num2=num2, direction=direction)
    values, vectors, HM = solver.EigenStates(alpha=alpha, beta=beta)
    GE = values[0]
    print("alpha={0:.4f},beta={1:.4f}, GE={2}".format(alpha, beta, GE))
    values, vectors, HM = solver.EigenStates(alpha=alpha, beta=beta)

    J, K, G, GP = 1.0, 2.0, 3.0, 4.0
    solver = JKGGPModelEDSolver(num1=num1, num2=num2, direction=direction)
    values, vectors, HM = solver.EigenStates(J=J, K=K, G=G, GP=GP)
    GE = values[0]
    msg = "J={0:.1f},K={1:.1f},G={2:.1f},GP={3:.1f}, GE={4}"
    print(msg.format(J, K, G, GP, GE))
    values, vectors, HM = solver.EigenStates(J=J, K=K, G=G, GP=GP)
