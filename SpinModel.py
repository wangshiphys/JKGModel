"""
Exact diagonalization of the J-K-Gamma-Gamma'(J-K-G-GP) model on triangular
lattice.
"""


import logging
import warnings
from pathlib import Path
from time import time

import numpy as np
import tables as tb
from HamiltonianPy import SpinInteraction, SpinOperator
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigsh

from cluster import TriangularCluster


logging.getLogger(__name__).addHandler(logging.NullHandler())
warnings.filterwarnings("ignore", category=tb.NaturalNameWarning)


class JKGModelSolver(TriangularCluster):
    """
    Exact diagonalization of the J-K-Gamma(J-K-G) model on triangular lattice.
    """

    def _TermMatrix(self):
        # Check whether the matrix representation for the J, K, Gamma term
        # exist on the "tmp/" directory relative to the current working
        # directory(CWD). If exist, load these matrices; if not, calculate
        # the matrix representation and save the result to the "tmp/"
        # directory relative CWD, the "tmp/" directory is created if necessary.

        directory = Path("tmp/")
        name_template = "TRIANGLE_NUMX={0}_NUMY={1}_H{{0}}.npz".format(
            self.numx, self.numy
        )
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
            logger.info("Load HJ, HK, HG, dt=%.6fs", t1 - t0)
        else:
            site_num = self.site_num
            configs = [("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y")]

            HJ = HK = HG = 0.0
            msg = "%s-bond: %2d/%2d, dt=%.6fs"
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

    @staticmethod
    def _load_gs_data(hdf5_file_name, ket_name):
        if Path(hdf5_file_name).exists():
            h5f = tb.open_file(hdf5_file_name, mode="r")
            try:
                ket_carray = h5f.get_node("/", ket_name)
                gse = ket_carray.get_attr("gse")
                ket = ket_carray.read()
                res = (gse, ket)
            except tb.NoSuchNodeError:
                res = None
            h5f.close()
        else:
            res = None
        return res

    @staticmethod
    def _save_gs_data(hdf5_file_name, ket_name, ket, attrs):
        hdf5 = tb.open_file(hdf5_file_name, mode="a")
        ket_array = hdf5.create_carray(
            "/", ket_name, obj=ket,
            filters=tb.Filters(complevel=9, complib="zlib")
        )
        for attrname, attrvalue in attrs.items():
            ket_array.set_attr(attrname, attrvalue)
        hdf5.close()

    def GS(self, gs_path="data/SpinModel/", tol=0.0, **model_params):
        """
        Calculate the ground state energy and vector of the J-K-Gamma model
        Hamiltonian.

        This method assumes that the ground state data are stored in
        pytables-files(hdf5-files) located in the specified `gs_path`. The
        names of the hdf5-files have the following pattern:
            "GS_numx={numx}_numy={numy}_alpha={alpha:.3f}.zlib.h5"
        The ground state vector for a specific `alpha` and `beta` is stored
        in the hdf5-file as a `tables.CArray`, the model parameters `alpha`,
        `beta` and ground state energy `gse` are saved as attributes of the
        `tables.CArray`. The names of the `tables.CArray` have the following
        pattern:
            "GS_numx={numx}_numy={numy}_alpha={alpha:.3f}_beta={beta:.3f}"
        The variables in the "{}"s are replaced with the actual variables.

        This method first check whether the ground state data for the given
        `alpha` and `beta` exists in the given `gs_path`:
            1. If exist, then stop;
            2. If the requested data does not exist in the given `gs_path`:
                2.1 Calculate the ground state data;
                2.2 Save the ground state data according to the rules just
                described.

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
            class variable `DEFAULT_MODEL_PARAMS` will be used.

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
        ket : np.ndarray
            The corresponding ground state vector.
        HM : csr_matrix
            The corresponding Hamiltonian matrix.
        """

        actual_model_params = dict(self.DEFAULT_MODEL_PARAMETERS)
        actual_model_params.update(model_params)
        alpha = actual_model_params["alpha"]
        beta = actual_model_params["beta"]
        J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
        K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
        G = np.cos(alpha * np.pi)
        HJ, HK, HG = self._TermMatrix()
        HM = J * HJ + K * HK + G * HG
        del HJ, HK, HG

        prefix = "GS_numx={0}_numy={1}".format(self.numx, self.numy)
        fixed_params = "alpha={0:.3f}".format(alpha)
        full_params = "alpha={0:.3f}_beta={1:.3f}".format(alpha, beta)

        # `gs_path` is created if not exist
        Path(gs_path).mkdir(exist_ok=True, parents=True)
        hdf5_file_name = "".join(
            [gs_path, prefix, "_", fixed_params, ".zlib.h5"]
        )
        ket_name = "".join([prefix, "_", full_params])

        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "GS"])
        )

        gs_data = self._load_gs_data(hdf5_file_name, ket_name)
        if gs_data is None:
            t0 = time()
            (gse, ), ket = eigsh(HM, k=1, which="SA", tol=tol)
            t1 = time()
            logger.info(
                "GS for alpha=%.3f, beta=%.3f, dt=%.6fs", alpha, beta, t1 - t0
            )

            # Save the ground state data
            attrs = {
                "numx": self.numx, "numy": self.numy,
                "alpha": alpha, "beta": beta, "gse": gse,
            }
            self._save_gs_data(hdf5_file_name, ket_name, ket, attrs=attrs)
            logger.info("Save GS data to %s:/%s", hdf5_file_name, ket_name)
        else:
            gse, ket = gs_data
            logger.info(
                "GS data for alpha=%.3f, beta=%.3f already exist", alpha, beta
            )
        return gse, ket, HM

    def excited_states(self, gs_ket):
        """
        Calculate excited states.

        Parameters
        ----------
        gs_ket : np.ndarray
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


class JKGGPModelSolver(JKGModelSolver):
    """
    Exact diagonalization of the J-K-Gamma-Gamma' model on triangular lattice.
    """

    def _TermMatrix(self):
        # Check whether the matrix representation for the J, K, Gamma,
        # Gamma' term exist on the "tmp/" directory relative to the current
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
            logger.info("Load HGP, dt=%.6fs", t1 - t0)
        else:
            site_num = self.site_num
            configs = [("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y")]

            HGP = 0.0
            msg = "%s-bond: %2d/%2d, dt=%.6fs"
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
        Calculate the ground state energy and vector of the J-K-Gamma-Gamma'
        model Hamiltonian.

        This method assumes that the ground state data are stored in
        pytables-files(hdf5-files) located in the specified `gs_path`. The
        names of the hdf5-files have the following pattern:
          "GS_numx={numx}_numy={numy}_J={J:.4f}_K={K:.4f}_G={G:.4f}.zlib.h5"
        The ground state vector for a specific `J, K, G, GP` is stored
        in the hdf5-file as a `tables.CArray`, the model parameters
        `J, K, G, GP` and ground state energy `gse` are saved as attributes
        of the `tables.CArray`. The names of the `tables.CArray` have the
        following pattern:
          "GS_numx={numx}_numy={numy}_J={J:.4f}_K={K:.4f}_G={G:.4f}_GP={GP:.4f}"
        The variables in the "{}"s are replaced with the actual variables.

        This method first check whether the ground state data for the given
        `J, K, G, GP` exists in the given `gs_path`:
            1. If exist, then stop;
            2. If the requested data does not exist in the given `gs_path`:
                2.1 Calculate the ground state data;
                2.2 Save the ground state data according to the rules just
                described.

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
            defined in class variable `DEFAULT_MODEL_PARAMS` will be used.

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
        HM = J * HJ + K * HK + G * HG + GP * HGP
        del HJ, HK, HG, HGP

        prefix = "GS_numx={0}_numy={1}".format(self.numx, self.numy)
        fixed_params = "J={J:.4f}_K={K:.4f}_G={G:.4f}".format(J=J, K=K, G=G)
        full_params = "J={J:.4f}_K={K:.4f}_G={G:.4f}_GP={GP:.4f}".format(
            J=J, K=K, G=G, GP=GP
        )

        # `gs_path` is created if not exist
        Path(gs_path).mkdir(exist_ok=True, parents=True)
        hdf5_file_name = "".join(
            [gs_path, prefix, "_", fixed_params, ".zlib.h5"]
        )
        ket_name = "".join([prefix, "_", full_params])

        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "GS"])
        )

        gs_data = self._load_gs_data(hdf5_file_name, ket_name)
        if gs_data is None:
            t0 = time()
            (gse, ), ket = eigsh(HM, k=1, which="SA", tol=tol)
            t1 = time()
            msg = "J=%.4f, K=%.4f, G=%.4f, GP=%.4f, dt=%.6fs"
            logger.info(msg, J, K, G, GP, t1 - t0)

            # Save the ground state data
            attrs = {
                "numx": self.numx, "numy": self.numy,
                "J": J, "K": K, "G": G, "GP": GP, "gse": gse,
            }
            self._save_gs_data(hdf5_file_name, ket_name, ket, attrs=attrs)
            logger.info("Save GS data to %s:/%s", hdf5_file_name, ket_name)
        else:
            gse, ket = gs_data
            logger.info(
                "GS for J=%.4f, K=%.4f, G=%.4f, GP=%.4f already exist",
                J, K, G, GP
            )
        return gse, ket, HM


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    numx = 3
    numy = 4
    solver0 = JKGModelSolver(numx, numy)
    solver1 = JKGGPModelSolver(numx, numy)

    solver0.GS(alpha=1.2, beta=0.8)
    solver0.GS(alpha=1.2, beta=0.8)
    solver0.GS(alpha=1.0, beta=0.6)
    solver0.GS(alpha=1.0, beta=0.7)

    solver1.GS(J=-1, K=-6, G=8, GP=-4)
    solver1.GS(J=-1, K=-6, G=8, GP=-4)
    solver1.GS(J=1, K=6, G=-8, GP=4)
    solver1.GS(J=1, K=6, G=-8, GP=5)
