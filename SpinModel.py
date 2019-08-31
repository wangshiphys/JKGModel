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
from HamiltonianPy import lattice_generator, SpinInteraction
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigsh

logging.getLogger(__name__).addHandler(logging.NullHandler())
warnings.filterwarnings("ignore", category=tb.NaturalNameWarning)


class JKGModelSolver:
    """
    Exact diagonalization of the J-K-Gamma(J-K-G) model on triangular lattice.

    Attributes
    ----------
    numx: int
        The number of lattice site along the 1st translation vector.
    numy: int
        The number of lattice site along the 2nd translation vector.
    site_num: int
        The number of lattice site of the system.
    """

    # Default model parameters
    # "alpha" and "beta" for the J-K-Gamma model
    # J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
    # K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
    # G = np.cos(alpha * np.pi)
    # "J", "K", "G" and "GP" for the J-K-Gamma-Gamma' model.
    # The default model parameters correspond to ferromagnetic Heisenberg model
    DEFAULT_MODEL_PARAMS = {
        "alpha": 0.5, "beta": -0.5,
        "J": -1.0, "K": 0.0, "G": 0.0, "GP": 0.0,
    }

    def __init__(self, numx, numy=None):
        """
        Customize the newly created instance.

        Parameters
        ----------
        numx : int
            The number of lattice site along the 1st translation vector.
        numy : int or None, optional
            The number of lattice site along the 2nd translation vector.
            Default: numy = numx.
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
        The `numx` attribute.
        """

        return self._numx

    @property
    def numy(self):
        """
        The `numy` attribute.
        """

        return self._numy

    @property
    def site_num(self):
        """
        The `site_num` attribute.
        """

        return self._site_num

    def _bonds(self):
        # Generate all nearest neighbor bonds
        # Categorize the nearest neighbor bonds according to their direction
        cluster = lattice_generator(
            "triangle", num0=self._numx, num1=self._numy
        )
        intra, inter = cluster.bonds(nth=1)

        x_bonds = []
        y_bonds = []
        z_bonds = []
        for bond in intra + inter:
            p0, p1 = bond.endpoints
            index0 = cluster.getIndex(site=p0, fold=True)
            index1 = cluster.getIndex(site=p1, fold=True)
            bond_index = (index0, index1)

            azimuth = bond.getAzimuth(ndigits=0)
            if azimuth in (-180, 0, 180):
                x_bonds.append(bond_index)
            elif azimuth in (-120, 60):
                y_bonds.append(bond_index)
            elif azimuth in (-60, 120):
                z_bonds.append(bond_index)
            else:
                raise ValueError("Invalid azimuth: {0}".format(azimuth))
        return x_bonds, y_bonds, z_bonds

    def _TermMatrix(self):
        # Check whether the matrix representation for the J, K, Gamma term
        # exist on the "tmp/" directory relative to the current working
        # directory(CWD). If exist, load these matrices; if not, calculate
        # the matrix representation and save the result to the "tmp/"
        # directory relative CWD, the "tmp/" directory is created if necessary.

        directory = Path("tmp/")
        name_template = "TRIANGLE_NUMX={0}_NUMY={1}_H{{0}}.npz".format(
            self._numx, self._numy
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
            site_num = self._site_num
            all_bonds = self._bonds()
            configs = [("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y")]

            HJ = HK = HG = 0.0
            msg = "%s-bond: %2d/%2d, dt=%.6fs"
            m_func = SpinInteraction.matrix_function
            for (gamma, alpha, beta), bonds in zip(configs, all_bonds):
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
            save_npz(file_HJ, HJ, compressed=False)
            save_npz(file_HK, HK, compressed=False)
            save_npz(file_HG, HG, compressed=False)

        # Caching these matrices for reuse
        self._cache = (HJ, HK, HG)

    @staticmethod
    def _already_exist(hdf5_file_name, ket_name):
        if Path(hdf5_file_name).exists():
            h5f = tb.open_file(hdf5_file_name, mode="r")
            try:
                h5f.get_node("/", ket_name)
            except tb.NoSuchNodeError:
                exist = False
            else:
                exist = True
            h5f.close()
        else:
            exist = False
        return exist

    @staticmethod
    def _save_data(hdf5_file_name, ket_name, ket, attrs):
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
        """

        actual_model_params = dict(self.DEFAULT_MODEL_PARAMS)
        actual_model_params.update(model_params)
        alpha = actual_model_params["alpha"]
        beta = actual_model_params["beta"]

        prefix = "GS_numx={0}_numy={1}".format(self._numx, self._numy)
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
        if self._already_exist(hdf5_file_name, ket_name):
            msg = "GS for alpha=%.3f, beta=%.3f already exist"
            logger.info(msg, alpha, beta)
        else:
            J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
            K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
            G = np.cos(alpha * np.pi)

            if not hasattr(self, "_cache"):
                self._TermMatrix()
            HJ, HK, HG = self._cache

            t0 = time()
            gse, ket = eigsh(J * HJ + K * HK + G * HG, k=1, which="SA", tol=tol)
            t1 = time()
            logger.info("alpha=%.3f, beta=%.3f, dt=%.6fs", alpha, beta, t1 - t0)

            # Save the ground state data
            attrs = {
                "numx": self._numx, "numy": self._numy,
                "alpha": alpha, "beta": beta, "gse": gse,
            }
            self._save_data(hdf5_file_name, ket_name, ket, attrs=attrs)
            logger.info("Save GS data to %s:/%s", hdf5_file_name, ket_name)

    __call__ = GS


class JKGGPModelSolver(JKGModelSolver):
    """
    Exact diagonalization of the J-K-Gamma-Gamma' model on triangular lattice.

    Attributes
    ----------
    numx: int
        The number of lattice site along the 1st translation vector.
    numy: int
        The number of lattice site along the 2nd translation vector.
    site_num: int
        The number of lattice site of the system.
    """

    def _TermMatrix(self):
        # Check whether the matrix representation for the J, K, Gamma,
        # Gamma' term exist on the "tmp/" directory relative to the current
        # working directory(CWD). If exist, load these matrices; if not,
        # calculate the matrix representation and save the result to the
        # "tmp/" directory relative CWD, the "tmp/" directory is created if
        # necessary.

        # Prepare HJ, HK, HG first
        super()._TermMatrix()

        file_HGP = Path("tmp/") / "TRIANGLE_NUMX={0}_NUMY={1}_HGP.npz".format(
            self._numx, self._numy
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
            site_num = self._site_num
            all_bonds = self._bonds()
            configs = [("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y")]

            HGP = 0.0
            msg = "%s-bond: %2d/%2d, dt=%.6fs"
            m_func = SpinInteraction.matrix_function
            for (gamma, alpha, beta), bonds in zip(configs, all_bonds):
                bond_num = len(bonds)
                for count, (index0, index1) in enumerate(bonds, start=1):
                    t0 = time()
                    HGP += m_func([(index0, alpha), (index1, gamma)], site_num)
                    HGP += m_func([(index0, gamma), (index1, alpha)], site_num)
                    HGP += m_func([(index0, beta), (index1, gamma)], site_num)
                    HGP += m_func([(index0, gamma), (index1, beta)], site_num)
                    t1 = time()
                    logger.info(msg, gamma, count, bond_num, t1 - t0)
            save_npz(file_HGP, HGP, compressed=False)

        # The self._cache on the right side has already been set properly by
        # calling super()._TermMatrix()
        self._cache = self._cache + (HGP, )

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

        actual_model_params = dict(self.DEFAULT_MODEL_PARAMS)
        actual_model_params.update(model_params)
        J = actual_model_params["J"]
        K = actual_model_params["K"]
        G = actual_model_params["G"]
        GP = actual_model_params["GP"]

        prefix = "GS_numx={0}_numy={1}".format(self._numx, self._numy)
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
        if self._already_exist(hdf5_file_name, ket_name):
            msg = "GS for J=%.4f, K=%.4f, G=%.4f, GP=%.4f already exist"
            logger.info(msg, J, K, G, GP)
        else:
            if not hasattr(self, "_cache"):
                self._TermMatrix()
            HJ, HK, HG, HGP = self._cache

            t0 = time()
            gse, ket = eigsh(
                J * HJ + K * HK + G * HG + GP * HGP, k=1, which="SA", tol=tol
            )
            t1 = time()
            msg = "J=%.4f, K=%.4f, G=%.4f, GP=%.4f, dt=%.6fs"
            logger.info(msg, J, K, G, GP, t1 - t0)

            # Save the ground state data
            attrs = {
                "numx": self._numx, "numy": self._numy,
                "J": J, "K": K, "G": G, "GP": GP, "gse": gse,
            }
            self._save_data(hdf5_file_name, ket_name, ket, attrs=attrs)
            logger.info("Save GS data to %s:/%s", hdf5_file_name, ket_name)

    __call__ = GS


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

    solver0(alpha=1.2, beta=0.8)
    solver0(alpha=1.2, beta=0.8)
    solver0(alpha=1.0, beta=0.6)
    solver0(alpha=1.0, beta=0.7)

    solver1(J=-1, K=-6, G=8, GP=-4)
    solver1(J=-1, K=-6, G=8, GP=-4)
    solver1(J=1, K=6, G=-8, GP=4)
    solver1(J=1, K=6, G=-8, GP=5)
