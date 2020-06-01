"""
Calculate spin excitation spectrum of the J-K-Gamma model on triangular lattice.
"""


import warnings
from pathlib import Path

import HamiltonianPy as HP
import numpy as np
import tables as tb

from SpinModel import JKGModelEDSolver

warnings.filterwarnings("ignore", category=tb.NaturalNameWarning)


def _save_lanczos_projection(name, GE, projected_matrices, projected_vectors):
    hdf5 = tb.open_file(name, mode="w")
    hdf5.set_node_attr("/", "GE", GE)

    vectors_group = hdf5.create_group("/", "projected_vectors")
    matrices_group = hdf5.create_group("/", "projected_matrices")
    for key in projected_matrices:
        carray = hdf5.create_carray(
            matrices_group, str(key), obj=projected_matrices[key]
        )
        hdf5.set_node_attr(carray, "site", key.site)
        hdf5.set_node_attr(carray, "otype", key.otype)

    for key0 in projected_vectors:
        sub_group = hdf5.create_group(vectors_group, str(key0))
        hdf5.set_node_attr(sub_group, "site", key0.site)
        hdf5.set_node_attr(sub_group, "otype", key0.otype)
        for key1 in projected_vectors[key0]:
            carray = hdf5.create_carray(
                sub_group, str(key1), obj=projected_vectors[key0][key1],
            )
            hdf5.set_node_attr(carray, "site", key1.site)
            hdf5.set_node_attr(carray, "otype", key1.otype)
    hdf5.close()


def _read_lanczos_projection(name):
    projected_vectors = dict()
    projected_matrices = dict()
    hdf5 = tb.open_file(name, mode="r")
    GE = hdf5.get_node_attr("/", "GE")

    vectors_group = hdf5.get_node("/", "projected_vectors")
    matrices_group = hdf5.get_node("/", "projected_matrices")
    for carray in matrices_group._f_iter_nodes():
        site = hdf5.get_node_attr(carray, "site")
        otype = hdf5.get_node_attr(carray, "otype")
        key = HP.SpinOperator(otype=otype, site=site)
        projected_matrices[key] = carray.read()

    for sub_group in vectors_group._f_iter_nodes():
        site0 = hdf5.get_node_attr(sub_group, "site")
        otype0 = hdf5.get_node_attr(sub_group, "otype")
        key0 = HP.SpinOperator(otype=otype0, site=site0)
        projected_vectors[key0] = dict()
        for carray in sub_group._f_iter_nodes():
            site1 = hdf5.get_node_attr(carray, "site")
            otype1 = hdf5.get_node_attr(carray, "otype")
            key1 = HP.SpinOperator(otype=otype1, site=site1)
            projected_vectors[key0][key1] = carray.read()
    hdf5.close()
    return GE, projected_matrices, projected_vectors


class SpectrumSolver(JKGModelEDSolver):
    def ClusterGF(
            self, omegas, excitation="Sm", eta=0.01, tol=0.0, **model_params
    ):
        if excitation == "Sm":
            As = [HP.SpinOperator("p", site) for site in self.cluster.points]
            Bs = [HP.SpinOperator("m", site) for site in self.cluster.points]
        elif excitation == "Sz":
            As = [HP.SpinOperator("z", site) for site in self.cluster.points]
            Bs = [HP.SpinOperator("z", site) for site in self.cluster.points]
        else:
            raise ValueError("Invalid `excitation`: {0}.".format(excitation))

        actual_model_params = dict(self.DEFAULT_MODEL_PARAMETERS)
        actual_model_params.update(model_params)
        alpha = actual_model_params["alpha"]
        beta = actual_model_params["beta"]
        projection_data_name = "_".join(
            [
                "tmp/Krylov", self.identity,
                "alpha={0:.4f}_beta={1:.4f}".format(alpha, beta),
                "excitation={0}.hdf5".format(excitation),
            ]
        )
        if Path(projection_data_name).exists():
            GE, projected_matrices, projected_vectors = \
                _read_lanczos_projection(projection_data_name)
        else:
            values, vectors, HM = self.EigenStates(k=1, tol=tol, **model_params)
            GE = values[0]
            gs_ket = vectors[:, 0]
            projected_matrices, projected_vectors = self.LanczosProjection(
                HM=HM, gs_ket=gs_ket, excitation=excitation
            )
            Path("tmp/").mkdir(parents=True, exist_ok=True)
            _save_lanczos_projection(
                projection_data_name, GE, projected_matrices, projected_vectors
            )

        gfs = HP.GFSolverLanczosMultiple(
            omegas=omegas, As=As, Bs=Bs, GE=GE,
            projected_vectors=projected_vectors,
            projected_matrices=projected_matrices,
            eta=eta, sign="-", structure="dict",
        )
        return gfs

    def ExcitationSpectrum(
            self, kpoints, omegas, excitation="Sm",
            eta=0.01, tol=0.0, **model_params
    ):
        cluster_gfs = self.ClusterGF(
            omegas, excitation, eta, tol, **model_params
        )

        spectrum = 0.0
        for A, B in cluster_gfs:
            p0 = A.site
            p1 = B.site
            phase_factors = np.exp(1j * np.dot(kpoints, p1 - p0))
            spectrum += np.outer(cluster_gfs[(A, B)], phase_factors).imag
        return -spectrum / (np.pi * self.site_num)
