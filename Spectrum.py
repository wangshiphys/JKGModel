import logging
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from pathlib import Path
from HamiltonianPy import SpinOperator, TRIANGLE_CELL_BS

from SpinModel import JKGModelEDSolver
from GFSolver import GFSolverLanczosMultiple

warnings.filterwarnings("ignore", category=tb.NaturalNameWarning)


def _SpinOperator2String(operator):
    return ",".join(
        [
            "otype={0}".format(operator.otype),
            "site=({0})".format(
                ",".join("{0:.6f}".format(coord) for coord in operator.site)
            )
        ]
    )


def _save_lanczos_projection(name, projected_matrices, projected_vectors):
    hdf5 = tb.open_file(name, mode="w")
    vectors_group = hdf5.create_group("/", "projected_vectors")
    matrices_group = hdf5.create_group("/", "projected_matrices")
    for key in projected_matrices:
        carray = hdf5.create_carray(
            matrices_group, _SpinOperator2String(key),
            obj=projected_matrices[key]
        )
        hdf5.set_node_attr(carray, "site", key.site)
        hdf5.set_node_attr(carray, "otype", key.otype)
    for key0 in projected_vectors:
        sub_group_name = _SpinOperator2String(key0)
        sub_group = hdf5.create_group(vectors_group, sub_group_name)
        hdf5.set_node_attr(sub_group, "site", key0.site)
        hdf5.set_node_attr(sub_group, "otype", key0.otype)
        for key1 in projected_vectors[key0]:
            carray = hdf5.create_carray(
                sub_group, _SpinOperator2String(key1),
                obj=projected_vectors[key0][key1],
            )
            hdf5.set_node_attr(carray, "site", key1.site)
            hdf5.set_node_attr(carray, "otype", key1.otype)
    hdf5.close()


def _read_lanczos_projection(name):
    projected_vectors = dict()
    projected_matrices = dict()
    hdf5 = tb.open_file(name, mode="r")
    vectors_group = hdf5.get_node("/", "projected_vectors")
    matrices_group = hdf5.get_node("/", "projected_matrices")
    for carray in matrices_group._f_iter_nodes():
        site = hdf5.get_node_attr(carray, "site")
        otype = hdf5.get_node_attr(carray, "otype")
        operator = SpinOperator(otype=otype, site=site)
        projected_matrices[operator] = carray.read()
    for sub_group in vectors_group._f_iter_nodes():
        site0 = hdf5.get_node_attr(sub_group, "site")
        otype0 = hdf5.get_node_attr(sub_group, "otype")
        operator0 = SpinOperator(otype=otype0, site=site0)
        projected_vectors[operator0] = dict()
        for carray in sub_group._f_iter_nodes():
            site1 = hdf5.get_node_attr(carray, "site")
            otype1 = hdf5.get_node_attr(carray, "otype")
            operator1 = SpinOperator(otype=otype1, site=site1)
            projected_vectors[operator0][operator1] = carray.read()
    hdf5.close()
    return projected_matrices, projected_vectors


class SpectrumSolver(JKGModelEDSolver):
    def _real_space_gfs(self, omegas, eta=0.01, tol=0.0, **model_params):
        actual_model_params = dict(self.DEFAULT_MODEL_PARAMETERS)
        actual_model_params.update(model_params)
        alpha = actual_model_params["alpha"]
        beta = actual_model_params["beta"]

        values, vectors, HM = self.GS(k=1, tol=tol, **model_params)
        GE = values[0]
        gs_ket = vectors[:, [0]]
        As = [SpinOperator("p", site) for site in self._cluster.points]
        Bs = [SpinOperator("m", site) for site in self._cluster.points]

        projection_data_name = "_".join(
            [
                "tmp/Krylov", self.identity,
                "alpha={0:.4f}_beta={1:.4f}.hdf5".format(alpha, beta)
            ]
        )
        if Path(projection_data_name).exists():
            projected_matrices, projected_vectors = \
                _read_lanczos_projection(projection_data_name)
        else:
            projected_matrices, projected_vectors = self.LanczosProjection(
                HM=HM, gs_ket=gs_ket, which="all"
            )
            _save_lanczos_projection(
                projection_data_name, projected_matrices, projected_vectors
            )
        gfs_vs_omegas = GFSolverLanczosMultiple(
            omegas=omegas, As=As, Bs=Bs, GE=GE,
            projected_vectors=projected_vectors,
            projected_matrices=projected_matrices,
            eta=eta, sign="-", structure="dict",
        )
        return gfs_vs_omegas

    def gfs_vs_ks_omegas(self, ks, omegas, eta=0.01, tol=0.0, **model_params):
        gfs = 0.0
        gfs_vs_omegas = self._real_space_gfs(omegas, eta, tol, **model_params)
        for A, B in gfs_vs_omegas:
            site0 = A.site
            site1 = B.site
            phase_factors = np.exp(1j * np.dot(ks, site1 - site0))
            gfs += np.outer(gfs_vs_omegas[(A, B)], phase_factors)
        return gfs / self.site_num


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    num1 = 3
    num2 = 4
    which = "xy"
    eta = 0.05
    alpha = 0.50
    beta = 0.80

    step = 0.005
    omegas = np.arange(-0.1, 4 + step, step)
    kpoint_ids = [[i, j] for i in range(num1 + 1) for j in range(num2 + 1)]
    kpoints_num = len(kpoint_ids)
    kpoints = np.array(
        [np.dot([i / num1, j / num2], TRIANGLE_CELL_BS) for i, j in kpoint_ids]
    )

    solver = SpectrumSolver(num1=num1, num2=num2, which=which)
    gfs = solver.gfs_vs_ks_omegas(kpoints, omegas, eta, alpha=alpha, beta=beta)
    spectrum = -gfs.imag / np.pi

    fig, ax = plt.subplots()
    cs = ax.contourf(
        np.arange(kpoints_num), omegas, spectrum, cmap="hot_r", levels=500
    )
    fig.colorbar(cs, ax=ax)

    ax.set_xticks(range(len(kpoints)))
    ax.set_xticklabels(
        ["({0},{1})".format(i, j) for i, j in kpoint_ids], rotation=45
    )
    ax.grid(True, ls="dashed", color="gray")

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
