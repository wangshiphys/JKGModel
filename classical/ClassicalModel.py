"""
Analysing the classical J-K-Gamma-Gamma' (J-K-G-GP) model.

The spins are viewed as unit-vectors in 3-dimension.
"""


import warnings
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from scipy.optimize import basinhopping

from utilities import *

warnings.filterwarnings("ignore", category=tb.NaturalNameWarning)


# Calculate the energy of the classical J-K-G-GP model with the given
# spin-vectors. The spin-vectors are specified by the `spin_angles` parameters.
def _EnergyForGeneralSpinConfig(spin_angles, all_bonds, all_Hs):
    phis = spin_angles[0::2]
    thetas = spin_angles[1::2]
    sin_phis = np.sin(phis)
    cos_phis = np.cos(phis)
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)
    vectors = np.array(
        [sin_phis * sin_thetas, sin_phis * cos_thetas, cos_phis],
        dtype=np.float64
    )

    energy = 0.0
    for bonds, H in zip(all_bonds, all_Hs):
        tmp = vectors[:, bonds]
        energy += np.sum(tmp[:, :, 0] * np.dot(H, tmp[:, :, 1]))
    return energy


# Calculate the energy of the classical J-K-G-GP model with the given
# spin-vectors. FM order is assumed.
def _EnergyForFMSpinConfig(spin_angle, H):
    phi = spin_angle[0]
    theta = spin_angle[1]
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    vector = [sin_phi * sin_theta, sin_phi * cos_theta, cos_phi]
    return np.dot(vector, np.dot(H, vector))


class ClassicalJKGGPModel(TriangularLattice):
    def MinimizeGeneralSpinConfig(
            self, alpha=0.5, beta=1.5, GP=0.0,
            J=-1.0, K=0.0, G=0.0, normalized_parameters=True,
            niter=200, show=True, **kwargs
    ):
        if normalized_parameters:
            J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
            K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
            G = np.cos(alpha * np.pi)
            model_params = {"alpha": alpha, "beta": beta, "GP": GP}
        else:
            model_params = {"J": J, "K": K, "G": G, "GP": GP}

        Hijx = np.array([[J + K, GP, GP], [GP, J, G], [GP, G, J]])
        Hijy = np.array([[J, GP, G], [GP, J + K, GP], [G, GP, J]])
        Hijz = np.array([[J, G, GP], [G, J, GP], [GP, GP, J + K]])
        args = (self.all_bonds, (Hijx, Hijy, Hijz))

        initial_spin_angles = np.pi * np.random.random(2 * self.site_num)
        initial_spin_angles[1::2] *= 2
        res = basinhopping(
            _EnergyForGeneralSpinConfig, initial_spin_angles,
            niter=niter, T=1.0, stepsize=0.5,
            minimizer_kwargs={"args": args}
        )
        phis = res.x[0::2]
        thetas = res.x[1::2]
        sin_phis = np.sin(phis)
        cos_phis = np.cos(phis)
        sin_thetas = np.sin(thetas)
        cos_thetas = np.cos(thetas)
        optimized_spin_vectors = np.array(
            [sin_phis * sin_thetas, sin_phis * cos_thetas, cos_phis],
            dtype=np.float64
        ).T
        energy_per_site = res.fun / self._site_num

        path = "data/GeneralSpinConfig/"
        Path(path).mkdir(exist_ok=True, parents=True)

        h5f_file_name = "".join(
            [
                path,
                "numx={0},numy={1},".format(self.numx, self.numy),
                ",".join(
                    "{0}={1:.3f}".format(key, value)
                    for key, value in model_params.items()
                ),
                ".h5"
            ]
        )
        h5f = tb.open_file(h5f_file_name, mode="a")
        try:
            current_count = h5f.get_node_attr("/", "count")
        except AttributeError:
            current_count = 0
            h5f.set_node_attr("/", "count", current_count)
            h5f.set_node_attr("/", "numx", self._numx)
            h5f.set_node_attr("/", "numy", self._numy)
            for key, value in model_params.items():
                h5f.set_node_attr("/", key, value)

        spin_vectors_carray = h5f.create_carray(
            "/", "Run{0:0>4d}".format(current_count + 1),
            obj=optimized_spin_vectors,
        )
        h5f.set_node_attr(spin_vectors_carray, "energy", energy_per_site)
        h5f.set_node_attr(spin_vectors_carray, "message", res.message)
        h5f.set_node_attr("/", "count", current_count + 1)
        h5f.close()

        if show:
            fig = plt.figure("OptimizedSpinConfigure")
            ax = fig.add_subplot(111, projection="3d")
            title = "E={0}\n".format(energy_per_site)
            title += ",".join(
                "{0}={1:.3f}".format(key, value)
                for key, value in model_params.items()
            )
            ShowVectorField3D(
                ax, self.cluster.points, optimized_spin_vectors,
                title=title, **kwargs
            )
            ax.set_zlim(-0.5, 0.5)
            try:
                plt.get_current_fig_manager().window.showMaximized()
            except Exception:
                pass
            plt.show()
            plt.close("all")


if __name__ == "__main__":
    import logging
    import sys
    from time import time

    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    max_run = 10
    numx = numy = 6
    solver = ClassicalJKGGPModel(numx, numy)
    for i in range(max_run):
        t0 = time()
        solver.MinimizeGeneralSpinConfig(
            J=1.0, K=-2.0,
            normalized_parameters=False, markersize=12
        )
        t1 = time()
        logging.info("%d/%d, dt=%.4fs", i, max_run, t1 - t0)
