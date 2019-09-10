"""
Analysing the classical J-K-Gamma-Gamma'(J-K-G-GP) model.

The spins are viewed as unit-vectors in 3-dimension.
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from cluster import TriangularCluster


class ClassicalJKGGPModel(TriangularCluster):
    def _HijGenerator(self, model_params):
        actual_model_params = dict(self.DEFAULT_MODEL_PARAMETERS)
        actual_model_params.update(model_params)
        J = actual_model_params["J"]
        K = actual_model_params["K"]
        G = actual_model_params["G"]
        GP = actual_model_params["GP"]

        Hijx = np.array(
            [[J + K, GP, GP], [GP, J, G], [GP, G, J]], dtype=np.float64
        )
        Hijy = np.array(
            [[J, GP, G], [GP, J + K, GP], [G, GP, J]], dtype=np.float64
        )
        Hijz = np.array(
            [[J, G, GP], [G, J, GP], [GP, GP, J + K]], dtype=np.float64
        )
        return Hijx, Hijy, Hijz

    def GeneralSpinConfig(self, spin_angles, model_params):
        phis = np.pi * spin_angles[0::2]
        thetas = 2 * np.pi * spin_angles[1::2]
        sin_phis = np.sin(phis)
        cos_phis = np.cos(phis)
        sin_thetas = np.sin(thetas)
        cos_thetas = np.cos(thetas)
        spin_vectors = np.stack(
            [sin_phis * sin_thetas, sin_phis * cos_thetas, cos_phis]
        ).T

        energy = 0.0
        Hijx, Hijy, Hijz = self._HijGenerator(model_params)
        for bonds, Hij in zip(self.all_bonds, [Hijx, Hijy, Hijz]):
            bonds = np.array(bonds)
            spins0 = spin_vectors[bonds[:, 0]]
            spins1 = spin_vectors[bonds[:, 1]]
            energy += np.sum(spins0 * np.dot(spins1, Hij))
        return energy

    def FMConfig(self, spin_angle, model_params):
        phi = np.pi * spin_angle[0]
        theta = 2 * np.pi * spin_angle[1]
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        spin_vector = [sin_phi * sin_theta, sin_phi * cos_theta, cos_phi]

        x_bond_num = len(self._x_bonds)
        y_bond_num = len(self._y_bonds)
        z_bond_num = len(self._z_bonds)
        Hijx, Hijy, Hijz = self._HijGenerator(model_params)
        H = x_bond_num * Hijx + y_bond_num * Hijy + z_bond_num * Hijz

        return np.dot(spin_vector, np.dot(H, spin_vector))

    def Show3DVectorField(self, vectors, title=""):
        points = np.zeros((self.site_num, 3), dtype=np.float64)
        points[:, 0] = self.cluster.points[:, 0]
        points[:, 1] = self.cluster.points[:, 1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            points[:, 0], points[:, 1], points[:, 2],
            ls="", marker="o", color="k", ms=10,
        )
        ax.quiver(
            points[:, 0], points[:, 1], points[:, 2],
            vectors[:, 0], vectors[:, 1], vectors[:, 2],
            length=0.5, arrow_length_ratio=0.5, pivot="middle",
        )
        ax.set_zlim(-0.5, 0.5)
        ax.set_title(title, fontsize="xx-large")
        try:
            plt.get_current_fig_manager().window.showMaximized()
        except Exception:
            pass
        plt.show()
        plt.close("all")

    def MinimizeGeneralSpinConfig(self, model_params):
        initial_angles = np.random.random(2 * self.site_num)
        bounds = [(0, 1), ] * 2 * self.site_num
        res = minimize(
            self.GeneralSpinConfig, initial_angles,
            args=(model_params, ), bounds=bounds
        )
        if res.success:
            phis = np.pi * res.x[0::2]
            thetas = 2 * np.pi * res.x[1::2]
            xs = np.sin(phis) * np.sin(thetas)
            ys = np.sin(phis) * np.cos(thetas)
            zs = np.cos(phis)
            spin_vectors = np.stack([xs, ys, zs]).T
            title = "E={0:.8f}".format(res.fun)
            self.Show3DVectorField(spin_vectors, title=title)
        else:
            print("Failed to find a local minimum!")

    def MimimizeFMConfig(self, model_params):
        initial_angle = np.random.random(2)
        bounds = [(0, 1), (0, 1)]
        res = minimize(
            self.FMConfig, initial_angle,
            args=(model_params, ), bounds=bounds
        )
        title_temp = r"E={0:.8f}, $\alpha={1:.4f}\pi$, $\beta={2:.4f}\pi$"
        if res.success:
            phi = np.pi * res.x[0]
            theta = 2 * np.pi * res.x[1]
            spin_vectors = np.zeros((self.site_num, 3), dtype=np.float64)
            spin_vectors[:, 0] = np.sin(phi) * np.sin(theta)
            spin_vectors[:, 1] = np.sin(phi) * np.cos(theta)
            spin_vectors[:, 2] = np.cos(phi)
            title = title_temp.format(res.fun, res.x[0], 2 * res.x[1])
            self.Show3DVectorField(spin_vectors, title=title)
        else:
            print("Failed to find a local minimum!")

    def FMConfigEnergyVsDirection(self, model_params, mesh=200):
        x_bond_num = len(self._x_bonds)
        y_bond_num = len(self._y_bonds)
        z_bond_num = len(self._z_bonds)
        Hijx, Hijy, Hijz = self._HijGenerator(model_params)
        H = x_bond_num * Hijx + y_bond_num * Hijy + z_bond_num * Hijz

        phis = np.linspace(0, 1, num=mesh)
        thetas = np.linspace(0, 2, num=2*mesh)
        mesh_phis, mesh_thetas = np.meshgrid(phis, thetas)
        mesh_xs = np.sin(mesh_phis * np.pi) * np.sin(mesh_thetas * np.pi)
        mesh_ys = np.sin(mesh_phis * np.pi) * np.cos(mesh_thetas * np.pi)
        mesh_zs = np.cos(mesh_phis * np.pi)
        mesh_vectors = np.dstack([mesh_xs, mesh_ys, mesh_zs])
        mesh_energies = np.sum(mesh_vectors * np.dot(mesh_vectors, H), axis=-1)
        del mesh_xs, mesh_ys, mesh_zs, mesh_vectors

        vmin = np.min(mesh_energies)
        vmax = np.max(mesh_energies)
        amax = np.max(np.abs([vmin, vmax]))
        print("Minimum={0}, Maximum={1}".format(vmin, vmax))

        fig = plt.figure()
        ax0 = fig.add_subplot(121, projection="3d")
        surf = ax0.plot_surface(
            mesh_phis, mesh_thetas, mesh_energies,
            rcount=100, ccount=100,
            cmap="seismic",
            vmin=-amax, vmax=amax,
        )
        fig.colorbar(surf, ax=ax0)

        ax1 = fig.add_subplot(122)
        contour_sets = ax1.contourf(
            mesh_phis, mesh_thetas, mesh_energies,
            cmap="seismic", levels=500,
            vmin=-amax, vmax=amax,
        )
        phis_tmp = np.arctan2(
            1, -np.sin(thetas * np.pi) - np.cos(thetas * np.pi)
        ) / np.pi
        ax1.plot(phis_tmp, thetas, ls="dashed")
        fig.colorbar(contour_sets, ax=ax1)

        try:
            plt.get_current_fig_manager().window.showMaximized()
        except Exception:
            pass
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    max_iter = 30
    numx = numy = 6
    alpha = 1.0
    beta = 1.6
    model_params = {
        "J": np.sin(alpha * np.pi) * np.sin(beta * np.pi),
        "K": np.sin(alpha * np.pi) * np.cos(beta * np.pi),
        "G": np.cos(alpha * np.pi),
        "GP": 0.0,
    }

    solver = ClassicalJKGGPModel(numx=numx, numy=numy)
    solver.FMConfigEnergyVsDirection(model_params)

    for i in range(max_iter):
        solver.MimimizeFMConfig(model_params)
    for i in range(max_iter):
        solver.MinimizeGeneralSpinConfig(model_params)
