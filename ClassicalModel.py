"""
Analysing the classical J-K-Gamma (J-K-G) model.

The spins are viewed as unit-vectors in 3-dimension.
"""


__all__ = ["JKGModelClassicalSolver"]


import warnings
from pathlib import Path

import numpy as np
import tables as tb
from scipy.optimize import basinhopping

from utilities import TriangularLattice

warnings.filterwarnings("ignore", category=tb.NaturalNameWarning)


# Calculate the energy of the classical J-K-G model with respect to the given
# spin-vectors. The spin-vectors are specified by the `spin_angles` parameters.
def _EnergyForGeneralSpinConfig(spin_angles, all_bonds, all_Hs):
    phis = spin_angles[0::2]
    thetas = spin_angles[1::2]
    sin_phis = np.sin(phis)
    cos_phis = np.cos(phis)
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)

    energy = 0.0
    vectors = np.array([sin_phis * sin_thetas, sin_phis * cos_thetas, cos_phis])
    for bonds, H in zip(all_bonds, all_Hs):
        tmp = vectors[:, bonds]
        energy += np.sum(tmp[:, :, 0] * np.dot(H, tmp[:, :, 1]))
    return energy


class JKGModelClassicalSolver(TriangularLattice):
    def OptimizeSpinConfig(
            self, data_path="data/ClassicalSpinModel/OptimizedSpinConfig/",
            niter=200, **model_params,
    ):
        """
        Optimize spin configuration for classical J-K-Gamma model.

        This method saves the optimized spin configuration to hdf5 file with
        name:
        `OSC_num1={num1}_num2={num2}_direction={direction}_alpha={alpha:.4f}_beta={beta:.4f}.h5`
        The variables in the "{}"s are replaced with the actual variables.
        For more information about hdf5 file, see the documentation of PyTables.

        Parameters
        ----------
        data_path : str, optional
            Where to save the optimized spin configuration. It can be an
            absolute path or a path relative to the current working
            directory (CWD). The specified `data_path` will be created if
            necessary.
            Default: "data/ClassicalSpinModel/OptimizedSpinConfig/".
        niter : int, optional
            The number of basin-hopping iterations. For more information,
            see also `scipy.optimize.basinhopping`.
            Default: 200.
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
        energy_per_site : float
            Energy per lattice site.
        optimized_spin_config : np.ndarray
            The optimized spin configuration.
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
        GP = 0.0

        Hijx = np.array([[J + K, GP, GP], [GP, J, G], [GP, G, J]])
        Hijy = np.array([[J, GP, G], [GP, J + K, GP], [G, GP, J]])
        Hijz = np.array([[J, G, GP], [G, J, GP], [GP, GP, J + K]])
        args = (self.all_bonds, (Hijx, Hijy, Hijz))

        initial_spin_angles = np.pi * np.random.random(2 * self.site_num)
        initial_spin_angles[1::2] *= 2
        res = basinhopping(
            _EnergyForGeneralSpinConfig, initial_spin_angles,
            niter=niter, minimizer_kwargs={"args": args},
        )
        phis = res.x[0::2]
        thetas = res.x[1::2]
        sin_phis = np.sin(phis)
        cos_phis = np.cos(phis)
        sin_thetas = np.sin(thetas)
        cos_thetas = np.cos(thetas)
        optimized_spin_config = np.array(
            [sin_phis * sin_thetas, sin_phis * cos_thetas, cos_phis],
            dtype=np.float64
        ).T
        energy_per_site = res.fun / self.site_num

        # Save the optimized spin configuration
        Path(data_path).mkdir(exist_ok=True, parents=True)
        h5f_name = "OSC_" + self.identity
        h5f_name += "_alpha={0:.4f}_beta={1:.4f}.h5".format(alpha, beta)
        h5f = tb.open_file(data_path + h5f_name, mode="a")
        try:
            current_count = h5f.get_node_attr("/", "count")
        except AttributeError:
            current_count = 0
            h5f.set_node_attr("/", "count", current_count)
            h5f.set_node_attr("/", "num1", self.num1)
            h5f.set_node_attr("/", "num2", self.num2)
            h5f.set_node_attr("/", "site_num", self.site_num)
            h5f.set_node_attr("/", "direction", self.direction)
            h5f.set_node_attr("/", "alpha", alpha)
            h5f.set_node_attr("/", "beta", beta)

        spin_config_carray = h5f.create_carray(
            "/", "Run{0:0>4d}".format(current_count + 1),
            obj=optimized_spin_config,
        )
        h5f.set_node_attr(spin_config_carray, "message", res.message)
        h5f.set_node_attr(spin_config_carray, "energy", energy_per_site)
        h5f.set_node_attr("/", "count", current_count + 1)
        h5f.close()

        return energy_per_site, optimized_spin_config
