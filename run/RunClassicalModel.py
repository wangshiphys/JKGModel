import warnings
from time import time

import numpy as np
import tables as tb
from scipy.optimize import basinhopping

from ClassicalModel import *

warnings.filterwarnings("ignore", category=tb.NaturalNameWarning)

num0 = 12
num1 = 12
alpha = 0.50
beta = 0.50
h5f_name_temp = "OSC_num0={0}_num1={1}_alpha={2:.4f}_beta={3:.4f}.h5"
h5f_name = h5f_name_temp.format(num0, num1, alpha, beta)

Hx, Hy, Hz = HMatrixGenerator(alpha, beta)
cluster, category, x_bonds, y_bonds, z_bonds = ClusterGenerator(num0, num1)
args = (x_bonds, y_bonds, z_bonds, Hx, Hy, Hz)

log = "Time spend on {0:0>3d}th run: {1:.3f}s."
for index in range(50):
    t0 = time()
    initial_spin_angles = np.pi * np.random.random(2 * cluster.point_num)
    initial_spin_angles[1::2] *= 2
    res = basinhopping(
        EnergyCore1, initial_spin_angles,
        niter=200, minimizer_kwargs={"args": args}
    )
    phis = res.x[0::2]
    thetas = res.x[1::2]
    sin_phis = np.sin(phis)
    cos_phis = np.cos(phis)
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)

    energy_per_site = res.fun / cluster.point_num
    cluster_vectors = np.array(
        [sin_phis * cos_thetas, sin_phis * sin_thetas, cos_phis]
    ).T
    print(res.message[0])
    print("Energy Per-Site: {0}".format(energy_per_site))

    h5f = tb.open_file(h5f_name, mode="a")
    try:
        current_count = h5f.get_node_attr("/", "count")
    except AttributeError:
        current_count = 0
        h5f.set_node_attr("/", "num0", num0)
        h5f.set_node_attr("/", "num1", num1)
        h5f.set_node_attr("/", "beta", beta)
        h5f.set_node_attr("/", "alpha", alpha)
        h5f.set_node_attr("/", "count", current_count)

    spin_config_carray = h5f.create_carray(
        "/", "Run{0:0>4d}".format(current_count + 1), obj=cluster_vectors,
    )
    h5f.set_node_attr("/", "count", current_count + 1)
    h5f.set_node_attr(spin_config_carray, "message", res.message[0])
    h5f.set_node_attr(spin_config_carray, "energy", energy_per_site)
    h5f.close()
    t1 = time()
    print(log.format(index, t1 - t0), flush=True)
