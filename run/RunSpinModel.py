import logging
import sys
import matplotlib.pyplot as plt
import numpy as np

from SpinModel import *
from utilities import derivation


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.info("Program start running")

num1 = 3
num2 = 4
which = "xz"
alpha = 0.5
betas = np.arange(0, 2, 0.01)

solver = JKGModelEDSolver(num1, num2, which=which)
for beta in betas:
    solver.GS(alpha=alpha, beta=beta)

gses = []
es_path = "data/QuantumSpinModel/ES/"
es_name_temp = "ES_" + solver.identity + "_alpha={0:.4f}_beta={1:.4f}.npz"
for beta in betas:
    es_full_name = es_path + es_name_temp.format(alpha, beta)
    with np.load(es_full_name) as ld:
        gses.append(np.sort(ld["values"])[0])
gses = np.array(gses, dtype=np.float64)
d2betas, d2gses = derivation(betas, gses, nth=2)

fig, ax_gses = plt.subplots(num=solver.identity)
ax_d2gses = ax_gses.twinx()
color_gses = "tab:blue"
color_d2gses = "tab:orange"
ax_gses.plot(betas, gses, color=color_gses)
ax_d2gses.plot(d2betas, -d2gses / (np.pi ** 2), color=color_d2gses)

ax_gses.set_xlim(betas[0], betas[-1])
ax_gses.set_title(r"$\alpha={0:.4f}\pi$".format(alpha), fontsize="xx-large")
ax_gses.set_xlabel(r"$\beta/\pi$", fontsize="xx-large")
ax_gses.set_ylabel("E", rotation=0, fontsize="xx-large", color=color_gses)
ax_gses.tick_params("y", colors=color_gses)
ax_gses.grid(ls="dashed", axis="x", color="gray")
ax_gses.grid(ls="dashed", axis="y", color=color_gses)

ax_d2gses.set_ylabel(
    r"$E^{''}$", rotation=0, fontsize="xx-large", color=color_d2gses
)
ax_d2gses.tick_params("y", colors=color_d2gses)
ax_d2gses.grid(ls="dashed", axis="y", color=color_d2gses)
plt.get_current_fig_manager().window.showMaximized()
plt.show()
plt.close("all")

logging.info("Program stop running")
