import logging
import sys
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np

from MomentDirection import FMDirection

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s",
)
logging.info("Program start running")

PS_DATA_PATH = "TestFM/"
Path(PS_DATA_PATH).mkdir(exist_ok=True)
ES_DATA_PATH = "C:/Users/swang/Desktop/JKGModel/data/QuantumSpinModel/ES/"
ES_DATA_NAME_TEMP = "ES_num1={0}_num2={1}_which={2}_alpha={3:.4f}_beta={4:.4f}.npz"
PS_DATA_NAME_TEMP = "PS_num1={0}_num2={1}_which={2}_alpha={3:.4f}_beta={4:.4f}.npz"

# Prepare thetas and phis
step = 0.01
thetas = np.pi * np.arange(0.0, 1.0 + step, step)
phis = np.pi * np.arange(0.0, 2.0 + step, step)

num1 = 4
num2 = 6
alpha = 0.30
beta = 1.30
which = "xy"
es_data_name = ES_DATA_NAME_TEMP.format(num1, num2, which, alpha, beta)
ps_data_name = PS_DATA_NAME_TEMP.format(num1, num2, which, alpha, beta)
ps_fig_name = ps_data_name.replace("npz", "png")

with np.load(ES_DATA_PATH + es_data_name) as ld:
    ket = ld["vectors"][:, 0]

t0 = time()
probabilities = FMDirection(thetas, phis, ket, num1*num2)
np.savez(
    PS_DATA_PATH + ps_data_name,
    size=[num1, num2], which=[which], parameters=[alpha, beta],
    thetas=thetas, phis=phis, probabilities=probabilities,
)
t1 = time()
logging.info(
    "Total time spend on alpha=%.3f, beta=%.3f: %.4fs", alpha, beta, t1 - t0
)

fig, ax = plt.subplots()
cs = ax.pcolormesh(
    phis / np.pi, thetas / np.pi, probabilities,
    zorder=0, cmap="magma", shading="gouraud",
)
colorbar = fig.colorbar(cs, ax=ax)

ax.set_aspect("equal")
ax.grid(True, ls="dashed", color="gray")
ax.set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
ax.set_title(r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(alpha, beta))
plt.get_current_fig_manager().window.showMaximized()
plt.show()
fig.savefig(PS_DATA_PATH + ps_fig_name, dpi=200, transparent=True)
plt.close("all")

logging.info("Program stop running")
