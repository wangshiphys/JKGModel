import logging
import sys
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np

from MomentDirection import StripeDirection

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s",
)
logging.info("Program start running")
log_msg = "Probs for alpha=%.4f, beta=%.4f, direction=%s: dt=%.3fs"

ES_DATA_PATH = "data/QuantumSpinModel/ES/"
PS_DATA_PATH = "data/QuantumSpinModel/Probs/TestStripe/"
Path(PS_DATA_PATH).mkdir(exist_ok=True, parents=True)

lattice_id = "num1={0}_num2={1}_direction={2}"
ES_DATA_NAME_TEMP = "ES_" + lattice_id + "_alpha={3:.4f}_beta={4:.4f}.npz"
PS_DATA_NAME_TEMP = "PS_" + lattice_id + "_config={3}" \
                                         "_alpha={4:.4f}_beta={5:.4f}.npz"

step = 0.01
thetas = np.pi * np.arange(0.0, 1.0 + step, step)
phis = np.pi * np.arange(0.0, 2.0 + step, step)

num1 = 4
num2 = 6
alpha = 0.50
beta = 0.75
config = "StripeX"

total_probs = 0.0
for direction in ("xy", "xz", "yx", "yz", "zx", "zy"):
    es_data_name = ES_DATA_NAME_TEMP.format(num1, num2, direction, alpha, beta)
    ps_data_name = PS_DATA_NAME_TEMP.format(
        num1, num2, direction, config, alpha, beta
    )

    with np.load(ES_DATA_PATH + es_data_name) as ld:
        ket = ld["vectors"][:, 0]

    t0 = time()
    probabilities = StripeDirection(
        thetas, phis, ket, config=config,
        num1=num1, num2=num2, direction=direction,
    )
    np.savez(
        PS_DATA_PATH + ps_data_name,
        size=[num1, num2], direction=[direction],
        parameters=[alpha, beta], config=[config],
        thetas=thetas, phis=phis, probabilities=probabilities,
    )
    t1 = time()
    total_probs += probabilities
    logging.info(log_msg, alpha, beta, direction, t1 - t0)

avg_probs = total_probs / 6
ps_data_name = PS_DATA_NAME_TEMP.format(num1, num2, "avg", config, alpha, beta)
np.savez(
    PS_DATA_PATH +ps_data_name,
    size=[num1, num2], direction=["avg"],
    parameters=[alpha, beta], config=[config],
    thetas=thetas, phis=phis, probabilities=avg_probs,
)

fig, ax = plt.subplots()
cs = ax.pcolormesh(
    phis / np.pi, thetas / np.pi, avg_probs,
    zorder=0, cmap="magma", shading="gouraud",
)
colorbar = fig.colorbar(cs, ax=ax)

ax.set_aspect("equal")
ax.grid(True, ls="dashed", color="gray")
ax.set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
ax.set_title(r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(alpha, beta))

plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
plt.close("all")

logging.info("Program stop running")
