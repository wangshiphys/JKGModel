import matplotlib.pyplot as plt
import numpy as np

from FontSize import *

data_name = "data/Spectrum_num1=4_num2=6_direction=xy_" \
            "alpha=0.0500_beta=0.1000_excitation=Sm.npz"

with np.load(data_name) as ld:
    omegas = ld["omegas"]
    ids = ld["kpoint_ids"]
    spectrum = ld["spectrum"]
kpoint_num = ids.shape[0]

fig, ax = plt.subplots(num="Spectrum")
im = ax.pcolormesh(
    range(kpoint_num + 1), omegas, spectrum,
    cmap="hot", edgecolor="face",
    vmin=0, vmax=3,
)
im.set_edgecolor("face")
colorbar = fig.colorbar(im, ax=ax, pad=0.01)
colorbar.ax.tick_params(axis="y", labelsize=SMALL)
ax.set_ylim(0, 2)
ax.grid(ls="dashed")
ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
ax.set_xticks(np.arange(kpoint_num) + 0.5)
ytick_labels = ["0.0", "0.5", "1.0", "1.5", "2.0"]
xtick_labels = [
    "$\~M$", "", "", r"$\~\Gamma$", "", "", "$\~M$", "", "",  r"$\~\Gamma$"
]
ax.set_xticklabels(xtick_labels, fontsize=SMALL)
ax.set_yticklabels(ytick_labels, fontsize=SMALL)

plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
fig.savefig("figures/Spectrum.pdf", transparent=True)
plt.close("all")
