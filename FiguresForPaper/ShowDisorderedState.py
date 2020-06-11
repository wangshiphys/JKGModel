"""
Show the disordered state for pure positive Gamma model.
"""

import matplotlib.pyplot as plt
import tables as tb

from HamiltonianPy import lattice_generator


which = 32
markersize = 12
cluster = lattice_generator("triangle", num0=12, num1=12)
xs = cluster.points[:, 0]
ys = cluster.points[:, 1]

h5f_name = "data/OSC_num1=12_num1=12_alpha=0.0000_beta=0.0000.h5"
h5f = tb.open_file(h5f_name, mode="r")
carray = h5f.get_node("/", name="Run{0:0>4d}".format(which))
spin_vectors = carray.read()
colors = 0.5 * spin_vectors + 0.5
h5f.close()

fig, ax = plt.subplots(num="Run{0:0>4d}".format(which))
ax.plot(xs, ys, ls="", marker="o", ms=markersize, color="black", zorder=0)
ax.quiver(
    xs, ys, spin_vectors[:, 0], spin_vectors[:, 1], color=colors,
    units="xy", scale_units="xy", scale=1, width=0.06, pivot="mid", zorder=1
)
ax.set_axis_off()
ax.set_aspect("equal")
fig.set_size_inches(16, 9.26)
plt.tight_layout()
plt.show()
print(fig.get_size_inches())
# fig.savefig("figures/DisorderedState.pdf", dpi=200)
# fig.savefig("figures/DisorderedState.png", dpi=200)
plt.close("all")
