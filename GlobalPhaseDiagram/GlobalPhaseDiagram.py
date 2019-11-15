import matplotlib.pyplot as plt
import numpy as np

from PhaseBoundaries import COLORS, Phase_Indices, \
    Phase_Transition_Points_Fixed_Alpha


key_template = "alpha={0:.3f}"
height = 0.005
extra_height = height / 5

xticks = np.arange(0, 2, 0.25)
fig, (ax_north, ax_south) = plt.subplots(
    nrows=1, ncols=2, subplot_kw={"polar": True}
)

r = 0
ax_north.bar(
    x=0, width=2 * np.pi, bottom=r, height=height / 2 + extra_height,
    color=COLORS[10], align="edge",
)
r += height
for i in range(3):
    ax_north.bar(
        x=0, width=2 * np.pi, bottom=r - height/2, height=height + extra_height,
        color=COLORS[10], align="edge",
    )
    r += height

alphas = np.arange(0.02, 0.505, 0.005)
for alpha in alphas:
    key = "alpha={0:.3f}".format(alpha)
    edges = np.array(Phase_Transition_Points_Fixed_Alpha[key])
    widths = np.append(edges[1:] - edges[:-1], 2 + edges[0] - edges[-1])
    ax_north.bar(
        x=edges * np.pi, width=widths * np.pi,
        bottom=r - height / 2, height=height + extra_height,
        color=COLORS[Phase_Indices[key]], align="edge",
    )
    r += height

ax_north.set_xticks(xticks * np.pi)
ax_north.set_xticklabels(
    [r"${0:.0f}^\circ$".format(beta * 180) for beta in xticks],
    ha="center", va="center", fontsize="medium",
)

yticks = np.array([0, 1/6, 1/3, 1/2])
ax_north.set_rticks(yticks)
ax_north.set_yticklabels(
    [r"${0:.0f}^\circ$".format(alpha * 180) for alpha in yticks],
    x=1.5 * np.pi, ha="center", va="center", fontsize="medium",
)

ax_north.spines["polar"].set_visible(False)
ax_north.grid(color="gray", ls="dashed")
ax_north.set_title(r"$\Gamma \geq 0$", fontsize="xx-large")

r = 0
ax_south.bar(
    x=0, width=2 * np.pi, bottom=r, height=height/2 + extra_height,
    color=COLORS[12], align="edge",
)
r += height
for i in range(3):
    ax_south.bar(
        x=0, width=2 * np.pi, bottom=r - height/2, height=height + extra_height,
        color=COLORS[12], align="edge",
    )
    r += height

alphas = np.arange(0.98, 0.495, -0.005)
for alpha in alphas:
    key = "alpha={0:.3f}".format(alpha)
    edges = np.array(Phase_Transition_Points_Fixed_Alpha[key])
    widths = np.append(edges[1:] - edges[:-1], 2 + edges[0] - edges[-1])
    ax_south.bar(
        x=edges * np.pi, width=widths * np.pi,
        bottom=r - height/2, height=height + extra_height,
        color=COLORS[Phase_Indices[key]], align="edge",
    )
    r += height

ax_south.set_xticks(xticks * np.pi)
ax_south.set_xticklabels(
    [r"${0:.0f}^\circ$".format(beta * 180) for beta in xticks],
    ha="center", va="center", fontsize="medium",
)

yticks = np.array([0, 1/6, 1/3, 1/2])
ytick_labels = 1 - yticks
ax_south.set_yticks(yticks)
ax_south.set_yticklabels(
    [r"${0:.0f}^\circ$".format(alpha * 180) for alpha in ytick_labels],
    x=1.5 * np.pi, ha="center", va="center", fontsize="medium",
)

ax_south.spines["polar"].set_visible(False)
ax_south.grid(color="gray", ls="dashed")
ax_south.set_title(r"$\Gamma \leq 0$", fontsize="xx-large")

phase_annotate_north = [
    (0.20 * np.pi, 0.25, r"$Z_2$-Vortex"),
    (0.49 * np.pi, 0.41, r"$120^\circ$ Neel"),
    (0.80 * np.pi, 0.33, "Stripe"),
    (1.38 * np.pi, 0.30, r"FM[111]$_\bot$"),
    (0.45 * np.pi, 0.045, "QSL"),
    (1.86 * np.pi, 0.45, "Spiral"),
]
for annotate in phase_annotate_north:
    ax_north.text(*annotate, ha="center", va="center", fontsize="x-large",)

phase_annotate_south = [
    (0.45 * np.pi, 0.23, r"Rotated Stripe"),
    (1.25 * np.pi, 0.25, r"FM[111]"),
]
for annotate in phase_annotate_south:
    ax_south.text(*annotate, ha="center", va="center", fontsize="x-large",)

plt.get_current_fig_manager().window.showMaximized()
plt.show()
fig.savefig("GlobalPhaseDiagram.pdf", dpi=200)
fig.savefig("GlobalPhaseDiagram.png", dpi=200)
fig.savefig("GlobalPhaseDiagram.jpg", dpi=200)
plt.close("all")
