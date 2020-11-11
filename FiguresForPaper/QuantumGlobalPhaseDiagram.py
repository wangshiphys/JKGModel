"""
Quantum global phase diagram of the triangular lattice J-K-Gamma model.
"""


import matplotlib.pyplot as plt
import numpy as np

from FontSize import *
from QuantumPhaseDiagramDataBase import *

ALPHA_STEP = HEIGHT = 0.005
EXTRA_HEIGHT = HEIGHT / 5

yticks = np.array([0.00, 0.25, 0.50])
xticks = np.array([0.00, 0.50, 1.00, 1.50])
xtick_labels = ["K = 1", "J = 1", "K = -1", "J = -1"]
fig, (ax_north, ax_south) = plt.subplots(
    nrows=1, ncols=2, subplot_kw={"polar": True},
    num="QuantumGlobalPhaseDiagram",
)
# fig, (ax_south, ax_north) = plt.subplots(
#     nrows=1, ncols=2, subplot_kw={"polar": True},
#     num="QuantumGlobalPhaseDiagram",
# )

# Draw the north hemisphere
radius = 0.00
for alpha in np.arange(0.000, 0.505, ALPHA_STEP):
    key = "alpha={0:.3f}".format(alpha)
    edges = np.array(PhaseTransitionPointsWithFixedAlpha[key])
    colors = [PhaseNames2Colors[name] for name in PhaseNames[key]]
    widths = np.append(edges[1:] - edges[:-1], 2 + edges[0] - edges[-1])
    if alpha == 0.00:
        bottom = 0.00
        height = HEIGHT / 2 + EXTRA_HEIGHT
    else:
        bottom = radius - HEIGHT / 2
        height = HEIGHT + EXTRA_HEIGHT

    ax_north.bar(
        x=edges * np.pi, width=widths * np.pi,
        bottom=bottom, height=height, color=colors, align="edge",
    )
    radius += HEIGHT
ax_north.plot(
    [0.26 * np.pi, 0.26 * np.pi], [0.09, 0.44],
    ls="dashed", color="black", lw=2, alpha=0.70,
)

# Draw the south hemisphere
radius = 0.00
for alpha in np.arange(1.000, 0.495, -ALPHA_STEP):
    key = "alpha={0:.3f}".format(alpha)
    edges = np.array(PhaseTransitionPointsWithFixedAlpha[key])
    colors = [PhaseNames2Colors[name] for name in PhaseNames[key]]
    widths = np.append(edges[1:] - edges[:-1], 2 + edges[0] - edges[-1])
    if alpha == 1.00:
        bottom = 0.00
        height = HEIGHT / 2 + EXTRA_HEIGHT
    else:
        bottom = radius - HEIGHT / 2
        height = HEIGHT + EXTRA_HEIGHT

    ax_south.bar(
        x=edges * np.pi, width=widths * np.pi,
        bottom=bottom, height=height, color=colors, align="edge",
    )
    radius += HEIGHT

# Set the properties of ax_north
ax_north.set_yticks(yticks)
ax_north.set_xticks(xticks * np.pi)
ax_north.set_xticklabels(xtick_labels, ha="center", va="center", fontsize=SMALL)
ax_north.set_yticklabels(
    [r"$\alpha=0$", r"$\alpha=\frac{\pi}{4}$", r"$\alpha=\frac{\pi}{2}$"],
    ha="center", va="center", fontsize=SMALL, x=1.15 * np.pi,
)
ax_north.annotate(
    "", xy=(0.40 * np.pi, 0.52), xytext=(0.30 * np.pi, 0.52),
    arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.1"},
)
ax_north.text(
    0.35 * np.pi, 0.52, r"$\beta$",
    ha="center", va="bottom", fontsize=SMALL, rotation=-15,
)
ax_north.spines["polar"].set_visible(False)
ax_north.grid(color="gray", ls="dashed")
ax_north.set_title(r"$\Gamma \geq 0$", fontsize=LARGE)

# Set the properties of ax_south
ax_south.set_yticks(yticks)
ax_south.set_xticks(xticks * np.pi)
ax_south.set_xticklabels(xtick_labels, ha="center", va="center", fontsize=SMALL)
ax_south.set_yticklabels(
    # [r"$\alpha={0:.0f}^\circ$".format(alpha * 180) for alpha in (1 - yticks)],
    [r"$\alpha=\pi$", r"$\alpha=\frac{3}{4}\pi$", r"$\alpha=\frac{\pi}{2}$"],
    ha="center", va="center", fontsize=SMALL, x=1.15 * np.pi,
)
ax_south.annotate(
    "", xy=(0.40 * np.pi, 0.52), xytext=(0.30 * np.pi, 0.52),
    arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.1"},
)
ax_south.text(
    0.35 * np.pi, 0.52, r"$\beta$",
    ha="center", va="bottom", fontsize=SMALL, rotation=-15,
)
ax_south.spines["polar"].set_visible(False)
ax_south.grid(color="gray", ls="dashed")
ax_south.set_title(r"$\Gamma \leq 0$", fontsize=LARGE)

# Annotate the different phases
phase_annotate_north = [
    (0.05 * np.pi, 0.35, "Stripe-C"),
    (0.49 * np.pi, 0.41, r"120$^\circ$ N$\mathsf{\acute e}$el"),
    (0.40 * np.pi, 0.24, "Modulated\n Stripe"),
    (0.81 * np.pi, 0.30, "Stripe-A"),
    (1.50 * np.pi, 0.35, "FM-B"),
    (0.45 * np.pi, 0.05, "QSL"),
]
for annotate in phase_annotate_north:
    ax_north.text(*annotate, ha="center", va="center", fontsize=LARGE)

# ax_north.annotate(
#     "Stripe-C", xy=(0.07 * np.pi, 0.50), xytext=(0.10 * np.pi, 0.60),
#     ha="center", va="bottom", fontsize=LARGE,
#     arrowprops={
#         "arrowstyle": "fancy",
#         "connectionstyle": "angle3",
#         "color": PhaseNames2Colors["StripeC"],
#     },
# )
ax_north.annotate(
    r"Dual N$\mathsf{\acute e}$el",
    xy=(1.85 * np.pi, 0.50), xytext=(1.82 * np.pi, 0.62),
    ha="center", va="bottom", fontsize=LARGE,
    arrowprops={
        "arrowstyle": "fancy",
        "connectionstyle": "angle3",
        "color": PhaseNames2Colors["DualNeel"],
    },
)

ax_south.annotate(
    r"Dual N$\mathsf{\acute e}$el",
    xy=(1.85 * np.pi, 0.50), xytext=(1.82 * np.pi, 0.62),
    ha="center", va="bottom", fontsize=LARGE,
    arrowprops={
        "arrowstyle": "fancy",
        "connectionstyle": "angle3",
        "color": PhaseNames2Colors["DualNeel"],
    },
)

phase_annotate_south = [
    (0.49 * np.pi, 0.43, r"120$^\circ$ N$\mathsf{\acute e}$el"),
    (0.50 * np.pi, 0.20, "Stripe-A"),
    (1.50 * np.pi, 0.35, "FM-C"),
]
for annotate in phase_annotate_south:
    ax_south.text(*annotate, ha="center", va="center", fontsize=LARGE)

ax_north.annotate(
    "Stripe-B", xy=(0.07 * np.pi, 0.50), xytext=(0.10 * np.pi, 0.60),
    ha="center", va="bottom", fontsize=LARGE,
    arrowprops={
        "arrowstyle": "fancy",
        "connectionstyle": "angle3",
        "color": PhaseNames2Colors["StripeC"],
    },
)

ax_south.annotate(
    "Stripe-B", xy=(0.07 * np.pi, 0.50), xytext=(0.10 * np.pi, 0.60),
    ha="center", va="bottom", fontsize=LARGE,
    arrowprops={
        "arrowstyle": "fancy",
        "connectionstyle": "angle3",
        "color": PhaseNames2Colors["StripeC"],
    },
)

ax_north.annotate(
    "FM-A", xy=(1.15 * np.pi, 0.50), xytext=(1.16 * np.pi, 0.60),
    ha="center", va="top", fontsize=LARGE,
    arrowprops={
        "arrowstyle": "fancy",
        "connectionstyle": "angle3",
        "color": PhaseNames2Colors["FMB"],
    },
)

ax_south.annotate(
    "FM-A", xy=(1.15 * np.pi, 0.50), xytext=(1.16 * np.pi, 0.60),
    ha="center", va="top", fontsize=LARGE,
    arrowprops={
        "arrowstyle": "fancy",
        "connectionstyle": "angle3",
        "color": PhaseNames2Colors["FMB"],
    },
)

fig.text(
    0.02, 0.98, "(a)",
    ha="left", va="top", fontsize=LARGE
)
fig.set_size_inches(18, 9)
plt.tight_layout()
plt.show()
fig.savefig("figures/QuantumGlobalPhaseDiagram.pdf", transparent=True)
plt.close("all")
