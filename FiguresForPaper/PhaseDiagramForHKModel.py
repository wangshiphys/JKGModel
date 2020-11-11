import matplotlib.pyplot as plt
import numpy as np

from FontSize import *
from QuantumPhaseDiagramDataBase import *

RADIUS = 1
HEIGHT = 0.1

alpha = 0.50
key = "alpha={0:.3f}".format(alpha)
edges = np.array(PhaseTransitionPointsWithFixedAlpha[key])
colors = [PhaseNames2Colors[name] for name in PhaseNames[key]]
widths = np.append(edges[1:] - edges[:-1], 2 + edges[0] - edges[-1])
centers = np.pi * np.append(
    edges[1:] + edges[:-1], 2 + edges[0] + edges[-1]
) / 2

fig, ax = plt.subplots(subplot_kw={"polar": True})
ax.bar(
    x=edges * np.pi, width=widths * np.pi,
    bottom=RADIUS-HEIGHT/2, height=HEIGHT,
    color=colors, align="edge",
)

# Parameter pairs connected by FST
ax.annotate(
    "", xy=(0.10 * np.pi, RADIUS - HEIGHT / 2),
    xytext=(1.94 * np.pi, RADIUS - HEIGHT / 2),
    arrowprops={
        "linewidth": 1.2, "arrowstyle": "<->",
        "connectionstyle": "arc3, rad=0.0",
    },
)
ax.annotate(
    "", xy=(0.23 * np.pi, RADIUS - HEIGHT / 2),
    xytext=(1.90 * np.pi, RADIUS - HEIGHT / 2),
    arrowprops={
        "linewidth": 1.2, "arrowstyle": "<->",
        "connectionstyle": "arc3, rad=0.0",
    },
)
ax.annotate(
    "", xy=(0.50 * np.pi, RADIUS - HEIGHT / 2),
    xytext=(1.85 * np.pi, RADIUS - HEIGHT / 2),
    arrowprops={
        "linewidth": 1.2, "arrowstyle": "<->",
        "connectionstyle": "arc3, rad=0.0",
    },
)
ax.annotate(
    "", xy=(0.64 * np.pi, RADIUS - HEIGHT / 2),
    xytext=(1.82 * np.pi, RADIUS - HEIGHT / 2),
    arrowprops={
        "linewidth": 1.2, "arrowstyle": "<->",
        "connectionstyle": "arc3, rad=0.0",
    },
)
ax.annotate(
    "", xy=(0.85 * np.pi, RADIUS - HEIGHT / 2),
    xytext=(1.50 * np.pi, RADIUS - HEIGHT / 2),
    arrowprops={
        "linewidth": 1.2, "arrowstyle": "<->",
        "connectionstyle": "arc3, rad=0.0",
    },
)

# Phase boundaries
ax.text(
    edges[0] * np.pi, RADIUS + HEIGHT / 2, r"{0:.2f}$\pi$".format(edges[0]),
    ha="left", va="bottom", fontsize=LARGE
)
ax.text(
    edges[1] * np.pi, RADIUS + HEIGHT / 2, r"{0:.2f}$\pi$".format(edges[1]),
    ha="right", va="bottom", fontsize=LARGE
)
ax.text(
    edges[2] * np.pi, RADIUS + HEIGHT / 2, r"{0:.2f}$\pi$".format(edges[2]),
    ha="right", va="center", fontsize=LARGE
)
# ax.text(
#     (edges[3] - 0.02) * np.pi, RADIUS + HEIGHT / 2 - 0.01,
#     r"{0:.2f}$\pi$".format(edges[3]),
#     ha="left", va="top", fontsize=LARGE
# )
# ax.text(
#     edges[4] * np.pi, RADIUS + HEIGHT / 2 + 0.02,
#     r"{0:.2f}$\pi$".format(edges[4]),
#     ha="left", va="bottom", fontsize=LARGE
# )
ax.annotate(
    r"{0:.2f}$\pi$".format(edges[3]), xy=(edges[3]*np.pi, RADIUS+HEIGHT/2),
    xytext=[(edges[3]-0.1)*np.pi, RADIUS+ HEIGHT/2 + 0.2], fontsize=LARGE,
    arrowprops={
        "linewidth": 1.2, "arrowstyle": "->",
        "connectionstyle": "arc3, rad=0.3"
    },
)
ax.annotate(
    r"{0:.2f}$\pi$".format(edges[4]),
    xy=(edges[4] * np.pi, RADIUS + HEIGHT / 2),
    xytext=[(edges[4] + 0.07) * np.pi, RADIUS + HEIGHT / 2 + 0.06],
    fontsize=LARGE,
    arrowprops={
        "linewidth": 1.2, "arrowstyle": "->",
        "connectionstyle": "arc3, rad=-0.3"
    },
)

# Phase name
ax.text(
    0.07 * np.pi, RADIUS + HEIGHT / 2, "Stripe-B", fontsize=LARGE,
    ha="left", va="bottom", color=PhaseNames2Colors["StripeC"],
)
ax.text(
    0.45 * np.pi, RADIUS + HEIGHT, r"120$^\circ$ N$\mathsf{\acute e}$el",
    ha="center", va="bottom", color=PhaseNames2Colors["Neel"], fontsize=LARGE
)
ax.text(
    0.85 * np.pi, RADIUS + HEIGHT / 2, "Stripe-A",
    ha="right", va="bottom", color=PhaseNames2Colors["StripeA"], fontsize=LARGE
)
ax.text(
    1.50 * np.pi, RADIUS + HEIGHT, "FM-A",
    ha="center", va="top", color=PhaseNames2Colors["FMB"], fontsize=LARGE
)
ax.text(
    1.85 * np.pi, RADIUS + HEIGHT / 2 + 0.01, r"Dual N$\mathsf{\acute e}$el",
    ha="left", va="center", color=PhaseNames2Colors["DualNeel"], fontsize=LARGE
)

xticks = np.array([0.00, 0.50, 1.00, 1.50])
ax.set_xticks(xticks * np.pi)
ax.set_xticklabels([])

ax.set_yticks([])
ax.spines["polar"].set_visible(False)
ax.grid(ls="dashed", lw=1)
fig.set_size_inches(10.2, 9.26)
plt.show()
print(fig.get_size_inches())
fig.savefig("figures/PhaseDiagramForHKModel.pdf", transparent=True)
plt.close("all")
