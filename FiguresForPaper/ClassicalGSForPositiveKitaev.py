"""
Classical ground state spin configurations for pure positive Kitaev model.
"""


import matplotlib.pyplot as plt
import numpy as np


POINTS = np.dot(
    [[i, j] for i in range(6) for j in range(6)],
    np.array([[0.5, np.sqrt(3) / 2], [1.0, 0.0]]),
)

XCHAINS = [
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, 31, 32, 33, 34, 35],
]

YCHAINS = [
    [0],
    [1,  6],
    [2,  7, 12],
    [3,  8, 13, 18],
    [4,  9, 14, 19, 24],
    [5, 10, 15, 20, 25, 30],
    [11, 16, 21, 26, 31],
    [17, 22, 27, 32],
    [23, 28, 33],
    [29, 34],
    [35],
]

ZCHAINS = [
    [0,  6, 12, 18, 24, 30],
    [1,  7, 13, 19, 25, 31],
    [2,  8, 14, 20, 26, 32],
    [3,  9, 15, 21, 27, 33],
    [4, 10, 16, 22, 28, 34],
    [5, 11, 17, 23, 29, 35],
]


def StripeGenerator(config="StripeX"):
    if config == "StripeX":
        chains = XCHAINS
        spin_up = [1.0, 0.0, 0.0]
        spin_down = [-1.0, 0.0, 0.0]
    elif config == "StripeY":
        chains = YCHAINS
        spin_up = [-0.5, np.sqrt(3) / 2, 0.0]
        spin_down = [0.5, -np.sqrt(3) / 2, 0.0]
    elif config == "StripeZ":
        chains = ZCHAINS
        spin_up = [0.5, np.sqrt(3) / 2, 0.0]
        spin_down = [-0.5, -np.sqrt(3) / 2, 0.0]
    else:
        raise ValueError("Invalid `config`: {0}".format(config))

    spin_vectors = np.empty((36, 3), dtype=np.float64)
    for index, chain in enumerate(chains):
        if index % 2 == 0:
            spin_vectors[chain] = spin_up
        else:
            spin_vectors[chain] = spin_down
    return spin_vectors


def NematicGenerator(config="NematicX"):
    if config == "NematicX":
        chains = XCHAINS
        spin_up = [1.0, 0.0, 0.0]
        spin_down = [-1.0, 0.0, 0.0]
    elif config == "NematicY":
        chains = YCHAINS
        spin_up = [-0.5, np.sqrt(3) / 2, 0.0]
        spin_down = [0.5, -np.sqrt(3) / 2, 0.0]
    elif config == "NematicZ":
        chains = ZCHAINS
        spin_up = [0.5, np.sqrt(3) / 2, 0.0]
        spin_down = [-0.5, -np.sqrt(3) / 2, 0.0]
    else:
        raise ValueError("Invalid `config`: {0}".format(config))

    spin_vectors = np.empty((36, 3), dtype=np.float64)
    for chain in chains:
        indices0 = chain[0::2]
        indices1 = chain[1::2]
        if np.random.random() < 0.5:
            spin_vectors[indices0] = spin_up
            spin_vectors[indices1] = spin_down
        else:
            spin_vectors[indices1] = spin_up
            spin_vectors[indices0] = spin_down
    return spin_vectors


vectors0 = StripeGenerator("StripeX")
vectors1 = StripeGenerator("StripeY")
vectors2 = StripeGenerator("StripeZ")
vectors3 = NematicGenerator("NematicX")
vectors4 = NematicGenerator("NematicY")
vectors5 = NematicGenerator("NematicZ")

sub_fig_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
tmp = [vectors0, vectors1, vectors2, vectors3, vectors4, vectors5]

fig, axes = plt.subplots(nrows=2, ncols=3, sharex="all", sharey="all")
for index in range(6):
    spin_vectors = tmp[index]
    ax = axes[divmod(index, 3)]
    sub_fig_tag = sub_fig_tags[index]

    ax.plot(
        POINTS[:, 0], POINTS[:, 1],
        ls="", marker="o", ms=10, color="black", zorder=0,
    )

    colors = 0.5 * spin_vectors + 0.5
    ax.quiver(
        POINTS[:, 0], POINTS[:, 1], spin_vectors[:, 0], spin_vectors[:, 1],
        color=colors, units="xy", scale_units="xy", scale=1.45,
        width=0.06, pivot="mid", clip_on=False, zorder=1,
    )
    ax.text(
        1.0, 2.5 * np.sqrt(3), sub_fig_tag,
        ha="center", va="center", fontsize="xx-large",
    )
    ax.set_axis_off()
    ax.set_aspect("equal")

fig.subplots_adjust(
    top=0.88, bottom=0.11, left=0.11, right=0.90, hspace=0.00, wspace=0.00,
)
plt.get_current_fig_manager().window.showMaximized()
plt.show()
fig.savefig("figures/SpinConfigForPositiveKitaev.pdf", dpi=200)
fig.savefig("figures/SpinConfigForPositiveKitaev.png", dpi=200)
plt.close("all")
