"""
Plot the ground state energy versus model parameter
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ProjectDataBase.PhaseData import Phase_Transition_Points_Fixed_Alpha
from ProjectDataBase.PhaseData import Phase_Transition_Points_Fixed_Beta


# Useful data for plotting GEs versus alphas or betas
fontsize = 15
linewidth = 5
spinewidth = 3
title_pad = 15
tick_params = {
    "labelsize": 12,
    "which": "both",
    "length": 6,
    "width": spinewidth,
    "direction": "in",
}
color_map = plt.get_cmap("tab10")
colors = color_map(range(color_map.N))

xlabel_template = r"$\{var}(/\pi)$"
second_derivative = r"$-\frac{{d^2E}}{{d\{var}^2}}$"
title_template = r"E and -$\frac{{d^2E}}{{d\{var}^2}}$ vs $\{var}$ " \
                 r"at $\{fixed_which}={fixed_param:.3f}\pi$"


def derivation(xs, ys, nth=1):
    """
    Calculate the nth derivatives of `ys` versus `xs` discretely

    The derivatives are calculated using the following formula:
        dy/dx = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])

    Parameters
    ----------
    xs : 1-D array
        The independent variables
        `xs` is assumed to be sorted in ascending order and there are no
        identical values in `xs`.
    ys : 1-D array
        The dependent variables
        `ys` should be of the same length as `xs`.
    nth : int, optional
        The nth derivatives
        default: 1

    Returns
    -------
    xs : 1-D array
        The independent variables
    ys : 1-D array
        The nth derivatives corresponding to the returned `xs`
    """

    assert isinstance(nth, int) and nth >= 0
    assert isinstance(xs, np.ndarray) and xs.ndim == 1
    assert isinstance(ys, np.ndarray) and ys.shape == xs.shape

    for i in range(nth):
        ys = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        xs = (xs[1:] + xs[:-1]) / 2
    return xs, ys


def GEsVsParams(fixed_param, fixed_which="alpha", step=0.01, numx=3, numy=4,
                save_fig=True, fig_dir=None, show_marker=False):
    if fixed_which == "alpha":
        var = "beta"
        left, right = 0, 2
        color0, color1, color2 = colors[0:3]
        params = betas = np.arange(0, 2, step)
        alphas = np.zeros(betas.shape, dtype=np.float64) + fixed_param
        Phase_Transition_Points = Phase_Transition_Points_Fixed_Alpha
    elif fixed_which == "beta":
        var = "alpha"
        left, right = -1, 1.2
        color0, color1, color2 = colors[3:6]
        params = np.arange(-1, 1, step)
        alphas = np.abs(params)
        betas = np.zeros(params.shape, dtype=np.float64) + fixed_param
        betas[params < 0] = (fixed_param + 1) % 2
        Phase_Transition_Points = Phase_Transition_Points_Fixed_Beta
    else:
        raise ValueError("Invalid fixed_which parameter!")

    gse_data_collection = []
    ge_dir_template = "data/SpinModel/GE/alpha={alpha:.3f}/"
    ge_file_template = "GE_numx={0}_numy={1}_".format(numx, numy)
    ge_file_template += "alpha={alpha:.3f}_beta={beta:.3f}.npy"
    ge_full_name_template = ge_dir_template + ge_file_template
    for param, alpha, beta in zip(params, alphas, betas):
        ge_full_name = ge_full_name_template.format(alpha=alpha, beta=beta)
        try:
            gse = np.load(ge_full_name)[-1]
            gse_data_collection.append([param, gse])
        except FileNotFoundError:
            pass

    gse_data_collection = np.array(gse_data_collection)
    params = gse_data_collection[:, 0]
    params = np.concatenate([params, params + 2])
    Es = gse_data_collection[:, 1]
    Es = np.concatenate([Es, Es])

    # Plot the ground state energies versus params
    fig, ax_Es = plt.subplots()
    ax_d2Es = ax_Es.twinx()

    marker_properties = {"marker": "o", "markersize": 10} if show_marker else {}
    d2params, d2Es = derivation(params, Es, nth=2)
    line_Es, = ax_Es.plot(params, Es, color=color0, lw=linewidth)
    line_d2Es, = ax_d2Es.plot(
        d2params, -d2Es / (np.pi ** 2),
        color=color1, lw=linewidth, **marker_properties
    )

    d2Es_ylabel = d2Es_legend = second_derivative.format(var=var)
    ax_d2Es.set_ylabel(
        d2Es_ylabel, color=color1, fontsize=fontsize, rotation="horizontal"
    )
    ax_d2Es.tick_params("y", colors=color1, **tick_params)

    try:
        y = ax_d2Es.get_ylim()[0]
        key = "{0}={1:.3f}".format(fixed_which, fixed_param)
        for x in  Phase_Transition_Points[key]:
            ax_d2Es.axvline(x, color=color2, ls="dashed")
            ax_d2Es.text(
                x, y, "{0:.3f}".format(x), ha="left", va="bottom",
                rotation=45, fontsize="xx-large", color=color2
            )
    except (KeyError, NameError):
        pass

    title = title_template.format(
        var=var, fixed_which=fixed_which, fixed_param=fixed_param
    )
    ax_Es.set_title(title, pad=title_pad, fontsize=fontsize+3)
    ax_Es.set_ylabel(
        "E", color=color0, fontsize=fontsize, rotation="horizontal"
    )
    ax_Es.tick_params("y", colors=color0, **tick_params)

    # xticks = np.arange(left, right+0.05, 0.1)
    # ax_Es.set_xticks(xticks)
    # ax_Es.set_xticklabels(["{0:.1f}".format(x) for x in xticks], rotation=45)
    # ax_Es.tick_params("x", **tick_params)

    ax_Es.set_xlim(left, right)
    ax_Es.set_xlabel(xlabel_template.format(var=var), fontsize=fontsize)
    ax_Es.legend((line_Es, line_d2Es), ("E", d2Es_legend), loc=0)

    for which, spine in ax_Es.spines.items():
        spine.set_linewidth(spinewidth)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

    if save_fig:
        if fig_dir is None:
            fig_dir = "figure/SpinModel/fixed_{0}/".format(fixed_which)

        fig_name = "GE_numx={0}_numy={1}_step={2:.3f}_{3}={4:.3f}.".format(
            numx, numy, step, fixed_which, fixed_param
        )

        for format in ["pdf", "png"]:
            directory = Path(fig_dir) / format
            directory.mkdir(parents=True, exist_ok=True)
            fig.savefig(directory / (fig_name + format))
    plt.close("all")


if __name__ == "__main__":
    step = 0.01
    alphas = np.arange(0, 1 + step / 2, step)
    betas = np.arange(0, 2, step)
    for alpha in alphas:
        GEsVsParams(
            fixed_param=alpha,
            fixed_which="alpha",
            numx=4,
            numy=6,
            save_fig=False,
            show_marker=True,
        )

    for beta in betas:
        GEsVsParams(
            fixed_param=beta,
            fixed_which="beta",
            numx=4,
            numy=6,
            save_fig=False,
            show_marker=True,
        )