"""
Plot the ground state energy versus model parameter.
"""


import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks

GE_FILE_NAME_TEMP = "GE_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"


def derivation(xs, ys, nth=1):
    """
    Calculate the nth derivatives of `ys` versus `xs` discretely.

    The derivatives are calculated using the following formula:
        dy / dx = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])

    Parameters
    ----------
    xs : 1-D array
        The independent variables.
        `xs` is assumed to be sorted in ascending order and there are no
        identical values in `xs`.
    ys : 1-D array
        The dependent variables.
        `ys` should be of the same length as `xs`.
    nth : int, optional
        The nth derivatives.
        Default: 1.

    Returns
    -------
    xs : 1-D array
        The independent variables.
    ys : 1-D array
        The nth derivatives corresponding to the returned `xs`.
    """

    assert isinstance(nth, int) and nth >= 0
    assert isinstance(xs, np.ndarray) and xs.ndim == 1
    assert isinstance(ys, np.ndarray) and ys.shape == xs.shape

    for i in range(nth):
        ys = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        xs = (xs[1:] + xs[:-1]) / 2
    return xs, ys


def ShowGEs(
        params, gses, fix_which, fixed_param, *,
        line_width=4, show_marker=False, marker_size=6
):
    if fix_which == "alpha":
        title = r"$\alpha={0:.4f}\pi$".format(fixed_param)
        xlabel = r"$\beta/\pi$"
        d2gses_ylabel = r"$-\frac{d^2E}{d\beta^2}$"
    elif fix_which == "beta":
        title = r"$\beta={0:.4f}\pi$".format(fixed_param)
        xlabel = r"$\alpha/\pi$"
        d2gses_ylabel = r"$-\frac{d^2E}{d\alpha^2}$"
    else:
        raise ValueError("Invalid `fix_which`: {0}".format(fix_which))

    d2params, d2gses = derivation(params, gses, nth=2)
    peaks, properties = find_peaks(-d2gses)
    print("Peak positions of second derivatives:")
    print(", ".join("{0:.4f}".format(param) for param in d2params[peaks]))

    fig, ax_gses = plt.subplots()
    ax_d2gses = ax_gses.twinx()
    color_gses = "tab:blue"
    color_d2gses = "tab:red"
    if show_marker:
        marker_props = {"marker": "o", "markersize": marker_size}
    else:
        marker_props = {}

    ax_gses.plot(
        params, gses, lw=line_width, color=color_gses, **marker_props
    )
    ax_d2gses.plot(
        d2params, -d2gses, lw=line_width, color=color_d2gses, **marker_props
    )
    ax_gses.set_xlim(params[0], params[-1])
    ax_gses.set_title(title, fontsize="xx-large")
    ax_gses.set_xlabel(xlabel, fontsize="x-large")
    ax_gses.set_ylabel("$E$", rotation=0, fontsize="x-large", color=color_gses)
    ax_d2gses.set_ylabel(
        d2gses_ylabel, rotation=0, fontsize="x-large", color=color_d2gses
    )
    ax_gses.tick_params("both", colors=color_gses, labelsize="x-large")
    ax_d2gses.tick_params("y", colors=color_d2gses, labelsize="x-large")
    ax_gses.grid(ls="dashed", color=color_gses)
    # ax_d2gses.grid(ls="dashed", color=color_d2gses)
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()
    plt.close("all")


def _standardize(alpha, beta):
    J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
    K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
    G = np.cos(alpha * np.pi)
    new_beta = np.arctan2(J, K) / np.pi % 2 % 2
    new_alpha = np.arctan2(np.sqrt(J ** 2 + K ** 2), G) / np.pi
    return new_alpha, new_beta


def GEsVsAlphas(
        beta, alpha_start=0.0, alpha_end=1.0, *,
        numx=4, numy=6, data_path="data/QuantumSpinModel/GE/",
        line_width=4, show_marker=False, marker_size=6
):
    step = 1E-4
    alphas = np.arange(alpha_start, alpha_end + step, step)

    gses = []
    params = []
    for alpha in alphas:
        new_alpha, new_beta = _standardize(alpha, beta)
        ge_file_full_name = data_path + GE_FILE_NAME_TEMP.format(
            numx, numy, new_alpha, new_beta
        )
        try:
            with np.load(ge_file_full_name) as ld:
                params.append(alpha)
                gses.append(ld["gse"][0])
        except FileNotFoundError:
            pass
    gses = np.array(gses, dtype=np.float64)
    params = np.array(params, dtype=np.float64)
    ShowGEs(
        params, gses,
        fix_which="beta", fixed_param=beta,
        line_width=line_width,
        show_marker=show_marker,
        marker_size=marker_size,
    )


def GEsVsBetas(
        alpha, beta_start=0.0, beta_end=2.0, *,
        numx=4, numy=6, data_path="data/QuantumSpinModel/GE/",
        line_width=4, show_marker=False, marker_size=6
):
    step = 1E-4
    betas = np.arange(beta_start, beta_end + step, step)

    gses = []
    params = []
    for beta in betas:
        new_alpha, new_beta = _standardize(alpha, beta)
        ge_file_full_name = data_path + GE_FILE_NAME_TEMP.format(
            numx, numy, new_alpha, new_beta
        )
        try:
            with np.load(ge_file_full_name) as ld:
                params.append(beta)
                gses.append(ld["gse"][0])
        except FileNotFoundError:
            pass
    gses = np.array(gses, dtype=np.float64)
    params = np.array(params, dtype=np.float64)
    ShowGEs(
        params, gses,
        fix_which="alpha", fixed_param=alpha,
        line_width=line_width,
        show_marker=show_marker,
        marker_size=marker_size,
    )


if __name__ == "__main__":
    GEsVsAlphas(
        beta=0.00,
        alpha_start=-0.1,
        alpha_end=1.1,
        line_width=4,
        # show_marker=True,
        # marker_size=6,
    )
    GEsVsBetas(
        alpha=0.50,
        beta_start=0.0,
        beta_end=2.0,
        line_width=4,
        # show_marker=True,
        # marker_size=6,
    )
