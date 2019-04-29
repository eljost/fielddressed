#!/usr/bin/env python3

import argparse
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import yaml


mpl.rcParams['lines.linewidth'] = 2.0


def get_pot_fig(grid, energies0, dipoles, polarizabilities, efield,
                no_dipo=False, title=""):
    """Calculates field dressed potential.

    E(F) = E_0 - μF - 1/2 αF²
    """

    C1, C2 = grid

    # Dipole contribution
    dipole_contrib = -dipoles.dot(efield)
    if no_dipo:
        dipole_contrib = np.zeros_like(dipole_contrib)
    # Polarization contribution
    pol_contrib = -0.5*polarizabilities.dot(efield**2)
    energies0 -= energies0.min()

    org_min_ind = np.unravel_index(energies0.argmin(), energies0.shape)
    energies_dressed = energies0 + dipole_contrib + pol_contrib
    dressed_min_ind = np.unravel_index(energies_dressed.argmin(),
                                       energies_dressed.shape)

    fig, ax = plt.subplots()
    levels = np.linspace(min(0, energies_dressed.min()),
                         energies_dressed.max()*.5, 35)
    cf = ax.contourf(C1, C2, energies_dressed, levels=levels)
    colorbar = fig.colorbar(cf)
    colorbar.set_label("$\Delta E$ / au")

    org_min = C1[org_min_ind], C2[org_min_ind]
    dressed_min = C1[dressed_min_ind], C2[dressed_min_ind]
    ax.scatter(*org_min, label="Org. min.")
    ax.scatter(*dressed_min, label="Dressed. min.")

    bend_label = "$\sphericalangle$(HOH) / deg"
    roh_label = "r(OH)$_{/rm sym}$ / Å"
    ax.set_xlabel(bend_label)
    ax.set_ylabel(roh_label)

    efield_fn_str = "F_" + "_".join([str(comp) for comp in efield])
    efield_str = f"F = {np.array2string(efield, precision=4)}"
    suptitle = efield_str
    if title:
        suptitle += f", {title}"
    fig.suptitle(f"{suptitle}")
    ax.legend()
    fig.savefig(f"{efield_fn_str}_pes.pdf")

    cut_fig, cut_ax = plt.subplots()
    x_eq, y_eq = org_min_ind
    xs = C2[y_eq]
    eq_ens = energies0[x_eq]
    eq_min_ind = eq_ens.argmin()
    dressed_ens = energies_dressed[x_eq]
    dressed_min_ind = dressed_ens.argmin()
    cut_ax.plot(xs, eq_ens, label="Field free")
    cut_ax.plot(xs, dressed_ens, label="Field dressed")
    eq_min_x = xs[eq_min_ind]
    eq_min_y = eq_ens[eq_min_ind]
    cut_ax.scatter(eq_min_x, eq_min_y, label="Field free min.")
    dressed_min_x = xs[dressed_min_ind]
    dressed_min_y = dressed_ens[dressed_min_ind]
    cut_ax.scatter(dressed_min_x, dressed_min_y, label="Dressed min.")

    def add_ax_lines(ax, point, color):
        x, y = point

        x_min, x_max = ax.get_ylim()
        y_min, y_max = ax.get_ylim()
        x_line = lines.Line2D(xdata=[x, x], ydata=[y_min, y], ls="--", c=color)
        y_line = lines.Line2D(xdata=[x_min, x], ydata=[y, y], ls="--", c=color)
        ax.add_line(x_line)
        ax.add_line(y_line)
    add_ax_lines(cut_ax, (eq_min_x, eq_min_y), "C0")
    add_ax_lines(cut_ax, (dressed_min_x, dressed_min_y), "C1")

    cut_ax.set_xlabel(roh_label)
    cut_ax.set_ylabel("$\Delta E$ / au")
    cut_ax.legend()
    cut_fig.suptitle(f"{efield_str}, {title}, $\sphericalangle$(HOH) = {C1[x_eq][0]:.2f}°")
    cut_fig.savefig(f"{efield_fn_str}_bond.pdf")

    bend_fig, bend_ax = plt.subplots()
    xs = C1[:,y_eq]
    eq_ens = energies0[:,y_eq]
    dressed_ens = energies_dressed[:,y_eq]
    bend_ax.plot(xs, eq_ens, label="Field free")
    bend_ax.plot(xs, dressed_ens, label="Field dressed")

    eq_min_ind = eq_ens.argmin()
    dressed_min_ind = dressed_ens.argmin()
    eq_min_x = xs[eq_min_ind]
    eq_min_y = eq_ens[eq_min_ind]
    bend_ax.scatter(eq_min_x, eq_min_y, label="Field free min.")
    dressed_min_x = xs[dressed_min_ind]
    dressed_min_y = dressed_ens[dressed_min_ind]
    bend_ax.scatter(dressed_min_x, dressed_min_y, label="Dressed min.")
    add_ax_lines(bend_ax, (eq_min_x, eq_min_y), "C0")
    add_ax_lines(bend_ax, (dressed_min_x, dressed_min_y), "C1")

    bend_ax.set_xlabel(bend_label)
    bend_ax.set_ylabel("$\Delta E$ / au")
    bend_ax.legend()
    bend_ax.set_xlim(xs.min(), xs.max())
    bend_fig.suptitle(f"{efield_str}, {title}, r(OH)$_{{\\rm sym}}$ = {C2[:,y_eq][0]:.2f} Å")
    bend_fig.savefig(f"{efield_fn_str}_angle.pdf")

    return fig, cut_fig


def load_data(conf):
    grid_fn = conf["grid"]
    energies_fn = conf["energies"]
    dpms_fn = conf["dipoles"]
    pols_fn = conf["stat_pols"]
    title = conf["title"]

    grid = np.load(grid_fn)
    energies = np.load(energies_fn)
    dpms = np.load(dpms_fn)
    pols = np.load(pols_fn)

    return grid, energies, dpms, pols, title


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("yaml")
    parser.add_argument("--efield", nargs=3, type=float, default=(0, 0, 0),
        help="Strength of the applied field in a.u."
    )
    parser.add_argument("--no-dipo", action="store_true",
        help="Disable dipole contribution.",
    )

    return parser.parse_args()


def run():
    args = parse_args(sys.argv[1:])
    with open(args.yaml) as handle:
        conf = yaml.load(handle)

    efield = np.array(args.efield)
    no_dipo = args.no_dipo
    grid, energies, dpms, pols, title = load_data(conf)

    fig, cut_fig = get_pot_fig(grid, energies, dpms, pols, efield, no_dipo, title)
    plt.show()


if __name__ == "__main__":
    run()
