#!/usr/bin/env python3

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml


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

    bend_label = "∠(H-O-H) / deg"
    roh_label = "R(O-H)_sym / Å"
    ax.set_xlabel(bend_label)
    ax.set_ylabel(roh_label)

    efield_str = f"F = {np.array2string(efield, precision=4)}"
    suptitle = efield_str
    if title:
        suptitle += f", {title}"
    fig.suptitle(f"{suptitle}")

    ax.legend()

    cut_fig, cut_ax = plt.subplots()
    x_eq, y_eq = org_min_ind
    cut_ax.plot(C2[y_eq], energies0[x_eq], label="Field free")
    cut_ax.plot(C2[y_eq], energies_dressed[x_eq], label="Field dressed")
    cut_ax.set_xlabel(roh_label)
    cut_ax.set_ylabel("$\Delta E$ / au")
    cut_ax.legend()
    cut_fig.suptitle(f"{efield_str}, ∠(H-O-H) = {C1[x_eq][0]:.2f}°")

    bend_fig, bend_ax = plt.subplots()
    c1_coord = C1[:,y_eq]
    bend_ax.plot(c1_coord, energies0[:,y_eq], label="Field free")
    bend_ax.plot(c1_coord, energies_dressed[:,y_eq], label="Field dressed")
    bend_ax.set_xlabel(bend_label)
    bend_ax.set_ylabel("$\Delta E$ / au")
    bend_ax.legend()
    bend_ax.set_xlim(c1_coord.min(), c1_coord.max())
    bend_fig.suptitle(f"{efield_str}, R(O-H)_sym = {C2[:,y_eq][0]:.2f}°")

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
