#!/usr/bin/env python3

import argparse
from functools import partial
import sys
import textwrap

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objs as go
import yaml


np.set_printoptions(suppress=True, precision=4)


def get_potential(efield, rot_approx, pot_type,
                  grid, energies0, dipoles, polarizabilities):
    # Dipole contribution
    dipole_contrib = -dipoles.dot(efield)
    if rot_approx:
        dipole_contrib = np.zeros_like(dipole_contrib)
    pol_contrib = -0.5*polarizabilities.dot(efield**2)

    pots = {
        "all": (energies0 + dipole_contrib + pol_contrib).T,
        "dipole": dipole_contrib.T,
        "polarization": pol_contrib.T,
    }
    pot_contours = {
        "all": {"start":-.2, "end": 1, "size": .0125},
        "dipole": {"start":-.2, "end": .2, "size": .00625},
        "polarization": {"start":-.2, "end": .2, "size": .00625}
    }
    contours = pot_contours[pot_type]
    contours.update({"coloring": "heatmap"})

    org_min_ind = np.unravel_index(energies0.argmin(), energies0.shape)

    energies_dressed = energies0 + dipole_contrib + pol_contrib
    cur_min_ind = np.unravel_index(energies_dressed.argmin(), energies_dressed.shape)

    dim = energies0.ndim
    if dim == 1:
        data = [
            {"x": grid,
             "y": energies0,
             "mode": "lines+markers",
             "name": "E0",},

            {"x": grid,
             "y": dipole_contrib,
             "mode": "lines+markers",
             "name": "Dipole",},

            {"x": grid,
             "y": pol_contrib,
             "mode": "lines+markers",
             "name": "Polarization",},

            {"x": grid,
             "y": energies0+dipole_contrib+pol_contrib,
             "mode": "lines+markers",
             "name": "E(F)",
             "line": {
                "width": 6,
             },
             "marker": {
                 "size": 12,
             }
            },
        ]

        layout =  {
            "xaxis": {
                "title": "Scan coordinate",
            },
            "yaxis": {
                "title": "Energy / a.u.",
            }
        }
    elif dim == 2:
        c1, c2 = grid
        data = [
            go.Contour(
                # z=energies0.T+dipole_contrib.T+pol_contrib.T,
                z=pots[pot_type],
                x=c1[:,0],
                y=c2[0],
                colorscale="Viridis",
                contours=contours,
                colorbar={
                    "len": 0.75,
                    "title": "ΔE / au",
                },
            ),
            go.Scatter(
                x=(c1[org_min_ind], ),
                y=(c2[org_min_ind], ),
                name = "Field free minimum",
                marker = dict(
                    size=10,
                    color="rgb(0,0,0)",
                )
            ),
            go.Scatter(
                x=(c1[cur_min_ind], ),
                y=(c2[cur_min_ind], ),
                name = "Field dressed minimum",
                marker = dict(
                    size=10,
                    # color="rgb(122,122,122)",
                    color="rgb(255,255,255)",
                )
            ),
        ]
        layout = {
            "xaxis": {
                "title": "Coord 1",
            },
            "yaxis": {
                "title": "Coord 2",
            }
        }
    else:
        print("This should never happen :)")

    add_layout = {
            "hovermode": "closest",
            "width": 1200,
            "height": 800,
    }
    layout.update(add_layout)

    return data, layout


E_MIN = -0.074
E_MAX =  -1*E_MIN
E_STEP = 0.001


def get_E_slider(id_):
    return dcc.Slider(
                id=id_,
                min=E_MIN,
                max=E_MAX,
                step=E_STEP,
                value=0,
    )


def get_dash_app(grid, energies0, dpms, pols):
    xs = np.arange(energies0.shape[0])

    pot_kwargs = {
        "grid": grid,
        "energies0": energies0,
        "dipoles": dpms,
        "polarizabilities": pols,
    }
    pot_func = partial(get_potential, **pot_kwargs)

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id="pot-type",
                    options=[
                        {"label": "Pot. energy + dipole + polarization",
                         "value": "all"},
                        {"label": "Only dipole contribution",
                         "value": "dipole"},
                        {"label": "Only polarization contribution",
                         "value": "polarization"},
                    ], value="all",
                ),
            ], className="two columns"),
        ], className="row"),

        # Second row
        html.Div([
            # Potential plot
            html.Div([
                html.H2("Potential energy"),
                dcc.Graph(id="potentials")
            ], className="eight columns"),
            # Sliders
            html.Div([
                html.H2("E-Field strength"),
                get_E_slider("Ex-slider"),
                get_E_slider("Ey-slider"),
                get_E_slider("Ez-slider"),
                html.Div(id="Efield-au-output", style={"fontSize": 20}),
                html.Div(id="Efield-va-output", style={"fontSize": 20}),
                dcc.Checklist(id="rot-approx",
                    options=[
                        {"label": "Rotating wave approximation (no dipole  contribution)",
                         "value": "rot"},
                    ],
                    values=[],
                ),
                html.H2(id="point-h2"),
                html.Div([
                    dcc.Markdown(id="point-info"),
                ], style={"fontSize": 20}),
            ], className="four columns"
            ),
        ], className="row"),
    ])

    def inds_from_click(click_data):
        point = click_data["points"][0]
        if "pointNumber" in point:
            inds = (point["pointNumber"], )
        elif "curveNumber" in point:
            x = point["x"]
            y = point["y"]
            inds = (x, y)
        return inds

    @app.callback(
        Output("point-h2", "children"),
        [Input("potentials", "clickData"),
        ]
    )
    def set_point_heading(click_data):
        if click_data is None:
            return ""
        point_number = inds_from_click(click_data)
        return f"Point {point_number}"

    @app.callback(
        Output("point-info", "children"),
        [Input("potentials", "clickData"),
        ]
    )
    def show_point_info(click_data):
        if click_data is None:
            return ""
        inds = inds_from_click(click_data)
        if len(inds) == 1:
            x = inds
            E_0 = energies0[x]
            dpm = dpms[x]
            alpha = pols[x]
        elif len(inds) == 2:
            # x, y, = inds
            # E_0 = energies0[x][y]
            # dpm = dpms[x][y]
            # alpha = pols[x][y]
            return ""
        dpm = np.array2string(dpm, precision=4)
        alpha = np.array2string(alpha, precision=2)
        markdown = f"""
        E_0: {E_0:.4f} a.u.

        μ(x, y, z):  {dpm} a.u.

        α(xx, yy, zz): {alpha} au
        """
        return textwrap.dedent(markdown)

    @app.callback(
        Output("potentials", "figure"),
        [Input(component_id="Ex-slider", component_property="value"),
         Input(component_id="Ey-slider", component_property="value"),
         Input(component_id="Ez-slider", component_property="value"),
         Input(component_id="rot-approx", component_property="values"),
         Input(component_id="pot-type", component_property="value"),
        ]
    )
    def set_potential(Ex, Ey, Ez, use_rot_approx, pot_type):
        rot_approx = "rot" in use_rot_approx
        efield = np.array((Ex, Ey, Ez))
        data, layout = pot_func(efield, rot_approx, pot_type)

        figure={
            "data": data,
            "layout": layout,
        }
        return figure

    @app.callback(
        Output("Efield-au-output", "children"),
        [Input(component_id="Ex-slider", component_property="value"),
         Input(component_id="Ey-slider", component_property="value"),
         Input(component_id="Ez-slider", component_property="value"),
        ]
    )
    def set_efield_au_output(Ex, Ey, Ez):
        efield_str = f"F / au = ({Ex:.4f}, {Ey:.4f}, {Ez:.4f})"
        return efield_str

    @app.callback(
        Output("Efield-va-output", "children"),
        [Input(component_id="Ex-slider", component_property="value"),
         Input(component_id="Ey-slider", component_property="value"),
         Input(component_id="Ez-slider", component_property="value"),
        ]
    )
    def set_efield_va_output(Ex, Ey, Ez):
        efs = np.array((Ex, Ey, Ez)) * 51.422065
        Ex, Ey, Ez = efs
        efield_str = f"F / V/Å = ({Ex:.4f}, {Ey:.4f}, {Ez:.4f})"
        return efield_str

    return app


def load_data(conf):
    energies_fn = conf["energies"]
    dpms_fn = conf["dipoles"]
    pols_fn = conf["stat_pols"]
    energies = np.load(energies_fn).flatten()
    dpms = np.load(dpms_fn).reshape(energies.size, 3)
    pols = np.load(pols_fn)
    pols = np.diagonal(pols, axis1=2, axis2=3).reshape(energies.size, 3)

    xs = np.arange(energies.size)
    grid = xs

    # import pdb; pdb.set_trace()
    return grid, energies, dpms, pols


def load_data_2d(conf):
    grid_fn = conf["grid"]
    energies_fn = conf["energies"]
    dpms_fn = conf["dipoles"]
    pols_fn = conf["stat_pols"]

    grid = np.load(grid_fn)
    energies = np.load(energies_fn)
    dpms = np.load(dpms_fn)
    pols = np.load(pols_fn)

    energies -= energies.min()

    return grid, energies, dpms, pols


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("yaml")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--grid", action="store_true")

    return parser.parse_args()


def run():
    args = parse_args(sys.argv[1:])

    with open(args.yaml) as handle:
        conf = yaml.load(handle)

    load_func = load_data
    if args.grid:
        load_func = load_data_2d
    grid, energies, dpms, pols = load_func(conf)

    app = get_dash_app(grid, energies, dpms, pols)
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True
    app.run_server(debug=True, reloader_type="stat", port=args.port)


if __name__ == "__main__":
    run()
