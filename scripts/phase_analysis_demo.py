# from pathlib import Path
# import os
# import sys

# ROOT = Path(__file__).resolve().parents[1]
# os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
# Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
# sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

import evoscape.landscape_visuals as vis
from evoscape.landscapes import Landscape
from evoscape.modules import Center, Node, UnstableNode
from evoscape.morphogen_regimes import mr_const


def build_grid(limit, n_points):
    q = np.linspace(-limit, limit, n_points)
    return np.meshgrid(q, q, indexing="xy")


def print_summary(title, fixed_points, phase_result, basin_grid, saddle_manifolds=None):
    print(f"\n{title}")
    print(f"  phase method: {phase_result.get('method', 'unknown')}")
    print(f"  fixed points found: {len(fixed_points['points'])}")
    for idx, point in enumerate(fixed_points["points"]):
        eigvals = np.round(fixed_points["eigenvalues"][idx], 4)
        print(
            f"    fp {idx}: point={np.round(point, 4)} "
            f"stability={fixed_points['stability'][idx]} eig={eigvals}"
        )

    if saddle_manifolds is not None:
        print(f"  saddle manifolds computed: {len(saddle_manifolds['saddles'])}")

    print(f"  basin labels: {sorted(np.unique(basin_grid['basin_labels']).tolist())}")
    for attractor in phase_result["attractors"]:
        if attractor["type"] == "fixed_point":
            print(
                f"    basin {attractor['id']}: fixed point "
                f"{np.round(attractor['point'], 4)}"
            )
        else:
            print(
                f"    basin {attractor['id']}: cycle center="
                f"{np.round(attractor['center'], 4)} period~{attractor['period']:.3f}"
            )


def fixed_point_example():
    modules = [
        Node(x=-1.5, y=0.0, a=1.5, s=1.0),
        Node(x=1.5, y=0.0, a=2.0, s=0.8),
        Node(x=0.0, y=2.0, a=2.3, s=0.5),
        # Node(x=0.0, y=-2.0, a=2.3, s=0.5),

    ]
    landscape = Landscape(modules, A0=0.05, regime=mr_const, n_regimes=1)
    xx, yy = build_grid(3.0, 101)
    fixed_points = landscape.find_fixed_points(0.0, (-3.0, 3.0), (-3.0, 3.0), n_grid=23)
    phase_result = landscape.find_phase_objects_manifold(
        0.0,
        xx,
        yy,
        fixed_points=fixed_points,
        dt=0.06,
        n_steps=600,
        cycle_window=96,
    )
    basin_grid = landscape.find_attractor_basins_manifold(phase_result=phase_result)
    saddle_manifolds = phase_result["saddle_manifolds"]
    print_summary(
        "Example 1: multi-well fixed-point basins",
        fixed_points,
        phase_result,
        basin_grid,
        saddle_manifolds=saddle_manifolds,
    )
    fig, ax, _ = vis.plot_attractor_basins_t(
        landscape,
        xx,
        yy,
        0.0,
        phase_result=phase_result,
        basin_grid=basin_grid,
        show_saddle_manifolds=True,
    )
    # ax.set_title("Multi-well fixed-point basins with saddle manifolds")
    skeleton_fig, skeleton_ax, _ = vis.plot_phase_skeleton_t(
        landscape,
        xx,
        yy,
        0.0,
        phase_result=phase_result,
        show_saddle_manifolds=True,
    )
    # skeleton_ax.set_title("Multi-well phase skeleton")
    # potential_fig, potential_ax, _ = plot_phase_skeleton_on_potential_t(
    #     landscape,
    #     xx,
    #     yy,
    #     0.0,
    #     basin_result=basin_result,
    #     fixed_points=fixed_points,
    #     saddle_manifolds=saddle_manifolds,
    #     show_saddle_manifolds=True,
    #     color_surface_by_basin=False,
    #     elev=34,
    #     azim=-62,
    # )
    # potential_ax.set_title("Multi-well skeleton on basin-colored potential")
    return fig


def cycle_example():
    modules = [
        Node(0.0, 0.0, (2.,), (1.8,), tau=1.0),
        UnstableNode(0.0, 0.0, (3.,), (1.2,), tau=1.0),
        Center(0.0, 0.0, (1.0,), (2.,), tau=1.0),
        Node(2.5, 2.5, (4.,), (0.8,), tau=1.0),
    ]
    landscape = Landscape(modules, A0=0.01, regime=mr_const, n_regimes=1)
    xx, yy = build_grid(3.0, 151)
    t = 0.0

    fixed_points = landscape.find_fixed_points(t, (-3.0, 3.0), (-3.0, 3.0), n_grid=25)
    saddle_manifolds = landscape.find_saddle_manifolds(
        t,
        fixed_points=fixed_points,
        x_range=(-3.0, 3.0),
        y_range=(-3.0, 3.0),
        step_size=0.025,
        n_steps=800,
        termination_tol=2e-2,
        velocity_tol=2e-3,
    )
    phase_result = landscape.find_phase_objects_manifold(
        t,
        xx,
        yy,
        fixed_points=fixed_points,
        saddle_manifolds=saddle_manifolds,
        dt=0.04,
        n_steps=800,
        fp_tol=2e-2,
        vel_tol=2e-3,
        cycle_window=256,
    )
    basin_grid = landscape.find_attractor_basins_manifold(phase_result=phase_result)

    print_summary(
        "Example 2: static cycle basin with saddle manifolds",
        fixed_points,
        phase_result,
        basin_grid,
        saddle_manifolds=saddle_manifolds,
    )
    fig, ax, _ = vis.plot_attractor_basins_t(
        landscape,
        xx,
        yy,
        t,
        phase_result=phase_result,
        basin_grid=basin_grid,
        show_saddle_manifolds=True,
    )
    # ax.set_title("Static cycle basin with saddle manifolds")
    skeleton_fig, skeleton_ax, _ = vis.plot_phase_skeleton_t(
        landscape,
        xx,
        yy,
        t,
        phase_result=phase_result,
        show_saddle_manifolds=True,
    )
    # skeleton_ax.set_title("Static cycle phase skeleton")
    # potential_fig, potential_ax, _ = plot_phase_skeleton_on_potential_t(
    #     landscape,
    #     xx,
    #     yy,
    #     t,
    #     basin_result=basin_result,
    #     fixed_points=fixed_points,
    #     saddle_manifolds=saddle_manifolds,
    #     show_saddle_manifolds=True,
    #     color_surface_by_basin=False,
    #     elev=36,
    #     azim=-64,
    # )
    # potential_ax.set_title("Static cycle skeleton on basin-colored potential")
    return fig


def main():
    vis.update_params()
    plt.style.use("default")
    plt.rcParams.update({"figure.dpi": 120})

    fixed_point_example()
    cycle_example()

    plt.show()


    if "agg" in plt.get_backend().lower():
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
