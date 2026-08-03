from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from evoscape.landscapes import Landscape
from evoscape.landscape_visuals import visualize_landscape
from evoscape.modules import Node
from evoscape.morphogen_regimes import mr_const

USE_JAX = False


def make_landscape(s_left, s_right):
    modules = [
        Node(x=-2.0, y=0.0, a=1.8, s=s_left),
        Node(x=0.0, y=0.0, a=2.0, s=0.8),
        Node(x=2.0, y=0.0, a=1.8, s=s_right),
    ]
    return Landscape(modules, A0=0.05, regime=mr_const, n_regimes=1)


def main():
    plt.style.use("default")
    plt.rcParams.update({"figure.dpi": 120})
    print(f"use_jax={USE_JAX}")

    p1 = np.linspace(0.0, 1.0, 11)
    p2 = np.linspace(0.0, 1.0, 11)
    P1, P2 = np.meshgrid(p1, p2, indexing="xy")

    s_left_range = (0.3, 1.3)
    s_right_range = (0.3, 1.3)

    n_fp_grid = np.zeros((len(p2), len(p1)), dtype=int)
    catastrophe_info_grid = np.empty((len(p2), len(p1)), dtype=object)

    q = np.linspace(-3.0, 3.0, 31)
    xx, yy = np.meshgrid(q, q, indexing="xy")
    q_vis = np.linspace(-3.0, 3.0, 101)
    xx_vis, yy_vis = np.meshgrid(q_vis, q_vis, indexing="xy")

    total_points = len(p1) * len(p2)
    point_index = 0
    for row_index, p2_value in enumerate(p2):
        print(f"Computing row {row_index + 1}/{len(p2)} for p2={p2_value:.2f}")
        for col_index, p1_value in enumerate(p1):
            s_left = s_left_range[0] + p1_value * (s_left_range[1] - s_left_range[0])
            s_right = s_right_range[0] + p2_value * (s_right_range[1] - s_right_range[0])

            landscape = make_landscape(s_left, s_right)
            phase_result = landscape.find_phase_objects_manifold(
                0.0,
                xx,
                yy,
                dt=0.06,
                n_steps=250,
                cycle_window=96,
                use_jax=USE_JAX,
            )
            catastrophe_info = landscape.get_catastrophe_info(phase_result)
            catastrophe_info_grid[row_index, col_index] = catastrophe_info
            n_fp_grid[row_index, col_index] = catastrophe_info["n_fp"]

            point_index += 1
            if point_index % 11 == 0 or point_index == total_points:
                print(f"  completed {point_index}/{total_points} sweep points")

    print(f"catastrophe_info_grid shape: {catastrophe_info_grid.shape}")
    print(f"n_fp_grid shape: {n_fp_grid.shape}")
    print(f"Unique fixed-point counts: {np.unique(n_fp_grid).tolist()}")

    rng = np.random.default_rng(0)
    sampled_points = []
    save_dir = ROOT / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    for n_fp in np.unique(n_fp_grid):
        matching_indices = np.argwhere(n_fp_grid == n_fp)
        sample_row, sample_col = matching_indices[rng.integers(len(matching_indices))]
        sample_p1 = p1[sample_col]
        sample_p2 = p2[sample_row]
        sample_s_left = s_left_range[0] + sample_p1 * (s_left_range[1] - s_left_range[0])
        sample_s_right = s_right_range[0] + sample_p2 * (s_right_range[1] - s_right_range[0])
        sampled_points.append((int(n_fp), float(sample_p1), float(sample_p2)))

        sample_landscape = make_landscape(sample_s_left, sample_s_right)
        sample_fig = visualize_landscape(
            sample_landscape,
            xx_vis,
            yy_vis,
            regime=0,
            color_scheme="order",
        )
        sample_fig.axes[0].set_title(
            f"Landscape sample for n_fp={n_fp}\np1={sample_p1:.2f}, p2={sample_p2:.2f}"
        )
        sample_path = save_dir / f"catastrophe_diagram_region_{int(n_fp)}fp.png"
        sample_fig.savefig(sample_path, bbox_inches="tight")
        print(
            "Saved landscape sample for "
            f"n_fp={n_fp} at p1={sample_p1:.2f}, p2={sample_p2:.2f} to {sample_path}"
        )

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    image = ax.pcolormesh(
        P1,
        P2,
        n_fp_grid,
        cmap="viridis",
        shading="auto",
    )
    ax.set_aspect("auto")
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_title("Catastrophe diagram: fixed points vs left/right module s")
    for n_fp, sample_p1, sample_p2 in sampled_points:
        ax.scatter(
            sample_p1,
            sample_p2,
            s=55,
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
        )
        ax.text(
            sample_p1 + 0.015,
            sample_p2 + 0.015,
            str(n_fp),
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 1.5, "edgecolor": "none"},
            zorder=4,
        )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("number of fixed points")

    fig.tight_layout()

    save_path = save_dir / "catastrophe_diagram_example.png"
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved figure to {save_path}")

    if "agg" in plt.get_backend().lower():
        plt.close("all")
    else:
        plt.show()

    return catastrophe_info_grid, n_fp_grid, fig


if __name__ == "__main__":
    main()
