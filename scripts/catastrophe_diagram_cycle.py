from pathlib import Path
from io import BytesIO
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

import evoscape.landscape_visuals as vis
from evoscape.landscapes import Landscape
from evoscape.modules import Center, Node, UnstableNode
from evoscape.morphogen_regimes import mr_const


plt.style.use("default")
plt.rcParams.update({"figure.dpi": 120})

s_range = (0.1, 0.5)
sig = np.sqrt(s_range[0] * s_range[1])


def update_landscape(p1, p2):
    s_head = sig * np.exp(p2 / 2)
    s_fate = sig * np.exp(-p2 / 2)
    s_left = s_fate * np.exp(-p1 / 2)
    s_right = s_fate * np.exp(p1 / 2)

    modules = [
        Node(x=0.0, y=0.0, a=5.0, s=s_head),
        Node(x=0.0, y=1.0, a=3.0, s=s_left),
        Node(x=1.0, y=0.0, a=3.0, s=s_right),
        Node(x=0.5, y=0.6, a=0.5, s=0.3),
        Center(x=0, y=0, a=5, s=0.3),
        UnstableNode(x=0, y=0, a=10.0, s=s_head * 0.8),
    ]
    return Landscape(modules, A0=0.05, regime=mr_const, n_regimes=1)


def unstable_connection_signature(catastrophe_info):
    return tuple(
        sorted(
            (
                int(connection["connection"][0]),
                int(connection["connection"][1]),
                str(connection["target_type"]),
            )
            for connection in catastrophe_info["unstable_connections"]
        )
    )


def signature_label(signature):
    if not signature:
        return "no unstable connections"
    return "; ".join(f"{src}->{dst} ({target_type})" for src, dst, target_type in signature)


def mix_with_white(color, amount):
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple((1.0 - amount) * rgb + amount * np.ones(3, dtype=float))


order_colors = [
    "tab:purple",
    "tab:blue",
    "tab:green",
    "gold",
]


def render_skeleton_thumb(p1_value, p2_value, xx, yy):
    landscape = update_landscape(p1_value, p2_value)
    phase_result = landscape.find_phase_objects_manifold(
        0.0,
        xx,
        yy,
        dt=0.1,
        n_steps=2000,
        cycle_window=2000,
    )
    fig_skel, ax_skel, _ = vis.plot_phase_skeleton_t(
        landscape,
        xx,
        yy,
        0.0,
        phase_result=phase_result,
        show_cycles=True,
        show_saddle_manifolds=True,
        plot_stable_manifolds=False,
        basin_coloring="node",
        module_order_colors=order_colors,
    )
    fig_skel.set_size_inches(4, 4)
    fig_skel.patch.set_alpha(0.0)
    ax_skel.set_facecolor((1, 1, 1, 0))

    buf = BytesIO()
    fig_skel.savefig(
        buf,
        format="png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig_skel)
    buf.seek(0)
    image = plt.imread(buf)
    buf.close()
    return image


def main():
    start_time = time.perf_counter()
    save_dir = ROOT / "figures" / "catastrophe_diagram_cycle"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_grid_points = 11
    p1 = np.linspace(-1, 2, n_grid_points)
    p2 = np.linspace(-0.5, 1, n_grid_points)
    # q = np.linspace(-1.5, 1.5, 101)
    # xx, yy = np.meshgrid(q + 0.5, q + 0.5, indexing="xy")

    q = np.linspace(-2., 2., 101)
    xx, yy = np.meshgrid(q, q, indexing="xy")

    P1, P2 = np.meshgrid(p1, p2, indexing="xy")

    n_fp_grid = np.zeros((len(p2), len(p1)), dtype=int)
    catastrophe_info_grid = np.empty((len(p2), len(p1)), dtype=object)

    total_points = len(p1) * len(p2)
    point_index = 0

    for row_index, p2_value in enumerate(p2):
        print(f"Computing row {row_index + 1}/{len(p2)}")
        for col_index, p1_value in enumerate(p1):
            landscape = update_landscape(p1_value, p2_value)
            phase_result = landscape.find_phase_objects_manifold(
                0.0,
                xx,
                yy,
                dt=0.2,
                n_steps=10000,
                cycle_window=2000,
            )
            catastrophe_info = landscape.get_catastrophe_info(phase_result)
            catastrophe_info_grid[row_index, col_index] = catastrophe_info
            n_fp_grid[row_index, col_index] = catastrophe_info["n_fp"]

            point_index += 1
            if point_index % 11 == 0 or point_index == total_points:
                print(f"  completed {point_index}/{total_points} sweep points")

    print("Unique fixed-point counts:", np.unique(n_fp_grid).tolist())

    unique_n_fp = sorted(np.unique(n_fp_grid).tolist())

    base_cmap = plt.get_cmap("Accent")
    base_colors = {
        int(n_fp): base_cmap(idx)[:3]
        for idx, n_fp in enumerate(unique_n_fp)
    }

    region_indices = {}
    for row_index in range(len(p2)):
        for col_index in range(len(p1)):
            catastrophe_info = catastrophe_info_grid[row_index, col_index]
            key = (
                int(catastrophe_info["n_fp"]),
                unstable_connection_signature(catastrophe_info),
            )
            region_indices.setdefault(key, []).append((row_index, col_index))

    region_color_map = {}
    region_summary = []
    rgba_grid = np.zeros((len(p2), len(p1), 4), dtype=float)

    for n_fp in unique_n_fp:
        signatures = sorted(
            {signature for (fp_value, signature) in region_indices if fp_value == n_fp},
            key=str,
        )
        shade_amounts = np.array([0.0]) if len(signatures) == 1 else np.linspace(0.0, 0.8, len(signatures))

        for signature_index, (signature, shade_amount) in enumerate(zip(signatures, shade_amounts)):
            key = (int(n_fp), signature)
            region_color_map[key] = mix_with_white(base_colors[int(n_fp)], float(shade_amount))
            region_summary.append(
                {
                    "n_fp": int(n_fp),
                    "signature": signature,
                    "signature_index": int(signature_index),
                    "n_signatures": int(len(signatures)),
                    "n_cells": int(len(region_indices[key])),
                }
            )

    for row_index in range(len(p2)):
        for col_index in range(len(p1)):
            catastrophe_info = catastrophe_info_grid[row_index, col_index]
            key = (
                int(catastrophe_info["n_fp"]),
                unstable_connection_signature(catastrophe_info),
            )
            rgba_grid[row_index, col_index, :3] = region_color_map[key]
            rgba_grid[row_index, col_index, 3] = 1.0

    for entry in sorted(region_summary, key=lambda item: (item["n_fp"], str(item["signature"]))):
        print(
            f"n_fp={entry['n_fp']}, variant {entry['signature_index'] + 1}/{entry['n_signatures']}, "
            f"cells={entry['n_cells']}, unstable_connections={signature_label(entry['signature'])}"
        )

    rng = np.random.default_rng(0)
    sampled_regions = []

    for key, indices in sorted(region_indices.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        n_fp, signature = key
        sample_row, sample_col = indices[rng.integers(len(indices))]
        sample_p1 = p1[sample_col]
        sample_p2 = p2[sample_row]
        sample_index = len(sampled_regions)
        sampled_regions.append(
            {
                "n_fp": int(n_fp),
                "signature": signature,
                "p1": float(sample_p1),
                "p2": float(sample_p2),
                "n_cells": int(len(indices)),
            }
        )

        sample_landscape = update_landscape(sample_p1, sample_p2)
        sample_phase_result = sample_landscape.find_phase_objects_manifold(
            0.0,
            xx,
            yy,
            dt=0.1,
            n_steps=2000,
            cycle_window=2000,
        )
        sample_landscape_fig = vis.visualize_landscape(
            sample_landscape,
            xx,
            yy,
            regime=0,
            color_scheme="fp_types",
        )
        sample_landscape_fig.axes[0].set_title(
            f"Landscape\nn_fp={n_fp}\n{signature_label(signature)}\np1={sample_p1:.2f}, p2={sample_p2:.2f}"
        )
        sample_skeleton_fig, sample_skeleton_ax, _ = vis.plot_phase_skeleton_t(
            sample_landscape,
            xx,
            yy,
            0.0,
            phase_result=sample_phase_result,
            show_cycles=True,
            show_saddle_manifolds=True,
        )
        sample_skeleton_ax.set_title(
            f"Phase skeleton\nn_fp={n_fp}\n{signature_label(signature)}\np1={sample_p1:.2f}, p2={sample_p2:.2f}"
        )
        landscape_path = save_dir / f"sample_{sample_index:02d}_landscape.png"
        skeleton_path = save_dir / f"sample_{sample_index:02d}_skeleton.png"
        sample_landscape_fig.savefig(landscape_path, bbox_inches="tight")
        sample_skeleton_fig.savefig(skeleton_path, bbox_inches="tight")
        plt.close(sample_landscape_fig)
        plt.close(sample_skeleton_fig)
        print(f"Saved {landscape_path}")
        print(f"Saved {skeleton_path}")

    dp1 = p1[1] - p1[0]
    dp2 = p2[1] - p2[0]
    extent = (
        p1[0] - 0.5 * dp1,
        p1[-1] + 0.5 * dp1,
        p2[0] - 0.5 * dp2,
        p2[-1] + 0.5 * dp2,
    )

    fig = plt.figure(figsize=(9, 9))
    ax = plt.gca()
    ax.imshow(rgba_grid, origin="lower", extent=extent, interpolation="nearest", aspect="equal", alpha=0.9)
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")

    thumb_size = 0.15
    thumb_pad = 0.0
    x0, x1, y0, y1 = extent
    thumb_cache = {}

    for sample in sampled_regions[:]:
        key = (sample["p1"], sample["p2"])
        if key not in thumb_cache:
            thumb_cache[key] = render_skeleton_thumb(*key, xx, yy)
        image = thumb_cache[key]

        u = (sample["p1"] - x0) / (x1 - x0)
        v = (sample["p2"] - y0) / (y1 - y0)
        left = np.clip(u - 0.5 * thumb_size, thumb_pad, 1.0 - thumb_size - thumb_pad)
        bottom = np.clip(v - 0.5 * thumb_size, thumb_pad, 1.0 - thumb_size - thumb_pad)

        inset = ax.inset_axes([left, bottom, thumb_size, thumb_size], transform=ax.transAxes, zorder=4)
        inset.imshow(image)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_facecolor((1, 1, 1, 0))
        for spine in inset.spines.values():
            spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=base_colors[int(n_fp)], edgecolor="none", label=f"n_fp = {int(n_fp)}")
        for n_fp in unique_n_fp
    ]
    plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_aspect(2)

    diagram_path = save_dir / "catastrophe_diagram_cycle.png"
    fig.savefig(diagram_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {diagram_path}")
    print(f"elapsed seconds: {time.perf_counter() - start_time:.2f}")

    return catastrophe_info_grid, n_fp_grid, fig


if __name__ == "__main__":
    main()
