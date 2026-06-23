from pathlib import Path
from io import BytesIO
import multiprocessing as mp
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
plt.rcParams.update({"figure.dpi": 200})

N_PROCESSES = 8

s_range = (0.1, 0.5)
sig = np.sqrt(s_range[0] * s_range[1])


# def update_landscape(p1, p2):
#     s_head = sig * np.exp(p2 / 2)
#     s_fate = sig * np.exp(-p2 / 2)
#     s_left = s_fate * np.exp(-p1 / 2)
#     s_right = s_fate * np.exp(p1 / 2)
#
#     modules = [
#         Node(x=0.0, y=0.0, a=5.0, s=s_head),
#         Node(x=0.0, y=1.0, a=3.0, s=s_left),
#         Node(x=1.0, y=0.0, a=3.0, s=s_right),
#         Node(x=0.5, y=0.6, a=0.5, s=0.3),
#         Center(x=0, y=0, a=5, s=0.3),
#         UnstableNode(x=0, y=0, a=10.0, s=s_head * 0.8),
#     ]
#     return Landscape(modules, A0=0.05, regime=mr_const, n_regimes=1)


def update_landscape(p1, p2):
    amp = 4.
    a_head = amp*np.exp(-p2/2)
    a_fate = amp*np.exp(p2/2)
    a_left = a_fate*np.exp(-p1/2)
    a_right = a_fate*np.exp(p1/2)

    # print(a_head, a_left, a_right)

    s_fate = 0.65 #0.75
    modules = [
        Node(x=-0.0, y=0.8, a=4., s=0.8),

        Node(x=-1.1, y=-1., a=a_left, s=s_fate+0.01),
        Node(x=+1.1, y=-1., a=a_right, s=s_fate),

        UnstableNode(x=0.0, y=0.8, a=a_head*2, s=0.4),
        Center(x=0, y=.8, a=3., s=0.7),
    ]
    return Landscape(modules, A0=0.05, x0=(0., -0.8), regime=mr_const, n_regimes=1)


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


def continuous_region_components(region_id_grid):
    labels = -np.ones(region_id_grid.shape, dtype=int)
    components = []
    for row_index in range(region_id_grid.shape[0]):
        for col_index in range(region_id_grid.shape[1]):
            if labels[row_index, col_index] >= 0:
                continue
            region_id = int(region_id_grid[row_index, col_index])
            component_id = len(components)
            stack = [(row_index, col_index)]
            labels[row_index, col_index] = component_id
            indices = []
            while stack:
                row, col = stack.pop()
                indices.append((row, col))
                for next_row, next_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if (
                        next_row < 0
                        or next_row >= region_id_grid.shape[0]
                        or next_col < 0
                        or next_col >= region_id_grid.shape[1]
                        or labels[next_row, next_col] >= 0
                        or int(region_id_grid[next_row, next_col]) != region_id
                    ):
                        continue
                    labels[next_row, next_col] = component_id
                    stack.append((next_row, next_col))
            components.append({"region_id": region_id, "indices": indices})
    return labels, components


order_colors = [
    "tab:green",
    "tab:blue",
    "tab:purple",
    "gold",
]


def compute_pixel(args):
    row_index, col_index, p1_value, p2_value, q = args
    xx, yy = np.meshgrid(q + 0.5, q + 0.5, indexing="xy")
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
    return row_index, col_index, catastrophe_info, int(catastrophe_info["n_fp"])


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
    save_dir = ROOT / "figures" / "catastrophe_diagram_multi_cycle_amp_21"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"n_processes={N_PROCESSES}")

    n_grid_points = 11
    p1 = np.linspace(-1, 1, n_grid_points)
    p2 = np.linspace(-1, 1, n_grid_points)
    # q = np.linspace(-1.5, 1.5, 101)
    # xx, yy = np.meshgrid(q + 0.5, q + 0.5, indexing="xy")
    q = np.linspace(-2., 2., 101)
    xx, yy = np.meshgrid(q, q, indexing="xy")

    P1, P2 = np.meshgrid(p1, p2, indexing="xy")

    n_fp_grid = np.zeros((len(p2), len(p1)), dtype=int)
    n_c_grid = np.zeros((len(p2), len(p1)), dtype=int)
    catastrophe_info_grid = np.empty((len(p2), len(p1)), dtype=object)

    tasks = []
    for row_index, p2_value in enumerate(p2):
        for col_index, p1_value in enumerate(p1):
            tasks.append((row_index, col_index, p1_value, p2_value, q))

    total_points = len(tasks)
    with mp.Pool(N_PROCESSES) as pool:
        for point_index, result in enumerate(pool.imap_unordered(compute_pixel, tasks), start=1):
            row_index, col_index, catastrophe_info, n_fp = result
            catastrophe_info_grid[row_index, col_index] = catastrophe_info
            n_fp_grid[row_index, col_index] = n_fp
            n_c_grid[row_index, col_index] = int(catastrophe_info["n_cycles"])
            if point_index % 11 == 0 or point_index == total_points:
                print(f"  completed {point_index}/{total_points} sweep points")

    print("Unique fixed-point counts:", np.unique(n_fp_grid).tolist())
    print("Unique cycle counts:", np.unique(n_c_grid).tolist())

    base_keys = sorted(
        {
            (int(n_fp_grid[row_index, col_index]), int(n_c_grid[row_index, col_index]))
            for row_index in range(len(p2))
            for col_index in range(len(p1))
        }
    )

    base_cmap = plt.get_cmap("tab10")
    base_colors = {
        key: base_cmap(idx % base_cmap.N)[:3]
        for idx, key in enumerate(base_keys)
    }

    region_indices = {}
    region_id_grid = np.zeros((len(p2), len(p1)), dtype=int)
    region_key_to_id = {}
    for row_index in range(len(p2)):
        for col_index in range(len(p1)):
            catastrophe_info = catastrophe_info_grid[row_index, col_index]
            key = (
                int(catastrophe_info["n_fp"]),
                int(catastrophe_info["n_cycles"]),
                unstable_connection_signature(catastrophe_info),
            )
            if key not in region_key_to_id:
                region_key_to_id[key] = len(region_key_to_id)
            region_indices.setdefault(key, []).append((row_index, col_index))
            region_id_grid[row_index, col_index] = region_key_to_id[key]

    region_summary = []
    rgba_grid = np.zeros((len(p2), len(p1), 4), dtype=float)

    for n_fp, n_c in base_keys:
        signatures = sorted(
            {
                signature
                for (fp_value, cycle_value, signature) in region_indices
                if fp_value == n_fp and cycle_value == n_c
            },
            key=str,
        )
        shade_amounts = np.array([0.0]) if len(signatures) == 1 else np.linspace(0.0, 0.8, len(signatures))

        for signature_index, (signature, shade_amount) in enumerate(zip(signatures, shade_amounts)):
            key = (int(n_fp), int(n_c), signature)
            # Previous shading mode:
            # shaded_color = mix_with_white(base_colors[(int(n_fp), int(n_c))], float(shade_amount))
            region_summary.append(
                {
                    "n_fp": int(n_fp),
                    "n_c": int(n_c),
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
                int(catastrophe_info["n_cycles"]),
                unstable_connection_signature(catastrophe_info),
            )
            rgba_grid[row_index, col_index, :3] = base_colors[key[:2]]
            rgba_grid[row_index, col_index, 3] = 1.0

    for entry in sorted(region_summary, key=lambda item: (item["n_fp"], item["n_c"], str(item["signature"]))):
        print(
            f"n_fp={entry['n_fp']}, n_c={entry['n_c']}, "
            f"variant {entry['signature_index'] + 1}/{entry['n_signatures']}, "
            f"cells={entry['n_cells']}, unstable_connections={signature_label(entry['signature'])}"
        )

    sampled_regions = []
    region_id_to_key = {region_id: key for key, region_id in region_key_to_id.items()}
    continuous_region_labels, continuous_regions = continuous_region_components(region_id_grid)

    for component in continuous_regions:
        key = region_id_to_key[int(component["region_id"])]
        n_fp, n_c, signature = key
        indices = component["indices"]
        coords = np.array([[p1[col_index], p2[row_index]] for row_index, col_index in indices], dtype=float)
        centroid = coords.mean(axis=0)
        sample_offset = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
        sample_row, sample_col = indices[sample_offset]
        sample_p1 = p1[sample_col]
        sample_p2 = p2[sample_row]
        sample_index = len(sampled_regions)
        sampled_regions.append(
            {
                "n_fp": int(n_fp),
                "n_c": int(n_c),
                "signature": signature,
                "p1": float(sample_p1),
                "p2": float(sample_p2),
                "n_cells": int(len(indices)),
                "component_id": int(continuous_region_labels[sample_row, sample_col]),
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
            f"Landscape\nn_fp={n_fp}, n_c={n_c}\n{signature_label(signature)}\np1={sample_p1:.2f}, p2={sample_p2:.2f}"
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
            f"Phase skeleton\nn_fp={n_fp}, n_c={n_c}\n{signature_label(signature)}\np1={sample_p1:.2f}, p2={sample_p2:.2f}"
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
    ax.imshow(rgba_grid, origin="lower", extent=extent, interpolation="nearest", aspect="equal", alpha=0.2)
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")

    x_edges = np.linspace(extent[0], extent[1], len(p1) + 1)
    y_edges = np.linspace(extent[2], extent[3], len(p2) + 1)
    border_color = "white"
    for row_index in range(len(p2)):
        for col_index in range(len(p1) - 1):
            if region_id_grid[row_index, col_index] == region_id_grid[row_index, col_index + 1]:
                continue
            ax.plot(
                [x_edges[col_index + 1], x_edges[col_index + 1]],
                [y_edges[row_index], y_edges[row_index + 1]],
                color=border_color,
                linewidth=0.8,
                zorder=3,
            )
    for row_index in range(len(p2) - 1):
        for col_index in range(len(p1)):
            if region_id_grid[row_index, col_index] == region_id_grid[row_index + 1, col_index]:
                continue
            ax.plot(
                [x_edges[col_index], x_edges[col_index + 1]],
                [y_edges[row_index + 1], y_edges[row_index + 1]],
                color=border_color,
                linewidth=0.8,
                zorder=3,
            )

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
        Patch(facecolor=base_colors[key], edgecolor="none", label=f"n_fp = {key[0]}, n_c = {key[1]}")
        for key in base_keys
    ]
    # plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_aspect('equal')

    diagram_path = save_dir / "catastrophe_diagram_multi_cycle.png"
    fig.savefig(diagram_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {diagram_path}")
    print(f"elapsed seconds: {time.perf_counter() - start_time:.2f}")

    return catastrophe_info_grid, n_fp_grid, n_c_grid, fig


if __name__ == "__main__":
    main()
