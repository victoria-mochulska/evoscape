import matplotlib.pyplot as plt
import numpy as np
import colorsys
from copy import copy

from matplotlib.colors import ListedColormap, BoundaryNorm, CenteredNorm
# import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import imageio.v2 as imageio

from .morphogen_regimes import mr_current_regime
from .landscape_visuals_config import (
    figure_dpi,
    streamline_color,
    basin_streamline_color,
    surface_contour_color,
    neutral_color,
    unresolved_basin_color,
    trajectory_color,
    stable_manifold_color,
    unstable_manifold_color,
    fp_type_colors,
    order_colors,
    pastel_order_colors,
    cmap_state,
    norm_state,
    cmap_time,
    cmap_cells,
    potential_surface_cmap,
    rotational_surface_cmap,
    velocity_cmap,
    cycle_line_lightness_floor,
    cycle_line_lightness_scale,
    cycle_line_saturation_scale,
    phase_basin_palette,
    lighten_colors,
    indexed_cmap_color,
)


def update_params():
    plt.rcParams.update({"figure.dpi": figure_dpi})
# __________________________________________________________________________________________________

# def sync_order_colors_from_config():
#     global order_colors, pastel_order_colors, cmap_state, norm_state
#     order_colors = visuals_config.order_colors
#     pastel_order_colors = visuals_config.pastel_order_colors
#     cmap_state = visuals_config.cmap_state
#     norm_state = visuals_config.norm_state


# def set_order_colors(colors, lighten_amount=0.4):
#     visuals_config.set_order_colors(colors, lighten_amount=lighten_amount)
#     sync_order_colors_from_config()

#______________________________________________________________________________________________________________
# _____________________________________________________________________________________________________________

def visualize_landscape(landscape, xx, yy, regime, color_scheme='fp_types', draw_circles=True):
    """ Simple visualization of landscape flow and modules in one regime. """
    density = 0.5
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    circles = []
    for i, module in enumerate(landscape.module_list):
        if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
            curl[i] = 1

    if draw_circles:
        for i, module in enumerate(landscape.module_list):
            if module.a.size == 1 and module.s.size == 1 and regime == 0:
                sig = float(module.s)
                A = float(module.a)
            else:
                sig = module.s[regime]
                A = module.a[regime]

            if color_scheme == 'fp_types':
                color = fp_type_colors[module.__class__.__name__]
            elif color_scheme == 'order':
                color = order_colors[i]
            else:
                color = neutral_color

            # for negative amplitude - non-filled cicle
            if A < 0:
                fill = False
                lw = 2
            else:
                fill = True
                lw = 0
            circles.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                      fill=fill, alpha=0.22 * np.sqrt(np.abs(A)), clip_on=True, linewidth=lw))
    morphogen_times = landscape.morphogen_times
    landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
    (dX, dY), potential, rot_potential = landscape(float(regime), (xx, yy), return_potentials=True)

    fig, stream_ax = plt.subplots(1, 1, figsize=(5, 5))
    circles_ax = stream_ax
    if draw_circles:
        for circle in _ordered_circle_patches(circles):
            circles_ax.add_patch(copy(circle))

    stream_ax.streamplot(xx, yy, dX, dY, density=density, arrowsize=2., arrowstyle='->', linewidth=1,
                         color=streamline_color)
    stream_ax.contour(xx, yy, dX, (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
    stream_ax.contour(xx, yy, dY, (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

    stream_ax.set_xlim([np.min(xx), np.max(xx)])
    stream_ax.set_ylim([np.min(yy), np.max(yy)])
    stream_ax.set_xticks([])
    stream_ax.set_yticks([])
    landscape.morphogen_times = morphogen_times
    # plt.show()
    return fig


# _______________________________________________________________________________________________________


def visualize_landscape_t(landscape, xx, yy, t, color_scheme='fp_types', circles=True, nullclines=True, density=0.5,
                          traj_times=None, traj_init_cond=(0., 0.), traj_start=0, traj_color=trajectory_color, circle_opacity=0.25,
                          start_points=None, traj_arrow=True):
    """ Visualize the flow and modules at time t, with optional integrated trajectory in the frozen landscape. """

    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    circle_patches = []
    for i, module in enumerate(landscape.module_list):
        if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
            curl[i] = 1

    for i, module in enumerate(landscape.module_list):
        V, sig, A = module.get_current_pars(t, landscape.regime, *landscape.morphogen_times)
        if color_scheme == 'fp_types':
            color = fp_type_colors[module.__class__.__name__]
        elif color_scheme == 'order':
            color = order_colors[i]
        else:
            color = neutral_color

        if circles:
            # for negative amplitude - non-filled cicle
            if A < 0:
                fill = False
                lw = 2
            else:
                fill = True
                lw = 0
            circle_patches.append(plt.Circle((module.x, module.y), 1.18 * float(sig), color=color,
                                  fill=fill, alpha=circle_opacity * np.sqrt(float(np.abs(A))), clip_on=True, linewidth=lw))
    (dX, dY), potential, rot_potential = landscape(t, (xx, yy), return_potentials=True)

    fig, stream_ax = plt.subplots(1, 1, figsize=(5, 5))
    circles_ax = stream_ax

    if circles:
        for circle in _ordered_circle_patches(circle_patches):
            circles_ax.add_patch(copy(circle))
            circles_ax.set_xlim((np.min(xx), np.max(xx)))
            circles_ax.set_ylim((np.min(yy), np.max(yy)))

    stream_ax.streamplot(xx, yy, dX, dY, density=density, arrowsize=2., arrowstyle='->', linewidth=1,
                         color=streamline_color, start_points=start_points)
    if nullclines:
        stream_ax.contour(xx, yy, dX, (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
        stream_ax.contour(xx, yy, dY, (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

    if traj_times is not None:
        landscape.init_cells(1, traj_init_cond, noise=0.)
        traj, states = landscape.run_cells(traj_times[0], traj_times[1], traj_times[2], noise=0., ndt=50, frozen=True,
                                           t_freeze=t)
        stream_ax.plot(traj[0, 0, traj_start:], traj[1, 0, traj_start:], lw=3, color=traj_color)

        if traj_arrow:
            mid = traj_times[2] // 3 * 2
            x_coords = traj[0, 0, :]
            y_coords = traj[1, 0, :]
            arrow_size=0.3
            base = np.array([x_coords[mid], y_coords[mid]])
            direction = np.array([x_coords[mid] - x_coords[mid - 1], y_coords[mid] - y_coords[mid - 1], 0.])
            direction /= np.linalg.norm(direction)
            perp_vector = np.cross(direction, np.array([0, 0, 1]))[0:2]
            perp_vector /= np.linalg.norm(perp_vector)  # Normalize
            direction = direction[0:2]
            left = base + arrow_size * (perp_vector * 0.4 - direction)
            right = base + arrow_size * (-perp_vector * 0.4 - direction)
            stream_ax.plot(*zip(left, base, right), color=traj_color, linewidth=3, zorder=100)

    stream_ax.set_xlim(np.min(xx), np.max(xx))
    stream_ax.set_ylim(np.min(yy), np.max(yy))
    stream_ax.set_xticks([])
    stream_ax.set_yticks([])
    return fig, stream_ax

# ___________________________________________________________________________________________________
# __________________ Phase space plotting (fixed points, cycles, manifolds) _________________________

def _node_colors_and_cmap(landscape, colors=None, include_unassigned=True):
    base_colors = tuple(colors) if colors is not None else tuple(order_colors)
    if len(base_colors) == 0:
        base_colors = (neutral_color,)
    node_colors = tuple(base_colors[index % len(base_colors)] for index in range(landscape.n_nodes))
    if include_unassigned:
        node_cmap = ListedColormap([neutral_color] + list(node_colors))
        norm = BoundaryNorm(np.arange(len(node_colors) + 2) - 1.5, node_cmap.N)
    else:
        if node_colors:
            node_cmap = ListedColormap(node_colors)
            norm = BoundaryNorm(np.arange(len(node_colors) + 1) - 0.5, node_cmap.N)
        else:
            node_cmap = ListedColormap([neutral_color])
            norm = BoundaryNorm(np.array([-0.5, 0.5]), node_cmap.N)
    return node_colors, node_cmap, norm


def _phase_plot_colors(
    landscape,
    attractors,
    basin_grid,
    unresolved_color,
    basin_coloring='attractor',
    module_order_colors=None,
    x_coords=None,
    y_coords=None,
):
    basin_labels = None if basin_grid is None else np.asarray(basin_grid['basin_labels'], dtype=int)
    basin_coloring = str(basin_coloring).lower()

    if basin_coloring == 'attractor':
        cmap = phase_basin_palette(len(attractors), unresolved_color=unresolved_color)
        attractor_facecolors = {
            int(attractor['id']): indexed_cmap_color(cmap, int(attractor['id']) + 1)
            for attractor in attractors
        }
        if basin_labels is None:
            return None, cmap, None, attractor_facecolors
        norm = BoundaryNorm(np.arange(-0.5, len(attractors) + 1.5), cmap.N)
        basin_image = basin_labels + 1
        return basin_image, cmap, norm, attractor_facecolors

    if basin_coloring not in ('module', 'node'):
        raise ValueError("basin_coloring must be 'attractor', 'module', or 'node'.")

    node_labels = None if basin_grid is None else basin_grid.get('node_labels')
    if node_labels is not None:
        node_labels = np.asarray(node_labels, dtype=int)

    if module_order_colors is None:
        module_palette = pastel_order_colors
        attractor_palette = lighten_colors(tuple(order_colors), amount=0.4)
    else:
        module_palette = lighten_colors(tuple(module_order_colors), amount=0.7)
        attractor_palette = lighten_colors(tuple(module_order_colors), amount=0.2)

    colors = [unresolved_color]
    if len(module_palette) == 0:
        module_palette = (neutral_color,)
    if len(attractor_palette) == 0:
        attractor_palette = (neutral_color,)
    for node_state in range(landscape.n_nodes):
        colors.append(module_palette[node_state % len(module_palette)])

    attractor_color_ids = {}
    attractor_facecolors = {}
    no_node_attractors = []
    for attractor in attractors:
        attractor_id = int(attractor['id'])
        module_index = attractor.get('node_id')
        if module_index is None:
            no_node_attractors.append(attractor_id)
            continue
        attractor_color_ids[attractor_id] = int(module_index) + 1
        attractor_facecolors[attractor_id] = attractor_palette[int(module_index) % len(attractor_palette)]

    for attractor_id in no_node_attractors:
        colors.append(neutral_color)
        attractor_color_ids[attractor_id] = len(colors) - 1
        attractor_facecolors[attractor_id] = neutral_color

    if basin_labels is None:
        cmap = ListedColormap(colors)
        return None, cmap, None, attractor_facecolors

    basin_image = np.zeros_like(basin_labels, dtype=int)
    if node_labels is not None:
        for node_state in range(landscape.n_nodes):
            basin_image[node_labels == node_state] = node_state + 1
    else:
        for attractor_id, color_id in attractor_color_ids.items():
            basin_image[basin_labels == attractor_id] = int(color_id)

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)
    return basin_image, cmap, norm, attractor_facecolors


def _plot_phase_overlays(
    ax,
    attractors,
    fixed_points,
    cmap,
    attractor_facecolors=None,
    saddle_manifolds=None,
    show_saddle_manifolds=False,
    plot_stable_manifolds=True,
    show_cycles=True,
    show_fixed_points=True,
    stable_manifold_color=stable_manifold_color,
    unstable_manifold_color=unstable_manifold_color,
):
    if attractor_facecolors is None:
        attractor_facecolors = {}
    fixed_label_map = {
        attractor.get('fixed_point_index'): attractor['id']
        for attractor in attractors
        if attractor['type'] == 'fixed_point'
    }

    if show_saddle_manifolds and saddle_manifolds is not None:
        for saddle in saddle_manifolds.get('saddles', ()):
            if plot_stable_manifolds:
                for branch in saddle.get('stable', ()):
                    branch = np.asarray(branch, dtype=float)
                    if branch.shape[0] >= 2:
                        ax.plot(
                            branch[:, 0],
                            branch[:, 1],
                            color=stable_manifold_color,
                            linestyle='-',
                            linewidth=2.6,
                            alpha=0.95,
                            zorder=5,
                        )
            for branch in saddle.get('unstable', ()):
                branch = np.asarray(branch, dtype=float)
                if branch.shape[0] >= 2:
                    ax.plot(
                        branch[:, 0],
                        branch[:, 1],
                        color=unstable_manifold_color,
                        linestyle='-',
                        linewidth=2.6,
                        alpha=0.95,
                        zorder=5,
                    )

    if show_cycles:
        for attractor in attractors:
            if attractor['type'] != 'cycle':
                continue
            attractor_id = int(attractor['id'])
            color = attractor_facecolors.get(attractor_id, indexed_cmap_color(cmap, attractor_id + 1))
            h, l, s = colorsys.rgb_to_hls(*np.asarray(color[:3], dtype=float))
            dark_color = colorsys.hls_to_rgb(
                h,
                max(cycle_line_lightness_floor, cycle_line_lightness_scale * l),
                min(1.0, cycle_line_saturation_scale * s),
            )
            traj = np.asarray(attractor['trajectory'])
            period_steps = int(attractor.get('period_steps', 0))
            if period_steps > 0 and traj.shape[0] > period_steps + 1:
                traj = traj[-period_steps - 1:]
            ax.plot(traj[:, 0], traj[:, 1], color=dark_color, linewidth=2.9, zorder=6)

    if show_fixed_points and fixed_points['points'].size:
        marker_map = {
            'attractor': 'o',
            'repeller': 's',
            'saddle': 'X',
            'center_or_degenerate': 'D',
        }
        for idx, point in enumerate(fixed_points['points']):
            stability = str(fixed_points['stability'][idx])
            marker = marker_map.get(stability, 'o')
            if idx in fixed_label_map:
                attractor_id = fixed_label_map[idx]
                facecolor = attractor_facecolors.get(int(attractor_id), indexed_cmap_color(cmap, int(attractor_id) + 1))
                edgecolor = 'k'
                size = 150 #82
            else:
                facecolor = 'white'
                edgecolor = 'k'
                size = 120 # 72
            ax.scatter(
                point[0],
                point[1],
                marker=marker,
                facecolors=facecolor,
                edgecolors=edgecolor,
                linewidths=1.2,
                s=size,
                zorder=8,
            )


def plot_attractor_basins_t(
    landscape,
    xx,
    yy,
    t,
    phase_result=None,
    basin_grid=None,
    fixed_points=None,
    density=0.5,
    streamlines=True,
    nullclines=False,
    basin_alpha=0.65,
    unresolved_color=unresolved_basin_color,
    basin_coloring='attractor',
    module_order_colors=None,
    show_fixed_points=True,
    show_cycles=True,
    saddle_manifolds=None,
    show_saddle_manifolds=False,
    plot_stable_manifolds=True,
    stable_manifold_color=stable_manifold_color,
    unstable_manifold_color=unstable_manifold_color,
):
    if phase_result is None:
        phase_result = landscape.find_phase_objects_manifold(t, xx, yy, fixed_points=fixed_points)
    if basin_grid is None:
        basin_grid = landscape.find_attractor_basins_manifold(phase_result=phase_result)

    attractors = phase_result['attractors']
    fixed_points = phase_result['fixed_points']

    basin_image, cmap, norm, attractor_facecolors = _phase_plot_colors(
        landscape,
        attractors,
        basin_grid,
        unresolved_color=unresolved_color,
        basin_coloring=basin_coloring,
        module_order_colors=module_order_colors,
        x_coords=np.asarray(xx[0], dtype=float),
        y_coords=np.asarray(yy[:, 0], dtype=float),
    )
    extent = (np.min(xx), np.max(xx), np.min(yy), np.max(yy))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(
        basin_image,
        origin='lower',
        extent=extent,
        cmap=cmap,
        norm=norm,
        alpha=basin_alpha,
        interpolation='nearest',
    )

    draw_nullclines = bool(nullclines) and not bool(show_saddle_manifolds)

    if streamlines or draw_nullclines:
        dX, dY = landscape(t, (xx, yy), return_potentials=False)
        if streamlines:
            ax.streamplot(
                xx,
                yy,
                dX,
                dY,
                density=density,
                arrowsize=1.4,
                arrowstyle='->',
                linewidth=0.8,
                color=basin_streamline_color,
            )
        if draw_nullclines:
            ax.contour(xx, yy, dX, (0,), colors=('k',), linestyles='-', linewidths=1.0, alpha=0.5)
            ax.contour(xx, yy, dY, (0,), colors=('k',), linestyles='--', linewidths=1.0, alpha=0.5)

    if show_saddle_manifolds and saddle_manifolds is None:
        saddle_manifolds = phase_result['saddle_manifolds']

    _plot_phase_overlays(
        ax,
        attractors,
        fixed_points,
        cmap,
        attractor_facecolors=attractor_facecolors,
        saddle_manifolds=saddle_manifolds,
        show_saddle_manifolds=show_saddle_manifolds,
        plot_stable_manifolds=plot_stable_manifolds,
        show_cycles=show_cycles,
        show_fixed_points=show_fixed_points,
        stable_manifold_color=stable_manifold_color,
        unstable_manifold_color=unstable_manifold_color,
    )

    ax.set_xlim(np.min(xx), np.max(xx))
    ax.set_ylim(np.min(yy), np.max(yy))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    return fig, ax, basin_grid


def plot_phase_skeleton_t(
    landscape,
    xx,
    yy,
    t,
    phase_result=None,
    fixed_points=None,
    saddle_manifolds=None,
    show_fixed_points=True,
    show_cycles=True,
    show_saddle_manifolds=True,
    plot_stable_manifolds=True,
    unresolved_color=unresolved_basin_color,
    basin_coloring='attractor',
    module_order_colors=None,
    stable_manifold_color=stable_manifold_color,
    unstable_manifold_color=unstable_manifold_color,
    saddle_levels=False,
):
    if phase_result is None:
        phase_result = landscape.find_phase_objects_manifold(t, xx, yy, fixed_points=fixed_points)

    attractors = phase_result['attractors']
    fixed_points = phase_result['fixed_points']
    if (show_saddle_manifolds or saddle_levels) and saddle_manifolds is None:
        saddle_manifolds = phase_result['saddle_manifolds']

    _, cmap, _, attractor_facecolors = _phase_plot_colors(
        landscape,
        attractors,
        None,
        unresolved_color=unresolved_color,
        basin_coloring=basin_coloring,
        module_order_colors=module_order_colors,
        x_coords=np.asarray(xx[0], dtype=float),
        y_coords=np.asarray(yy[:, 0], dtype=float),
    )
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if saddle_levels and saddle_manifolds is not None:
        saddle_indices = [
            int(saddle['fixed_point_index'])
            for saddle in saddle_manifolds.get('saddles', ())
            if saddle.get('fixed_point_index') is not None
        ]
        if saddle_indices:
            saddle_indices = np.asarray(saddle_indices, dtype=int)
            saddle_points = np.asarray(fixed_points['points'], dtype=float)[saddle_indices]
            _, saddle_potential, _ = landscape(
                t,
                (saddle_points[:, 0], saddle_points[:, 1]),
                return_potentials=True,
            )
            _, potential, _ = landscape(t, (xx, yy), return_potentials=True)
            contour = plt.contour(xx, yy, potential, levels=np.unique(saddle_potential), alpha=0)
            for level_segments in contour.allsegs:
                for segment in level_segments:
                    if len(segment) >= 2:
                        ax.plot(segment[:, 0], segment[:, 1], color='#9CAF88', linewidth=2.6, zorder=4)

    _plot_phase_overlays(
        ax,
        attractors,
        fixed_points,
        cmap,
        attractor_facecolors=attractor_facecolors,
        saddle_manifolds=saddle_manifolds,
        show_saddle_manifolds=show_saddle_manifolds,
        plot_stable_manifolds=plot_stable_manifolds,
        show_cycles=show_cycles,
        show_fixed_points=show_fixed_points,
        stable_manifold_color=stable_manifold_color,
        unstable_manifold_color=unstable_manifold_color,
    )

    ax.set_xlim(np.min(xx), np.max(xx))
    ax.set_ylim(np.min(yy), np.max(yy))
    ax.set_aspect('equal')
    ax.set_axis_off()
    return fig, ax, phase_result


# def plot_phase_skeleton_on_potential_t(
#     landscape,
#     xx,
#     yy,
#     t,
#     basin_result=None,
#     fixed_points=None,
#     saddle_manifolds=None,
#     show_fixed_points=True,
#     show_cycles=True,
#     show_saddle_manifolds=True,
#     unresolved_color=unresolved_basin_color,
#     basin_coloring='attractor',
#     module_order_colors=None,
#     stable_manifold_color=stable_manifold_color,
#     unstable_manifold_color=unstable_manifold_color,
#     color_surface_by_basin=False,
#     basin_alpha=0.95,
#     elev=None,
#     azim=None,
#     offset=2,
#     cmap_center=None,
#     rot=False,
#     zlim=None,
#     axes=False,
#     z_lift=None,
# ):
#     if basin_result is None:
#         basin_result = landscape.find_attractor_basins(t, xx, yy, fixed_points=fixed_points)
#
#     attractors = basin_result['attractors']
#     fixed_points = basin_result['fixed_points']
#     if show_saddle_manifolds and saddle_manifolds is None:
#         saddle_manifolds = landscape.find_saddle_manifolds(
#             t,
#             fixed_points=fixed_points,
#             x_range=(float(np.min(xx)), float(np.max(xx))),
#             y_range=(float(np.min(yy)), float(np.max(yy))),
#         )
#
#     (dX, dY), potential, rot_potential = landscape(t, (xx, yy), return_potentials=True)
#     del dX, dY
#     surface = np.asarray(rot_potential if rot else potential, dtype=float)
#     surface_cmap = rotational_surface_cmap if rot else potential_surface_cmap
#     if cmap_center is None:
#         cmap_center = float(surface[0, 0])
#
#     basin_image, cmap, norm, attractor_facecolors = _phase_plot_colors(
#         landscape,
#         basin_result,
#         unresolved_color=unresolved_color,
#         basin_coloring=basin_coloring,
#         module_order_colors=module_order_colors,
#         x_coords=np.asarray(xx[0], dtype=float),
#         y_coords=np.asarray(yy[:, 0], dtype=float),
#     )
#     fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(6, 6), facecolor=figure_face_color)
#     ax.set_facecolor(axes_face_color)
#     ax.view_init(elev=elev, azim=azim)
#
#     if zlim is None:
#         ax.set_zlim([np.min(surface) - offset, np.max(surface) + 2])
#         zlow = float(np.min(surface) - offset)
#     else:
#         ax.set_zlim(zlim)
#         zlow = float(zlim[0])
#
#     if z_lift is None:
#         surface_span = float(np.max(surface) - np.min(surface))
#         z_lift = max(1e-3, 0.01 * max(surface_span, 1.0))
#
#     if color_surface_by_basin:
#         facecolors = cmap(norm(basin_image))
#         facecolors[..., 3] = basin_alpha
#         ax.contour(
#             xx,
#             yy,
#             surface,
#             zdir='z',
#             offset=zlow,
#             colors=(projection_contour_color,),
#             linewidths=0.8,
#             alpha=0.4,
#         )
#         ax.plot_surface(
#             xx,
#             yy,
#             surface,
#             facecolors=facecolors,
#             linewidth=0,
#             antialiased=False,
#             shade=False,
#         )
#     else:
#         ax.contour(xx, yy, surface, zdir='z', offset=zlow, cmap=surface_cmap, norm=CenteredNorm(cmap_center))
#         ax.plot_surface(
#             xx,
#             yy,
#             surface,
#             cmap=surface_cmap,
#             linewidth=0,
#             antialiased=False,
#             norm=CenteredNorm(cmap_center),
#         )
#
#     fixed_label_map = {
#         attractor.get('fixed_point_index'): attractor['id']
#         for attractor in attractors
#         if attractor['type'] == 'fixed_point'
#     }
#
#     def basin_color_for_id(attractor_id):
#         if attractor_color_ids is not None and int(attractor_id) in attractor_color_ids:
#             color_index = attractor_color_ids[int(attractor_id)]
#         else:
#             color_index = int(attractor_id) + 1
#         return _indexed_cmap_color(cmap, color_index)
#
#     def surface_values(x_coords, y_coords):
#         _, potential_values, rot_potential = landscape(t, (x_coords, y_coords), return_potentials=True)
#         return np.asarray(rot_potential if rot else potential_values, dtype=float)
#
#     if show_saddle_manifolds and saddle_manifolds is not None:
#         _plot_saddle_manifolds_3d(
#             ax,
#             landscape,
#             t,
#             saddle_manifolds,
#             stable_manifold_color=stable_manifold_color,
#             unstable_manifold_color=unstable_manifold_color,
#             rot=rot,
#             z_lift=z_lift,
#         )
#
#     if show_cycles:
#         for attractor in attractors:
#             if attractor['type'] != 'cycle':
#                 continue
#             color = basin_color_for_id(attractor['id'])
#             h, l, s = colorsys.rgb_to_hls(*np.asarray(color[:3], dtype=float))
#             dark_color = colorsys.hls_to_rgb(
#                 h,
#                 max(cycle_line_lightness_floor, cycle_line_lightness_scale * l),
#                 min(1.0, cycle_line_saturation_scale * s),
#             )
#             traj = np.asarray(attractor['trajectory'])
#             z_coords = surface_values(traj[:, 0], traj[:, 1]) + z_lift
#             ax.plot(traj[:, 0], traj[:, 1], z_coords, color=dark_color, linewidth=2.9, zorder=6)
#
#     if show_fixed_points and fixed_points['points'].size:
#         marker_map = {
#             'attractor': 'o',
#             'repeller': 's',
#             'saddle': 'X',
#             'center_or_degenerate': 'D',
#         }
#         points = np.asarray(fixed_points['points'], dtype=float)
#         z_points = surface_values(points[:, 0], points[:, 1]) + z_lift
#         for idx, point in enumerate(points):
#             stability = str(fixed_points['stability'][idx])
#             marker = marker_map.get(stability, 'o')
#             if idx in fixed_label_map:
#                 attractor_id = fixed_label_map[idx]
#                 facecolor = attractor_facecolors.get(attractor_id, basin_color_for_id(attractor_id))
#                 edgecolor = outline_color
#                 size = 82
#             else:
#                 facecolor = unassigned_fixed_point_face_color
#                 edgecolor = outline_color
#                 size = 72
#             ax.scatter(
#                 point[0],
#                 point[1],
#                 z_points[idx],
#                 marker=marker,
#                 facecolors=facecolor,
#                 edgecolors=edgecolor,
#                 linewidths=1.2,
#                 s=size,
#                 zorder=8,
#                 depthshade=False,
#             )
#
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.zaxis.set_tick_params(color=axes_face_color)
#     ax.set_zticklabels([])
#     ax.xaxis.set_pane_color(transparent_pane_color)
#     ax.yaxis.set_pane_color(transparent_pane_color)
#     ax.zaxis.set_pane_color(transparent_pane_color)
#     if not axes:
#         ax.set_axis_off()
#     return fig, ax, basin_result

# _____________________________________________________________________________________________________

def visualize_all(landscape, xx, yy, times, density=0.5, color_scheme='fp_types',
                  plot_velocities=True, plot_nullclines=True,
                  plot_traj=True, traj_times=(0., 100., 150), traj_start=50, traj_init_cond=(0., 1.), traj_noise=0., ):

    """
    Plot 4 panels: potential contour plot, rotational potential contour plot, flow plot with module circles,
    and flow plot with velocity magnitude
    :param landscape:
    :param xx:
    :param yy:
    :param times:
    :param density:
    :param color_scheme:
    :param plot_velocities:
    :param plot_nullclines:
    :param plot_traj:
    :param traj_times:
    :param traj_start:
    :param traj_init_cond:
    :param traj_noise:
    :return:
    """
    dX, dY = np.zeros((len(times), *xx.shape)), np.zeros((len(times), *xx.shape))
    
    figures = []

    for it in range(len(times)):

        (dX[it], dY[it]), potential, rot_potential = landscape(times[it], (xx, yy), return_potentials=True)

        circles = []
        for i, module in enumerate(landscape.module_list):
            V, sig, A = module.get_current_pars(times[it], landscape.regime, *landscape.morphogen_times)
            if color_scheme == 'fp_types':
                color = fp_type_colors[module.__class__.__name__]
            elif color_scheme == 'order':
                color = order_colors[i]
            else:
                color = neutral_color
            # for negative amplitude - non-filled cicle
            if A < 0:
                fill = False
                lw = 2
            else:
                fill = True
                lw = 0
            circles.append(plt.Circle((module.x, module.y), 1.18 * float(sig), color=color,
                                      fill=fill, alpha=0.22 * np.sqrt(float(np.abs(A))), clip_on=True, linewidth=lw))

        vrange = (np.max(rot_potential) - np.min(rot_potential))/2.
        if vrange == 0.:
            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            circles_ax = ax[1]
            stream_ax = ax[2]
            # vrange = 1.
        else:
            fig, ax = plt.subplots(1, 4, figsize=(18, 4))
            circles_ax = ax[2]
            stream_ax = ax[3]
            ax[1].imshow(rot_potential, cmap=rotational_surface_cmap, origin='lower', norm=CenteredNorm(0, vrange),
                         extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
            ax[1].contour(xx, yy, rot_potential, colors=surface_contour_color, linestyles='solid', origin='lower')

        ax[0].imshow(potential, cmap=potential_surface_cmap, origin='lower', norm=CenteredNorm(0),
                     extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
        ax[0].contour(xx, yy, -potential, origin='lower', colors=surface_contour_color)

        for iax in range(len(ax)):
            ax[iax].set_xticks([])
            ax[iax].set_yticks([])
            ax[iax].set_xlim((np.min(xx), np.max(xx)))
            ax[iax].set_ylim((np.min(yy), np.max(yy)))

        for circle in _ordered_circle_patches(circles):
            circles_ax.add_patch(copy(circle))

        if plot_velocities:
            velocities_sq = dX[it] ** 2 + dY[it] ** 2
            velocities = np.sqrt(velocities_sq)
            # print('Min velocity:', round(np.min(velocities), 3), ', Max:', round(np.max(velocities), 3),
            #       ', Mean:', round(np.mean(velocities), 3), ', Median:', round(np.median(velocities), 3))

            stream_ax.imshow(velocities, alpha=0.5, cmap=velocity_cmap, origin='lower',
                             extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))

            # An attempt to plot fixed points - often ends up missing some points - high resolution needed
            # fp_labels, nlabels = label(velocities_sq < 0.5, return_num=True)
            # for l in range(nlabels):
            #     if np.sum(fp_labels == l) <= 1000:
            #         fp = fp_labels == l
            #         stream_ax.imshow(fp, alpha=0.5, cmap='Blues', origin='lower',
            #                          extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
            # if np.sum(fp_labels == l) <= 50:
            # fp = velocities_sq == np.min(velocities_sq[fp_labels == l])
            # if np.sum(fp_labels == l) > 20:
            #     fp = (velocities_sq < 5e-4) * fp_labels == l
            # else:
            #     fp = fp_labels == l
            # stream_ax.scatter(xx[fp], yy[fp], marker='o', s=50, color='gold', edgecolor=None, zorder=10)

        circles_ax.streamplot(xx, yy, dX[it], dY[it], density=density, arrowsize=2., arrowstyle='->',
                              linewidth=1,
                              color=streamline_color)
        stream_ax.streamplot(xx, yy, dX[it], dY[it], density=density, arrowsize=2., arrowstyle='->',
                             linewidth=1,
                             color=streamline_color)

        if plot_nullclines:
            circles_ax.contour(xx, yy, dX[it], (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
            circles_ax.contour(xx, yy, dY[it], (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)
            stream_ax.contour(xx, yy, dX[it], (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
            stream_ax.contour(xx, yy, dY[it], (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

        if plot_traj:
            # calculate a trajectory in frozen landscape
            landscape.init_cells(1, traj_init_cond, noise=traj_noise)
            traj, states = landscape.run_cells(traj_times[0], traj_times[1], traj_times[2], noise=traj_noise,
                                               ndt=50, frozen=True, t_freeze=times[it])
            stream_ax.plot(traj[0, 0, traj_start:], traj[1, 0, traj_start:], lw=3, color=trajectory_color)



        figures.append(fig)
        # plt.show()

    return figures


# _____________________________________________________________________________________________________________________
# _______________ 3D plot _____________________________________________________________________________________________

def visualize_potential(landscape, xx, yy, regime=None, t=None, color_scheme='fp_types', elev=None, azim=None, offset=2,
                        cmap_center=None, rot=False, rot_contour=False, min_contour_segment=80, scatter=False, zlim=None, axes=True,
                        phase_result=None, show_saddle_manifolds=False, saddle_levels=False):
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    # circles = []
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(6, 6))
    ax.view_init(elev=elev, azim=azim)

    if t is None and regime is not None:
        morphogen_times = landscape.morphogen_times
        landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
        t = float(regime)
    (dX, dY), potential, rot_potential = landscape(t, (xx, yy), return_potentials=True)
    if cmap_center is None:
        cmap_center = potential[0, 0]
    if rot:
        potential = rot_potential
        cmap = rotational_surface_cmap
    else:
        cmap = potential_surface_cmap

    if zlim is None:
        ax.set_zlim([np.min(potential) - offset, np.max(potential) + 2])
        zlow = np.min(potential) - offset
    else:
        ax.set_zlim(zlim)
        zlow = zlim[0]

    ax.contour(xx, yy, potential, zdir='z', offset=zlow, cmap=cmap, norm=CenteredNorm(cmap_center))
    ax.plot_surface(xx, yy, potential, cmap=cmap, linewidth=0, antialiased=False, norm=CenteredNorm(cmap_center))
    if rot_contour:
        contour = plt.contour(xx, yy, rot_potential, levels=7, alpha=0)
        cmap_contour = plt.get_cmap(rotational_surface_cmap)
        norm = CenteredNorm(0., halfrange=np.max(np.abs(rot_potential)))
        for i, level_segments in enumerate(contour.allsegs[::-1]):
            level_value = contour.levels[-i]
            line_color = cmap_contour(norm(level_value))
            for segment in level_segments:
                if len(segment) < min_contour_segment:
                    continue  # Skip small segments
                x_coords = segment[:, 0]
                y_coords = segment[:, 1]
                derivs, z_coords, rot_z = landscape(t, (x_coords, y_coords), return_potentials=True)
                ax.plot(x_coords, y_coords, z_coords, color=line_color, linestyle='-', linewidth=2, zorder=100)

                arrow_size = 0.3

                if len(segment) > 80:
                    for mid in (len(x_coords) // 3, len(x_coords)//3*2):
                        base = np.array([x_coords[mid], y_coords[mid], z_coords[mid]])
                        direction = np.array([x_coords[mid] - x_coords[mid-1], y_coords[mid] - y_coords[mid-1],
                                              z_coords[mid] - z_coords[mid-1]])
                        direction /= np.linalg.norm(direction)
                        # if level_value < 0:
                        #     direction = -direction
                        perp_vector = np.cross(direction, np.array([0, 0, 1]))
                        perp_vector /= np.linalg.norm(perp_vector)  # Normalize
                        left = base + arrow_size * (perp_vector * 0.4 - direction)
                        right = base + arrow_size * (-perp_vector * 0.4 - direction)
                        ax.plot(*zip(left, base, right), color=line_color, linewidth=1.5, zorder=100)
    if phase_result:
        show_saddle_manifolds = True
    if show_saddle_manifolds or saddle_levels:
        z_lift = 0

        if phase_result is None:
            phase_result = landscape.find_phase_objects_manifold(
                t,
                xx,
                yy,
            )
        fixed_points = phase_result.get('fixed_points')
        saddle_manifolds = phase_result.get('saddle_manifolds')
        _plot_saddle_manifolds_3d(
            ax,
            landscape,
            t,
            saddle_manifolds,
            stable_manifold_color=stable_manifold_color,
            unstable_manifold_color=unstable_manifold_color,
            rot=rot,
            z_lift=z_lift,
        )
        attractor_indices = [
            int(attractor['fixed_point_index'])
            for attractor in phase_result.get('attractors', ())
            if attractor.get('type') == 'fixed_point' and attractor.get('fixed_point_index') is not None
        ]
        if attractor_indices:
            attractor_indices = np.asarray(attractor_indices, dtype=int)
            attractor_points = np.asarray(fixed_points['points'], dtype=float)[attractor_indices]
            _, potential_values, rot_values = landscape(
                t,
                (attractor_points[:, 0], attractor_points[:, 1]),
                return_potentials=True,
            )
            attractor_z = np.asarray(rot_values if rot else potential_values, dtype=float) + z_lift
            ax.plot(
                attractor_points[:, 0],
                attractor_points[:, 1],
                attractor_z,
                markersize=4,
                lw=0,
                color='w',
                marker='o',
                # edgecolor=None,
                zorder=120,
                # depthshade=False,
            )

        saddle_indices = [
            int(saddle['fixed_point_index'])
            for saddle in saddle_manifolds.get('saddles', ())
            if saddle.get('fixed_point_index') is not None
        ]
        if saddle_indices:
            saddle_indices = np.asarray(saddle_indices, dtype=int)
            saddle_points = np.asarray(fixed_points['points'], dtype=float)[saddle_indices]
            _, potential_values, rot_values = landscape(
                t,
                (saddle_points[:, 0], saddle_points[:, 1]),
                return_potentials=True,
            )
            saddle_z = np.asarray(rot_values if rot else potential_values, dtype=float) + z_lift
            if saddle_levels:
                contour = plt.contour(xx, yy, potential, levels=np.unique(saddle_z - z_lift), alpha=0)
                for level_value, level_segments in zip(contour.levels, contour.allsegs):
                    for segment in level_segments:
                        if len(segment) < 2:
                            continue
                        ax.plot(
                            segment[:, 0],
                            segment[:, 1],
                            np.full(len(segment), level_value + z_lift),
                            color='w',
                            linewidth=1.2,
                            zorder=115,
                        )
            ax.plot(
                saddle_points[:, 0],
                saddle_points[:, 1],
                saddle_z,
                markersize=4,
                markeredgewidth=2.6,
                lw=0,
                color='w',
                marker='x',
                zorder=120,
            )

    if scatter:
        for i, module in enumerate(landscape.module_list):
            if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
                curl[i] = 1
            if color_scheme == 'fp_types':
                color = fp_type_colors[module.__class__.__name__]
            elif color_scheme == 'order':
                color = order_colors[i]
            else:
                color = neutral_color
            ax.scatter(module.x, module.y, zlow, s=25, color=color, marker='D', zorder=20)

    if regime is not None:
        landscape.morphogen_times = morphogen_times

    ax.set_xticks([])
    ax.set_yticks([])
    ax.zaxis.set_tick_params(color='white')
    ax.set_zticklabels([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # plt.tight_layout()
    if not axes:
        ax.set_axis_off()
    # plt.show()
    return fig


def _plot_saddle_manifolds_3d(
    ax,
    landscape,
    t,
    saddle_manifolds,
    stable_manifold_color=stable_manifold_color,
    unstable_manifold_color=unstable_manifold_color,
    rot=False,
    z_lift=0.0,
):
    if saddle_manifolds is None:
        return

    def surface_values(x_coords, y_coords):
        _, potential_values, rot_potential = landscape(t, (x_coords, y_coords), return_potentials=True)
        return np.asarray(rot_potential if rot else potential_values, dtype=float)

    for saddle in saddle_manifolds.get('saddles', ()):
        # for branch in saddle.get('stable', ()):
        #     branch = np.asarray(branch, dtype=float)
        #     if branch.shape[0] >= 2:
        #         z_coords = surface_values(branch[:, 0], branch[:, 1]) + z_lift
        #         ax.plot(
        #             branch[:, 0],
        #             branch[:, 1],
        #             z_coords,
        #             color='steelblue',
        #             linestyle='-',
        #             linewidth=2.,
        #             alpha=1.,
        #             zorder=100,
        #         )
        for branch in saddle.get('unstable', ()):
            branch = np.asarray(branch, dtype=float)
            if branch.shape[0] >= 2:
                z_coords = surface_values(branch[:, 0], branch[:, 1]) + z_lift
                ax.plot(
                    branch[:, 0],
                    branch[:, 1],
                    z_coords,
                    color='pink',
                    linestyle='-',
                    linewidth=2.,
                    alpha=1.,
                    zorder=110,
                )

# ____________________________ Cells, trajectories, misc __________________________________________________________

def visualize_cell_states(landscape, xx, yy, t, abs_threshold=0.):
    cell_states = landscape.get_cell_states(t, np.array((xx.flatten(), yy.flatten())), abs_threshold=abs_threshold)
    _, node_cmap, norm = _node_colors_and_cmap(landscape, include_unassigned=True)
    fig, ax = visualize_landscape_t(landscape, xx, yy, t=t, color_scheme='order')
    # plt.figure()
    plt.imshow(np.reshape(cell_states, xx.shape), cmap=node_cmap, norm=norm, origin='lower',
               extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)), alpha=0.3, interpolation='nearest')
    return fig, ax


def plot_cells(landscape, L, colors=None):
    """ Plot the current cell locations and states """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    coord = landscape.cell_coordinates
    states = landscape.cell_states
    _, node_cmap, norm = _node_colors_and_cmap(landscape, colors=colors, include_unassigned=True)
    ax.scatter(coord[0], coord[1], s=8, alpha=0.3, c=states, cmap=node_cmap, norm=norm, edgecolors=None)
    ax.set_xlim([-L, L])
    ax.set_ylim([-L, L])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    return fig


def get_and_plot_traj(landscape, t0, tf, nt, L, noise, ndt=50, s=6, frozen=False, t_freeze=None,
                      state_measure='gaussian', state_colors=None, state_names=None, t_ticks=None, t_names=None):
    """ Integrate trajectories for cells and visualize in 2 panels: colored by time and by cell state. """
    basin_q = np.linspace(-L, L, 201)
    basin_xx, basin_yy = np.meshgrid(basin_q, basin_q, indexing='xy')
    traj, states = landscape.run_cells(
        t0,
        tf,
        nt,
        noise=noise,
        ndt=ndt,
        frozen=frozen,
        t_freeze=t_freeze,
        get_states=state_measure,
        basin_xx=basin_xx,
        basin_yy=basin_yy,
    )

    fig = plt.figure(figsize=(9, 5))
    gs = GridSpec(2, 2, height_ratios=[20, 1], hspace=0.1, wspace=0.05)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_cbar, ax_state_cbar = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    node_colors, node_cmap, norm = _node_colors_and_cmap(
        landscape,
        colors=state_colors,
        include_unassigned=True,
    )

    time_values = np.tile(np.arange(nt), (states.shape[0], 1))
    ax0.scatter(traj[0, :, :], traj[1, :, :], s=s, alpha=0.2, c=time_values, cmap=cmap_time, edgecolor=None)
    sc0 = ax0.scatter(2*L*np.ones(nt), 2*L*np.ones(nt), c=np.linspace(t0, tf, nt), cmap=cmap_time, alpha=0.7, s=0.01, edgecolors='none')
    tbar = fig.colorbar(sc0, cax=ax_cbar, orientation='horizontal', label='Time')
    if t_ticks is not None:
        if isinstance(t_ticks, int):
            tbar.set_ticks(np.linspace(t0, tf, t_ticks))
        else:
            tbar.set_ticks(np.linspace(*t_ticks))
    else:
        tbar.set_ticks(np.linspace(t0, tf, 7))
    tbar.ax.tick_params(which='minor', length=0)
    if t_names is not None:
        tbar.set_ticklabels(t_names, fontsize=10)

    ax1.scatter(traj[0, :, :], traj[1, :, :], s=s, alpha=0.2, c=states, cmap=node_cmap, norm=norm, edgecolor=None)
    if landscape.n_nodes > 0:
        state_ticks = np.arange(landscape.n_nodes)
        sc1 = ax1.scatter(
            [2 * L] * landscape.n_nodes,
            [2 * L] * landscape.n_nodes,
            c=state_ticks,
            cmap=node_cmap,
            norm=norm,
            alpha=0.5,
            s=0.01,
        )
    else:
        state_ticks = np.array([], dtype=int)
        sc1 = ax1.scatter([2 * L], [2 * L], c=[-1], cmap=node_cmap, norm=norm, alpha=0.5, s=0.01)
    cbar = fig.colorbar(sc1, cax=ax_state_cbar, orientation='horizontal', label='Cell state')

    cbar.set_ticks(state_ticks)
    cbar.ax.tick_params(which='both', length=0)
    if state_names is not None:
        cbar.set_ticklabels(state_names)

    for ax in [ax0, ax1]:
        ax.set(xlim=[-L, L], ylim=[-L, L], aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def _ordered_circle_patches(circle_patches):
    return [
        patch
        for _, _, patch in sorted(
            (
                (-float(getattr(patch, "radius", 0.0)), index, patch)
                for index, patch in enumerate(circle_patches)
            ),
            key=lambda item: (item[0], item[1]),
        )
    ]


def circle_plot(landscape, regime=None, L=6, color_scheme='order', lw=4):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    circles = []
    if regime is not None:
        regimes = (regime,)
    else:
        regimes = range(len(landscape.module_list[0].a))
    for i in range(len(landscape.module_list)):
        m = landscape.module_list[i]
        if color_scheme == 'fp_types':
            color = fp_type_colors[m.__class__.__name__]
        elif color_scheme == 'order':
            color = order_colors[i]
        else:
            color = neutral_color
        for j in regimes:
            circle = plt.Circle((m.x, m.y), 1.18 * m.s[j], color=color, fill=True,
                                alpha=0.25 * np.sqrt(m.a[j]), clip_on=False, linewidth=0,
                                linestyle='solid', zorder=i * 10)
            circles.append(circle)
    for circle in _ordered_circle_patches(circles):
        ax.add_patch(circle)
    ax.set_xlim((-L, L))
    ax.set_ylim((-L, L))
    ax.axis('off')
    ax.set_aspect('equal')
    return fig


def make_movie_discrete(landscape, xx, yy, labels, time_pars, n_cells, noise, init_cond=0,
                        circles=True, circle_opacity=0.1, density=0.65, nullclines=False,
                        fps=10, save_dir='', filename='movie.gif'):
    """
    Generate a trajectory movie with background streamplots that change between regimes
    """
    n_frames = time_pars[2]
    streamplots = [] # generate streamplots only once per condition
    for i in range(len(labels)):
        t_stream = landscape.morphogen_times[i-1] if i > 0 else time_pars[0]
        fig, ax = visualize_landscape_t(landscape, xx, yy, t_stream, color_scheme='order', circles=circles,
                                            nullclines=nullclines, circle_opacity=circle_opacity, density=density)
        ax.text(0.02, 0.95, labels[i], transform=ax.transAxes, fontsize=15, fontweight='bold')
        streamplots.append((fig, ax))

    landscape.init_cells(n_cells, init_cond, noise)
    times = np.linspace(*time_pars)
    traj, states = landscape.run_cells(*time_pars, noise, ndt=10, frozen=False)
    for i in range(n_frames):
        regime = mr_current_regime(times[i], *landscape.morphogen_times)
        fig, ax = streamplots[regime]
        sc = ax.scatter(traj[0, :, i], traj[1, :, i], s=25, alpha=1., c=states[:, i], cmap=cmap_state, norm=norm_state, zorder=10)
        fig.savefig(save_dir+f"frame_{i:03d}.png", dpi=150, bbox_inches='tight')
        sc.remove()

    for fig, ax in streamplots:
        plt.close(fig)

    frames = [imageio.imread(save_dir+f"frame_{i:03d}.png") for i in range(n_frames)]
    imageio.mimsave(save_dir+filename, frames, fps=fps)
    del frames
    print(f"Movie saved to {save_dir+filename}")


def make_movie(landscape, xx, yy, time_pars, n_cells, noise, init_cond=0,
               circles=True, circle_opacity=0.1, density=0.65, nullclines=False,
               traj_times=None, traj_init_cond=None, traj_start=50,
               fps=10, save_dir='', filename='movie.gif'):
    """
    Generate a trajectory movie with a continuously changing background
    """
    n_frames = time_pars[2]
    # Starting points for streamlines
    all_points = np.column_stack([xx.ravel(), yy.ravel()])
    K = 25  # number of streamlines
    indices = np.random.choice(len(all_points), K, replace=False)
    start_points = all_points[indices]

    landscape.init_cells(n_cells, init_cond, noise)
    times = np.linspace(*time_pars)
    traj, states = landscape.run_cells(*time_pars, noise, ndt=10, frozen=False)
    for i in range(n_frames):
        fig, ax = visualize_landscape_t(landscape, xx, yy, times[i], color_scheme='fp_types', circles=circles,
                                        nullclines=nullclines, circle_opacity=circle_opacity, density=density,
                                        start_points=start_points, traj_times=traj_times,
                                        traj_init_cond=traj_init_cond, traj_start=traj_start, traj_arrow=False)
        sc = ax.scatter(traj[0, :, i], traj[1, :, i], s=50, alpha=1., c=np.arange(n_cells), cmap=cmap_cells, zorder=10)
        fig.savefig(save_dir+f"frame_{i:03d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    frames = [imageio.imread(save_dir+f"frame_{i:03d}.png") for i in range(n_frames)]
    imageio.mimsave(save_dir+filename, frames, fps=fps)
    del frames
    print(f"Movie saved to {save_dir+filename}")




# old function - plotting trajectories with colored segments
#         def plot_trajectories(self, n, times, L, noise, init_cond=None, ndt=10, color_scheme='state', slow=None):
#             if isinstance(init_cond, int):
#                 # print('int')
#                 module0 = self.module_list[init_cond]
#                 init_cond = np.array((module0.x, module0.y))
#             elif init_cond is None:
#                 init_cond = self.init_cond
#             if color_scheme == 'state':
#                 cmap = ListedColormap(['indianred', 'tab:orange', 'gold', 'tab:green', 'tab:blue', 'tab:purple', 'm'])
#                 norm = BoundaryNorm(np.arange(8) - 0.5, cmap.N)
#             elif color_scheme == 'time':
#                 cmap = 'viridis'
#                 norm = Normalize()
#             plt.figure(figsize=(6, 6))
#             for i in range(n):
#                 traj = self.get_trajectory_noisy(times, noise=noise, ndt=ndt, init_cond=init_cond, slow=slow)
#                 if color_scheme == 'state':
#                     states = np.zeros(len(times)).astype('int')
#                     for it in range(len(times)):
#                         states[it] = self.get_cell_state(traj[:, it])
#                 points = traj.T.reshape(-1, 1, 2)
#                 segments = np.concatenate([points[:-1], points[1:]], axis=1)
#
#                 lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.4)
#                 if color_scheme == 'state':
#                     lc.set_array(states)
#                 elif color_scheme == 'time':
#                     lc.set_array(np.arange(len(times)))
#                 lc.set_linewidth(1.5)
#                 plt.gca().add_collection(lc)
#                 # plt.plot(traj[0], traj[1], lw=1, c='w')
#             plt.xlim([-L, L])
#             plt.ylim([-L, L])
#             plt.show()

# ______________________________________________________________________________________________________________________
    # if plot_weights:
    #     fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    #     # for iax in range(4):
    #         # ax[iax].axis('off')
    #     ax[0].imshow(potential, cmap='Greens', origin='lower', extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
    #     ax[0].contour(xx, yy, potential, origin='lower', colors='w')
    #
    #     ax[1].imshow(rot_potential, cmap='RdBu_r', origin='lower', norm=CenteredNorm(0), extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
    #     ax[1].contour(xx, yy, rot_potential, colors='w', linestyles='solid', origin='lower', levels=12)
    #
    #     for iax in range(2):
    #         ax[iax].set_xticks([])
    #         ax[iax].set_yticks([])
    #
    #     # can add nullclines but they make the plot busy
    #         for i, module in enumerate(self.module_list):
    #             ax[iax].scatter(module.x, module.y, marker='x', c='k')
    #     circles_ax = ax[2]
    #     stream_ax = ax[3]

    # if plot_velocities:
    #     # fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    #     velocities_sq = dX[it] ** 2 + dY[it] ** 2
    #     velocities = np.sqrt(velocities_sq)
    #
    #     # vel_plot = ax[0].imshow(velocities, cmap='BuPu', origin='lower', vmin=0, vmax=1.)
    #     # plt.colorbar(vel_plot)
    #     print('Min velocity:', round(np.min(velocities), 3), ', Max:', round(np.max(velocities), 3),
    #           ', Mean:', round(np.mean(velocities), 3), ', Median:', round(np.median(velocities), 3))
    #     # stream_ax = ax[1]

    # if plot_velocities:
    #     stream_ax.imshow(velocities, alpha=0.5, cmap='Greys', origin='lower', extent = (np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
    #     fp_labels, nlabels = label(velocities_sq < 1e-3, return_num=True)
    #     for l in range(nlabels):
    #         # if np.sum(fp_labels == l) <= 50:
    #         fp = velocities_sq == np.min(velocities_sq[fp_labels == l])
    #         # if np.sum(fp_labels == l) > 20:
    #         #     fp = (velocities_sq < 5e-4) * fp_labels == l
    #         # else:
    #         #     fp = fp_labels == l
    #         stream_ax.scatter(xx[fp], yy[fp], marker='o', s=50, color='gold', edgecolor=None, zorder=10)
    #     # stream_ax.(velocities_sq<1e-, cmap='viridis', origin='lower', extent = (np.min(xx), np.max(xx), np.min(yy), np.max(yy)), interpolation=None)
