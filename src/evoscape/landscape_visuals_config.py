import numpy as np
import matplotlib.pyplot as plt

from cmcrameri import cm as scm
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb


figure_dpi = 200

figure_face_color = 'white'
axes_face_color = 'white'
streamline_color = 'grey'
basin_streamline_color = 'k'
outline_color = 'k'
surface_contour_color = 'w'
neutral_color = 'grey'
unresolved_basin_color = 'lightgrey'
unassigned_fixed_point_face_color = 'white'
projection_contour_color = '0.6'
transparent_pane_color = (1.0, 1.0, 1.0, 0.0)

trajectory_color = 'forestgreen'
stable_manifold_color = '#2f5aa6'
unstable_manifold_color = '#a34747'

fp_type_colors = {
    'Node': 'tab:green',
    'UnstableNode': 'tab:blue',
    'Center': 'tab:purple',
    'NegCenter': 'hotpink',
}

order_colors = (
    'indianred',
    'tab:orange',
    'gold',
    'tab:green',
    'tab:blue',
    'tab:purple',
)

cmap_state = ListedColormap(order_colors)
norm_state = BoundaryNorm(np.arange(len(order_colors) + 1) - 0.5, cmap_state.N)
cmap_time = 'viridis'
cmap_cells = scm.lipari

phase_basin_cmap_name = 'Pastel1'

def lighten_colors(colors, amount=0.4):
    white = np.ones(3, dtype=float)
    lightened_colors = []
    for color in colors:
        rgb = np.asarray(to_rgb(color), dtype=float)
        lightened_rgb = (1.0 - amount) * rgb + amount * white
        lightened_colors.append(tuple(np.clip(lightened_rgb, 0.0, 1.0)))
    return tuple(lightened_colors)

pastel_order_colors = lighten_colors(order_colors, amount=0.4)


def set_order_colors(colors, lighten_amount=0.4):
    global order_colors, cmap_state, norm_state, pastel_order_colors

    order_colors = tuple(colors)
    cmap_colors = order_colors if len(order_colors) > 0 else (neutral_color,)
    cmap_state = ListedColormap(cmap_colors)
    norm_state = BoundaryNorm(np.arange(len(cmap_colors) + 1) - 0.5, cmap_state.N)
    pastel_order_colors = lighten_colors(order_colors, amount=lighten_amount)


def phase_basin_palette(n_colors, unresolved_color=unresolved_basin_color):
    base = plt.get_cmap(phase_basin_cmap_name)
    colors = [unresolved_color]
    for idx in range(n_colors):
        colors.append(base(idx % base.N))
    return ListedColormap(colors)


potential_surface_cmap = scm.cork.reversed()
rotational_surface_cmap = 'RdBu_r'
velocity_cmap = 'Greys'

cycle_line_lightness_floor = 0.2
cycle_line_lightness_scale = 0.8
cycle_line_saturation_scale = 1.1
