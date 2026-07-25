import jax.random as jrd
import jax.numpy as jnp
import jax.image as jimg
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import io
import numpy as np
import pandas as pd
from jax import jit, vmap
from evoscape.landscape_visuals import visualize_landscape
from evoscape.jax.converters import pytree_to_landscape
from copy import copy
from evoscape.landscape_visuals import visualize_landscape_t, mr_current_regime, cmap_state, norm_state
# Initialization function used in the fitness function, we let the model choose where the cells should start in optimization
def init_cell(key,n,init_cond,noise):                    
    key,subkey = jrd.split(key)
    return  key,init_cond+noise*jrd.normal(subkey, shape=(2, n))

def init_cell_circle(n, center, r):
    angles = jnp.linspace(0, 2*jnp.pi, n)

    x = jnp.cos(angles)*r + center[0]
    y = jnp.sin(angles)*r + center[1]

    return jnp.stack([x,y])

# to use for modules if colored by type
fp_type_colors = {
    'Node': 'tab:green',
    'UnstableNode': 'tab:blue',
    'Center': 'tab:purple',
    'NegCenter': 'hotpink',
}

# to use for modules if colored by order in the module_list
order_colors = (
    'indianred',
    'tab:orange',
    'gold',
    'tab:green',
    'tab:blue',
    'tab:purple',
    # 'm',
)
def visualize_landscape_adapted_to_points(landscape, xx, yy, regime,
                        color_scheme='fp_types',
                        draw_circles=True,
                        points=None,
                        point_color='red',
                        point_size=40,
                        point_marker='o'):
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
                sig = module.s.item()
                A = module.a.item()
            else:
                sig = module.s[regime]
                A = module.a[regime]

            if color_scheme == 'fp_types':
                color = fp_type_colors[module.__class__.__name__]
            elif color_scheme == 'order':
                color = order_colors[i]
            else:
                color = 'grey'

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
        for i in range(len(landscape.module_list)):
            circles_ax.add_patch(copy(circles[i]))

    stream_ax.streamplot(xx, yy, dX, dY, density=density, arrowsize=2., arrowstyle='->', linewidth=1,
                         color='grey')
    stream_ax.contour(xx, yy, dX, (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
    stream_ax.contour(xx, yy, dY, (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

    stream_ax.set_xlim([np.min(xx), np.max(xx)])
    stream_ax.set_ylim([np.min(yy), np.max(yy)])
    stream_ax.set_xticks([])
    stream_ax.set_yticks([])
    landscape.morphogen_times = morphogen_times
    # plt.show()
    if points is not None:
        points = np.asarray(points)

        stream_ax.scatter(
            points[:, 0],
            points[:, 1],
            c=point_color,
            s=point_size,
            marker=point_marker,
            zorder=10
        )
    return fig

def make_movie_landscape(dynamics, static, xx, yy, n_frames,path, n_regime_chosen=0, points=None):
    frames = []
    indices = np.linspace(0, len(dynamics)-1, n_frames, dtype=int)
    for i in indices:
        dynamic = dynamics[i]
        if points is not None:
            pts = points[i]
        else:
            pts = None
        l = pytree_to_landscape(dynamic, static)
        fig = visualize_landscape_adapted_to_points(l, xx, yy, regime=n_regime_chosen, color_scheme='fp_types', points=pts)
        plt.close(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
    imageio.mimsave(path, frames, duration=10.)


def make_movie_discrete_with_traj(traj, states, landscape, xx, yy, labels, time_pars, n_cells, noise, init_cond=0,
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

@jit
def rescale(A, N, T):
    return vmap(lambda x: jimg.resize(x, (N, T), method="linear"))(A)

def get_drosophile_data(pathfile):
    df =pd.read_csv(pathfile, sep =",")
    #Hardcoded 
    data = np.zeros((4,930,291))
    for _, row in df.iterrows():
        t = int(row["time"])
        x = int(row["x"])
        data[0, x, t] = row["Gt_data"]
        data[1, x, t] = row["Kni_data"]
        data[2, x, t] = row["Hb_data"]
        data[3, x, t] = row["Kr_data"]
    
    return jnp.array(data)

def get_facs_data(pathfile_to_conditioned_facs_data):
    df =pd.read_csv(pathfile_to_conditioned_facs_data, sep =",")
    
    genes = ["TBX6", "BRA", "CDX2", "SOX2", "SOX1"]

    df = df.sort_values("timepoint").reset_index(drop=True)

    unique_timepoints = np.sort(df["timepoint"].unique())

    n_time = df["timepoint"].nunique()

    n_genes = 5

    values = df[genes].to_numpy()

    # (cells, time, genes)
    values = values.reshape(-1, n_time, n_genes)

    # (genes, cells, time)
    values = np.transpose(values, (2, 0, 1))

    x = jnp.array(values)

    return x, unique_timepoints



    

