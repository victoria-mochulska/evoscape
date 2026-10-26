import jax.random as jrd
import jax.numpy as jnp
import jax.image as jimg
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import io
import numpy as np
import pandas as pd
from jax import jit, vmap
from evoscape.jax.converters import pytree_to_landscape
from copy import copy
import json
from pathlib import Path
import orbax.checkpoint as ocp
import pickle
import logging

from evoscape.landscapes import Landscape
from evoscape.modules import Node, UnstableNode, Center, NegCenter, AnisotropicNode, AnisotropicUnstableNode, AnisotropicCenter, AnisotropicNegCenter
from evoscape.jax.regimes import mr_const, mr_sigmoid, mr_const_aniso
from evoscape.jax.dynamics import state_probs, state_probs_aniso
from evoscape.jax.converters import landscape_to_params
from evoscape.jax.nath_utils import *
from evoscape.jax.flax_models.landscape_flax import LandscapeFlax, AnisotropicLandscapeFlax
from evoscape.jax.flax_models.autoencoder_flax import AutoEncoder
from flax import nnx


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
        data[0][x][t] = row["Gt_data"]
        data[1][x][t] = row["Kni_data"]
        data[2][x][t] = row["Hb_data"]
        data[3][x][t] = row["Kr_data"]
    
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



    

def compare_trajectories(traj_sim, traj_real, gene_names=None, title=None):
    """
    Compare deux trajectoires de taille (4, T) sur un seul graphe.

    Parameters
    ----------
    traj_sim : ndarray, shape (4, T)
        Trajectoire simulée.
    traj_real : ndarray, shape (4, T)
        Trajectoire réelle.
    gene_names : list of str, optional
        Nom des 4 gènes.
    title : str, optional
        Titre de la figure.
    """
    traj_sim = np.asarray(traj_sim)
    traj_real = np.asarray(traj_real)

    if traj_sim.shape != traj_real.shape:
        raise ValueError("Les deux tableaux doivent avoir la même taille.")
    if traj_sim.shape[0] != 4:
        raise ValueError("Les tableaux doivent être de taille (4, T).")

    if gene_names is None:
        gene_names = [f"Gène {i+1}" for i in range(4)]

    T = traj_sim.shape[1]
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i in range(4):
        color = colors[i % len(colors)]

        ax.plot(
            t, traj_sim[i],
            color=color,
            linewidth=2,
            linestyle="-",
            label=f"{gene_names[i]} (simulation)"
        )

        ax.plot(
            t, traj_real[i],
            color=color,
            linewidth=2,
            linestyle="--",
            label=f"{gene_names[i]} (réel)"
        )

    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend(ncol=2)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


"""
def plot_colored_trajectories(
    traj_2d,
    traj_4d,
    filename,
    cmap="Blues",
    point_size=10,
    linewidth=0.5,
    alpha_line=0.3,
    figsize=(12, 12),
):
    """
"""
    Affiche les trajectoires 2D colorées par chacune des 4 coordonnées
    du tableau décodé.

    Parameters
    ----------
    traj_2d : ndarray, shape (2, N, T)
    traj_4d : ndarray, shape (4, N, T)
    """
"""

    gene_names = ["Gt", "Kni", "Hb", "Kr"]

    assert traj_2d.shape[0] == 2
    assert traj_4d.shape[0] == 4
    assert traj_2d.shape[1:] == traj_4d.shape[1:]

    n_particles, T = traj_2d.shape[1:]

    x = traj_2d[0]
    y = traj_2d[1]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()


    maximum_x = jnp.max(traj_2d[0])

    maximum_y = jnp.max(traj_2d[1])


    minimum_x = np.min(traj_2d[0,:,:])
    maximum_x = np.max(traj_2d[0,:,:])
    minimum_y = np.min(traj_2d[1,:,:])
    maximum_y = np.max(traj_2d[1,:,:])

    maximum = np.max([maximum_x, maximum_y])
    minimum = np.min([minimum_x, minimum_y])
    delta = maximum - minimum

    margin = 0.02 * delta if delta > 0 else 1.0

    for dim in range(4):

        values = traj_4d[dim]
        vmin = values.min()
        vmax = values.max()

        ax = axes[dim]

        for i in range(n_particles):

            # Trajectoire en gris
            ax.plot(
                x[i],
                y[i],
                color="lightgray",
                linewidth=linewidth,
                alpha=alpha_line,
                zorder=1,
            )

            # Points colorés
            sc = ax.scatter(
                x[i],
                y[i],
                c=values[i],
                cmap=cmap,
                s=point_size,
                vmin=vmin,
                vmax=vmax,
                zorder=2,
                rasterized=True,
            )

        ax.set_xlim(minimum - margin, maximum + margin)
        ax.set_ylim(minimum - margin, maximum + margin)

        ax.set_aspect("equal")
        ax.set_title(gene_names[dim])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    #plt.savefig(filename)
    return fig
"""


class KeySource:
    """nnx.Rngs-like key provider, implemented as a plain python object.

    nnx.Rngs keeps a traced RngCount that cannot be mutated from inside
    jax.vmap (TraceContextError). With noise = init_noise = 0 the key is
    never numerically used, so a stateless provider is equivalent here.
    """

    def __init__(self, key):
        self.key = key

    def default(self, *tags):
        return self.key

    def __getattr__(self, tag):
        return lambda: self.key


def make_autoencoder(run_key, nt, dims_encoder, dims_decoder, n_modules):
    """One candidate model: random landscape + fresh MLP weights."""
    params = random_init2(run_key, n_modules, (-3, 3), (-3, 3), -0.2, 0.2)

    # fixed module layout (random type selection is not used)
    """modules = [
        AnisotropicNode(x=params[0, 0], y=params[0, 1], a=np.array(params[0, 2]),
                        sx=np.array(params[0, 3]), sy=np.array(params[0, 4]),
                        th=np.array(params[0, 5]), tau=1.),
        AnisotropicCenter(x=params[1, 0], y=params[1, 1], a=np.array(params[1, 2]),
                          sx=np.array(params[1, 3]), sy=np.array(params[1, 4]),
                          th=np.array(params[1, 5]), tau=1.),
    ]"""
    subkeys = jax.random.split(run_key, n_modules)
    modules = []
    for i in range(n_modules):
        numb = jax.random.uniform(subkeys[i])
        if numb < 0.25:
            modules.append(AnisotropicNode(x=params[i, 0], y=params[i, 1], a=np.array(params[i, 2]), sx=np.array(params[i, 3]), sy=np.array(params[i, 4]), th=np.array(params[i, 5]), tau=1.))
        elif numb < 0.50:
            modules.append(AnisotropicUnstableNode(x=params[i, 0], y=params[i, 1], a=np.array(params[i, 2]), sx=np.array(params[i, 3]), sy=np.array(params[i, 4]), th=np.array(params[i, 5]), tau=1.))
        elif numb < 0.75:
            modules.append(AnisotropicCenter(x=params[i, 0], y=params[i, 1], a=np.array(params[i, 2]), sx=np.array(params[i, 3]), sy=np.array(params[i, 4]), th=np.array(params[i, 5]), tau=1.))
        elif numb < 1:
            modules.append(AnisotropicNegCenter(x=params[i, 0], y=params[i, 1], a=np.array(params[i, 2]), sx=np.array(params[i, 3]), sy=np.array(params[i, 4]), th=np.array(params[i, 5]), tau=1.))
    
    landscape = Landscape(module_list=modules, A0=0.00, init_cond=(0.0, 0.0),
                          regime=mr_const_aniso, n_regimes=1)

    landscape_flax = AnisotropicLandscapeFlax(landscape, KeySource(run_key))
    landscape_flax.set_simulation(init_noise=0., t0=0., tf=20., nt=nt, ndt=50, noise=0.)
    landscape_flax.set_regime_params(signal_param=None)
    landscape_flax.set_state_probs(state_probs_aniso)

    rngs = nnx.Rngs(run_key)  # consumed once, at MLP weight init
    return AutoEncoder(landscape_flax, dims_decoder=dims_decoder,
                       dims_encoder=dims_encoder, rngs=rngs)


def save_autoencoder2(autoencoder, path):

    # 1. Save all NNX state
    state = nnx.state(autoencoder, nnx.Param)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        path / "state",
        state,
    )
    checkpointer.wait_until_finished()
    # 2. Save architecture/configuration
    config = {
        "dims_encoder": autoencoder.encoder.dims,
        "dims_decoder": autoencoder.decoder.dims,
        "model_type": "AutoEncoder",
    }

    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Sauvegarde du landscape complet
    with open(path / f"autoencoder_landscape.pkl", "wb") as f:
        pickle.dump(autoencoder.landscape_flax, f)

def save_autoencoder(autoencoder, path):
    """
    Save an AutoEncoder and everything required to reconstruct it.

    Directory structure:
        path/
        ├── config.json
        ├── landscape.pkl
        └── state/
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Save trainable parameters only
    # ------------------------------------------------------------

    state = nnx.state(autoencoder, nnx.Param)

    checkpointer = ocp.StandardCheckpointer()

    checkpointer.save(
        path / "state",
        state,
    )

    # StandardCheckpointer saves asynchronously
    checkpointer.wait_until_finished()

    # ------------------------------------------------------------
    # 2. Save architecture
    # ------------------------------------------------------------

    config = {
        "model_type": "AutoEncoder",
        "dims_encoder": list(autoencoder.encoder.dims),
        "dims_decoder": list(autoencoder.decoder.dims),
    }

    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # ------------------------------------------------------------
    # 3. Save landscape
    # ------------------------------------------------------------

    with open(path / "landscape.pkl", "wb") as f:
        pickle.dump(autoencoder.landscape_flax, f)

def convert_numeric_keys(obj):
    if isinstance(obj, dict):
        return {
            int(k) if isinstance(k, str) and k.isdigit() else k:
                convert_numeric_keys(v)
            for k, v in obj.items()
        }

    return obj




def load_model2(path, rngs=None):
    """
    Load an AutoEncoder from a saved directory.

    Parameters
    ----------
    path : str or Path
        Directory containing config.json and state/.
    landscape : original landscape object
        Used to reconstruct LandscapeFlax.
    rngs : nnx.Rngs, optional
        RNGs used to instantiate the model.

    Returns
    -------
    AutoEncoder
        Fully reconstructed and trained model.
    """

    path = Path(path)

    # ------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------

    with open(path / "config.json", "r") as f:
        config = json.load(f)

    dims_encoder = config["dims_encoder"]
    dims_decoder = config["dims_decoder"]

    if rngs is None:
        rngs = nnx.Rngs(0)

    # ------------------------------------------------------------
    # 2. Reconstruct the LandscapeFlax
    # ------------------------------------------------------------

    with open(path / f"autoencoder_landscape.pkl", "rb") as f:
        landscape_flax = pickle.load(f)

    # ------------------------------------------------------------
    # 3. Reconstruct AutoEncoder architecture
    # ------------------------------------------------------------

    autoencoder = AutoEncoder(
        landscape_flax=landscape_flax,
        dims_decoder=dims_decoder,
        dims_encoder=dims_encoder,
        rngs=rngs,
    )

    # ------------------------------------------------------------
    # 4. Restore NNX parameters
    # ------------------------------------------------------------

    checkpointer = ocp.StandardCheckpointer()

    state = nnx.state(autoencoder, nnx.Param)


    restored_state = checkpointer.restore(
        path / "state",
    )
    checkpointer.wait_until_finished()
    restored_state = convert_numeric_keys(restored_state)
    nnx.update(
        autoencoder,
        restored_state,
    )

    return autoencoder


def load_model(path, rngs=None):
    """
    Load a fully reconstructed AutoEncoder.

    Parameters
    ----------
    path : str or Path
        Directory containing:
            config.json
            landscape.pkl
            state/

    rngs : nnx.Rngs, optional
        RNGs used to reconstruct the model.

    Returns
    -------
    AutoEncoder
        Reconstructed AutoEncoder with trained parameters restored.
    """

    path = Path(path)

    # ------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------

    with open(path / "config.json", "r") as f:
        config = json.load(f)

    if config["model_type"] != "AutoEncoder":
        raise ValueError(
            f"Unsupported model type: {config['model_type']}"
        )

    dims_encoder = config["dims_encoder"]
    dims_decoder = config["dims_decoder"]

    # ------------------------------------------------------------
    # 2. RNGs
    # ------------------------------------------------------------

    if rngs is None:
        rngs = nnx.Rngs(0)

    # ------------------------------------------------------------
    # 3. Load landscape
    # ------------------------------------------------------------

    with open(path / "landscape.pkl", "rb") as f:
        landscape_flax = pickle.load(f)

    # ------------------------------------------------------------
    # 4. Reconstruct model architecture
    # ------------------------------------------------------------

    autoencoder = AutoEncoder(
        landscape_flax=landscape_flax,
        dims_encoder=dims_encoder,
        dims_decoder=dims_decoder,
        rngs=rngs,
    )

    # ------------------------------------------------------------
    # 5. Create target parameter state
    # ------------------------------------------------------------

    state = nnx.state(autoencoder, nnx.Param)

    # ------------------------------------------------------------
    # 6. Restore parameters
    # ------------------------------------------------------------

    checkpointer = ocp.StandardCheckpointer()

    restored_state = checkpointer.restore(
        path / "state",
        state,
    )

    # ------------------------------------------------------------
    # 7. Update model
    # ------------------------------------------------------------

    nnx.update(
        autoencoder,
        restored_state,
    )

    return autoencoder


def landscape_summary(landscape_flax):
    """Trainable + static landscape params as plain python data (for JSON)."""
    landscape = landscape_flax.get_landscape_aniso()
    dynamic, static, Js = landscape_to_params(landscape)
    return {
        "dynamic": {"shape": list(np.asarray(dynamic).shape),
                    "values": np.asarray(dynamic).tolist()},
        "static": {"shape": list(np.asarray(static).shape),
                   "values": np.asarray(static).tolist()},
        "J": np.asarray(Js).tolist(),
        "A0": float(np.asarray(landscape.A0)),
        "init_cond": [float(v) for v in np.asarray(landscape.init_cond)],
        "n_regimes": int(landscape.n_regimes),
    }



def _key_name(prefix):
    parts = []
    for part in prefix:
        if hasattr(part, "name"):
            parts.append(str(part.name))
        elif hasattr(part, "key"):
            parts.append(str(part.key))
        elif hasattr(part, "index"):
            parts.append(str(part.index))
    return "_".join(parts)


def save_all_params(runset, path):
    """One .npz holding every run's params, landscape params and losses."""
    out = {}
    for prefix, leaf in jax.tree_util.tree_leaves_with_path(runset.params):
        out[_key_name(prefix)] = np.asarray(leaf)

    dyn = runset.params["landscape_flax"]["dynamic"]["module"]
    out["landscape_dynamic"] = np.stack(
        [np.asarray(dyn[k]) for k in ("x", "y", "a", "sx", "sy", "th")], axis=-1)
    out["landscape_J"] = np.asarray(runset.rest["landscape_flax"]["static"]["module"]["J"])
    out["losses"] = runset.losses
    out["init_losses"] = runset.init_losses
    np.savez(path, **out)



def analyze_best_run(i, runset, run_dir, data, xx, yy, gene_names, start_embryon_space, end_embryon_space):
    """Best-run-only analysis: PNGs + gifs written to the top-level run dir."""
    model = runset.model(i)
    ns, nt = data.shape[1], data.shape[2]
    traj_init = data[:, :, 0]

    trained_encoded_traj = model.encode_traj(data)
    trained_real_traj = model(traj_init)
    trained_traj_latent = model.get_latent_trajectory(traj_init)
    rms = float(RMS_points(data, trained_real_traj))

    logging.info("run %d (best) | RMS of trained traj: %.5f | final loss: %.4e",
                 i, rms, float(runset.losses[-1,-1, i]))

    show_init_latent(trained_encoded_traj, order=True,
                     filename=run_dir / "trained_init_pts_latent.png")

    if ns < 100:
        show_trajectory(np.swapaxes(trained_encoded_traj, 2, 1),
                        np.swapaxes(trained_traj_latent, 2, 1),
                        "trained trajectories", "Encoded data", "Simulated trajectories",
                        filename=run_dir / "trained_trajectories.gif")
    else:
        show_trajectory(np.swapaxes(trained_encoded_traj[:, ::int(100 / 10), :], 2, 1),
                        np.swapaxes(trained_traj_latent[:, ::int(100 / 10), :], 2, 1),
                        "trained trajectories", "Encoded data", "Simulated trajectories",
                        filename=run_dir / "trained_trajectories.gif")

    data_T = np.swapaxes(data, 1, 2)
    trained_real_traj_T = np.swapaxes(trained_real_traj, 1, 2)
    show_gene_evol(data_T[0], trained_real_traj_T[0], None,
                   data_T[1], trained_real_traj_T[1], None,
                   data_T[3], trained_real_traj_T[3], None,
                   data_T[2], trained_real_traj_T[2], None,
                   start=start_embryon_space, end=end_embryon_space,
                   filename=run_dir / "gene_evol_trained.gif")

    # genes in the latent space
    q_init_latent = trained_encoded_traj[:, :, 0]
    traj_2d, _states = model.landscape_flax(q_init_latent)
    fig = plot_colored_trajectories(traj_2d, trained_real_traj, gene_names)
    fig.savefig(run_dir / "genes_latent_space.png", dpi=150)
    plt.close(fig)

    return model, q_init_latent, trained_real_traj, rms