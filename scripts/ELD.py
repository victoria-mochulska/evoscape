import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from evoscape.jax.flax_models.landscape_flax import LandscapeFlax
from evoscape.jax.flax_models.autoencoder_flax import AutoEncoder


import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
import sys
from evoscape.landscapes import Landscape
from evoscape.modules import Node, UnstableNode, Center, NegCenter
from evoscape.morphogen_regimes import mr_const, mr_sigmoid
from evoscape.landscape_visuals import plot_traj, visualize_landscape, visualize_landscape_t, visualize_potential
from evoscape.jax.dynamics import state_probs
from evoscape.jax.utils import get_drosophile_data, get_facs_data
from evoscape.jax.config import DATA_DIR, EVOSCAPE_DIR
from evoscape.jax.losses import mmd_traj, sinkhorn_traj
from evoscape.jax.nath_utils import *

from sklearn.mixture import GaussianMixture

plt.style.use("default")


RESULT_DIR = EVOSCAPE_DIR.parent.parent / "results"




############ Creating Landscape #############

# nombre de module
n_modules = 10

# position des modules
xx = [jnp.cos(2*jnp.pi*k/n_modules)*3 for k in range(n_modules)] 
yy = [jnp.sin(2*jnp.pi*k/n_modules)*3 for k in range(n_modules)]

modules = [Node(x=x, y=y, a=np.array([0.2]), s=np.array([1.0]), tau=1.) for x,y in zip(xx, yy)]

landscape = Landscape(
    module_list=modules,
    A0=0.05,
    init_cond=(0.0, 0.0),
    regime=mr_const,
    n_regimes=1,
    # morphogen_times=(50.,),
)

# landscape node (au début)
print(landscape)


L = 3.
npoints = 201
q = np.linspace(-L, L, npoints)
xx, yy = np.meshgrid(q, q, indexing='xy')

## fig = visualize_landscape_t(landscape, xx, yy, 2., color_scheme='fp_types', traj_times=(0., 20., 201), traj_init_cond=(2.,2.), traj_start=100)
fig = visualize_landscape(landscape, xx, yy, regime=0, color_scheme='fp_types')
plt.savefig(RESULT_DIR / "streamplot_landscape.png")
plt.close(fig)
fig = visualize_potential(landscape, xx, yy, regime=0, color_scheme="fp_types")
plt.savefig(RESULT_DIR / "potential_landscape.png")
plt.close(fig)




# Initializing the landscape 
rngs = nnx.Rngs(0)
landscape_flax = LandscapeFlax(landscape, rngs)

init_noise = 0.
t0 = 0.
tf = 20.
nt = 17
ndt = 50
noise = 0.

landscape_flax.set_simulation(init_noise=init_noise, t0=t0, tf=tf, nt=nt, ndt=ndt, noise=noise)
landscape_flax.set_regime_params(signal_param=None)
landscape_flax.set_state_probs(state_probs)

# Initialiazing the decoder
dims_encoder = [4, 8, 4, 2]
dims_decoder = [2, 8, 8, 4]

autoencoder = AutoEncoder(landscape_flax, dims_decoder=dims_decoder, dims_encoder=dims_encoder, rngs=rngs)



filepath = DATA_DIR / "data_drosophiles" / "drosophiles_cleaned.csv"


target_traj_init = get_drosophile_data(filepath) 
print(target_traj_init.shape)




target_traj = target_traj_init[:,0:770,:170:10]

print(target_traj.shape)



encoded_traj = autoencoder.encode_traj(target_traj)
encoded_traj_init = encoded_traj[:, :, 0]
q_init = target_traj[:, :,0]
print(q_init.shape)

simulated_traj = autoencoder(q_init)
print(simulated_traj.shape)

loss = mmd_traj(simulated_traj, target_traj)
print(loss)




print(q_init.shape)
q_init_latent = nnx.vmap(lambda x : autoencoder.encoder(x), in_axes=1, out_axes=1)(q_init) # (2, n)

print(q_init_latent.shape)
plt.scatter(q_init_latent[0,:], q_init_latent[1,:])

print(RMS_points(encoded_traj_init, q_init_latent))

show_init_latent(encoded_traj, order=False, filename=RESULT_DIR / "init_pts_latent.png")
sys.exit("Stopping the script here.") 



########## TRAINING ###########



# Defining the optimizer 
tx = optax.adamw(
    learning_rate = 2e-2,
    weight_decay = 2e-3,
)

optimizer = nnx.Optimizer(autoencoder, tx, wrt=nnx.Param)

# Defining the loss function, and the train_step 

def loss_fn(autoencoder, target_traj):
    # Forward pass
    q_init = target_traj[:,:,0]
    simulated_traj = autoencoder(q_init)

    # Loss on cloud points
    # this loss is useful when we don't have the trajectories. But for the drosophila we have it.
    # loss_dynamics = mmd_traj(simulated_traj, target_traj)
    loss_dynamics = jnp.mean((simulated_traj - target_traj)**2)

    # Here the loss encoding is only on the initial condition, and in reality it is performed in the loss dynamics
    # because the first term of (simulated_traj - target_traj)**2 is (q_init - decoder(encoder(q_init)))

    return loss_dynamics 


@nnx.jit
def train_step(autoencoder, optimizer: nnx.Optimizer, target_traj):
    # nnx.value_and_grad computes the gradient with respect to the nnx.Params
    loss, grads = nnx.value_and_grad(loss_fn)(autoencoder, target_traj)
    
    optimizer.update(autoencoder, grads)
    
    return loss

# Training loop 
n_epochs = 500
loss_vals = []

for epoch in range(n_epochs):
    # autoencoder.landscape_flax.sim_params["noise"] = 0.1 * (jnp.exp(-jnp.abs(epoch-20)/10))
    loss_val = train_step(autoencoder, optimizer, target_traj)
    
    if epoch%10 == 0:
        print(f"Époque {epoch} | Loss: {loss_val:.4f}")

    loss_vals.append(loss_val)

plt.plot(loss_vals)



landscape = autoencoder.landscape_flax.get_landscape()

print(landscape)

L = 3.
npoints = 201
q = np.linspace(-L, L, npoints)
xx, yy = np.meshgrid(q, q, indexing='xy')

fig = visualize_landscape_t(landscape, xx, yy, 20., color_scheme='fp_types', traj_times=(0., 20., 201), traj_init_cond=(2.,2.), traj_start=100)






####### ANALYZING ##########



print(q_init.shape)
q_init_latent = nnx.vmap(lambda x : autoencoder.encoder(x), in_axes=1, out_axes=1)(q_init) # (2, n)

print(q_init_latent.shape)
plt.scatter(q_init_latent[0,:], q_init_latent[1,:])



traj, states = autoencoder.landscape_flax(q_init_latent)

print(traj.shape)

fig = plot_traj(traj, states, t0, tf, nt, L=4)



fig = plot_traj(target_traj, states, t0, tf, nt, L=4)
encoded_traj = autoencoder.encode_traj(target_traj)
print("encoded traj", encoded_traj.shape)
print("target traj in real space", target_traj.shape)
print("traj from model in latent space",traj.shape)
fig = plot_traj(encoded_traj, states, t0, tf, nt, L=4)
fig = plot_traj(traj, states, t0, tf, nt, L=4)



import numpy as np
import matplotlib.pyplot as plt


def plot_colored_trajectories(
    traj_2d,
    traj_4d,
    cmap="Blues",
    point_size=10,
    linewidth=0.5,
    alpha_line=0.3,
    figsize=(12, 12),
):
    """
    Affiche les trajectoires 2D colorées par chacune des 4 coordonnées
    du tableau décodé.

    Parameters
    ----------
    traj_2d : ndarray, shape (2, N, T)
    traj_4d : ndarray, shape (4, N, T)
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
            )

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(gene_names[dim])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.show()





traj_4d = autoencoder(q_init)

q_init_latent = nnx.vmap(lambda x : autoencoder.encoder(x), in_axes=1, out_axes=1)(q_init) # (2, n)
traj_2d, states = autoencoder.landscape_flax(q_init_latent)

print(traj_4d.shape)
print(traj_2d.shape)

plot_colored_trajectories(traj_2d, traj_4d)






import numpy as np
import matplotlib.pyplot as plt

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



sim_traj_4d = autoencoder(q_init)
print(sim_traj_4d.shape)

time_idx = 0

real_traj = target_traj[:, :, time_idx]
sim_traj = sim_traj_4d[:, :, time_idx]

print(real_traj.shape)
print(sim_traj.shape)

compare_trajectories(
    sim_traj,
    real_traj,
    gene_names=["Gt", "Kni", "Hb", "Kr"],
    title="Trajectories comparison"
)




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def make_gene_gif(
    sim,
    real=None,
    x=None,
    gene_names=None,
    fps=5,
    filename="gene_evolution.gif",
):
    """
    Génère un GIF montrant l'évolution temporelle des concentrations.

    Parameters
    ----------
    sim : ndarray (n_genes, n_space, n_time)
        Données simulées.
    real : ndarray (n_genes, n_space, n_time), optional
        Données réelles.
    x : ndarray (n_space,), optional
        Coordonnées spatiales.
    gene_names : list of str, optional
        Noms des gènes.
    fps : int
        Images par seconde.
    filename : str
        Nom du fichier gif.
    """

    sim = np.asarray(sim)

    if real is not None:
        real = np.asarray(real)
        if sim.shape != real.shape:
            raise ValueError("sim et real doivent avoir la même taille.")

    n_genes, n_space, n_time = sim.shape

    if x is None:
        x = np.arange(n_space)

    if gene_names is None:
        gene_names = [f"Gène {i+1}" for i in range(n_genes)]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8, 5))

    sim_lines = []
    real_lines = []

    # bornes fixes pour éviter que l'échelle change
    ymin = sim.min()
    ymax = sim.max()

    if real is not None:
        ymin = min(ymin, real.min())
        ymax = max(ymax, real.max())

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Coordonnée spatiale")
    ax.set_ylabel("Concentration")

    for g in range(n_genes):
        c = colors[g % len(colors)]

        line_sim, = ax.plot(
            x,
            sim[g, :, 0],
            color=c,
            lw=2,
            label=f"{gene_names[g]} (simulation)"
        )
        sim_lines.append(line_sim)

        if real is not None:
            line_real, = ax.plot(
                x,
                real[g, :, 0],
                "--",
                color=c,
                lw=2,
                label=f"{gene_names[g]} (réel)"
            )
            real_lines.append(line_real)

    title = ax.set_title("")

    ax.legend(ncol=2)

    def update(frame):

        for g in range(n_genes):
            sim_lines[g].set_ydata(sim[g, :, frame])

            if real is not None:
                real_lines[g].set_ydata(real[g, :, frame])

        title.set_text(f"Timepoint {frame}")

        artists = sim_lines.copy()
        if real is not None:
            artists += real_lines
        artists.append(title)

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=n_time,
        interval=1000 // fps,
        blit=True,
    )

    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"GIF enregistré dans {filename}")



