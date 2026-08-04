import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from .module_gaussian import *
from matplotlib.lines import Line2D



def make_derivative_4th_order(data):
    """Function that calculate the numerical derivative in 4th order. For the points in the middle we take the central difference 
    4th order. And we treat the boundaries with forward and backward difference 4th order. The data is assume to be of shape (coord, time, space),
    it will be changed later.

    Parameters
    ----------
    data : The data

    Returns
    -------
        The derivative of the data.
    """
    # Vector from central difference 4th order and the boundaries also have 4th order with forward and backward difference
    vector = (-np.roll(data, -2, axis=1) + 8 * np.roll(data, -1, axis=1) - 8 * np.roll(data, 1, axis=1) + np.roll(data, 2, axis=1))/(12)
    vector[:, -1, :] = (25 * data[:, -1, :] - 48 * data[:, -2, :] + 36 * data[:, -3, :] - 16 * data[:, -4, :] + 3 * data[:, -5, :])/(12)
    vector[:, -2, :] = (3 * data[:, -1, :] + 10 * data[:, -2, :] - 18 * data[:, -3, :] + 6 * data[:, -4, :] - data[:, -5, :])/(12)

    vector[:, 0, :] = (-25 * data[:, 0, :] + 48 * data[:, 1, :] - 36 * data[:, 2, :] + 16 * data[:, 3, :] - 3 * data[:, 4, :])/(12)
    vector[:, 1, :] = (-3 * data[:, 0, :] - 10 * data[:, 1, :] + 18 * data[:, 2, :] - 6 * data[:, 3, :] + data[:, 4, :])/(12)

    return vector

def extract_gene_data(file_path):
    """
    Extract the data from a .txt file

    Parameters
    ----------
    file_path : The path of the .txt. file

    Returns
    -------
        The data as a numpy array
    """
    return np.loadtxt(file_path)

def retrieve_data(simulated_data):
    """From a set of points in the context of the drosophila (simulated or not), retrieve the data from the latent space.
    
    Based on the encoder-decoder of https://doi.org/10.1073/pnas.2113651119. So this function is the decoder part.

    Parameters
    ----------
    simulated_data : Data that can be simulated or not

    Returns
    -------
        The 4 genes `Kr`, `Hb`, `Gt`, `Kni`
    """
    H1 = simulated_data[0]
    H2 = simulated_data[1]

    def reLU(x):
        return jnp.maximum(0,x)

    Kr = reLU(H1)
    Hb = reLU(H2)
    Gt = reLU(-H1)
    Kni = reLU(-H2)

    return Kr, Hb, Gt, Kni



def visu_latent_space(X, Y, limit):
    """Show the data in the latent space.

    Parameters 
    ----------
    X, Y : H1 and H2 in the context of the drosophila paper
    limit : limit in time to see up to a certain time

    Returns
    -------
        `fig` object of the plot
    """
    X = X[:limit, :]
    Y = Y[:limit, :]

    # Define color gradient along AP axis
    n_points = X.shape[1]
    t = np.linspace(0, 1, n_points)
    colors = cm.rainbow(t)

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Add all 291 manifolds
    for time in range(X.shape[0]):
        x = X[time]
        y = Y[time]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.3)
        ax.add_collection(lc)

    # Axis limits: based on all data
    max_val = max(np.max(np.abs(X)), np.max(np.abs(Y)))
    axis_limit = max_val * 1.6
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect('equal', adjustable='box')

    # Labels and title
    ax.set_xlabel("H1 (Kr - Gt)")
    ax.set_ylabel("H2 (Hb - Kni)")
    ax.set_title("All Manifolds")
    ax.grid(True)

    # Colorbar for % AP axis
    norm = mcolors.Normalize(vmin=0, vmax=n_points - 1)
    sm = cm.ScalarMappable(cmap='rainbow', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('% AP axis', rotation=270, labelpad=15)
    cbar.set_ticks([0, 186, 372, 558, 744, 930])
    cbar.set_ticklabels(['0', '20', '40', '60', '80', '100'])

    plt.tight_layout()
    return fig




def show_gene_evol(
    data1_real,
    data1_sim,
    data1_truncated,
    data2_real,
    data2_sim,
    data2_truncated,
    data3_real,
    data3_sim,
    data3_truncated,
    data4_real,
    data4_sim,
    data4_truncated,
    filename="animation.gif",
    interval=100,
    ylim=None
):
    """
    Temporal animation of the gene evolution.

    Parameters
    ----------
    data_real : Real data
    data_sim : Simulated data
    data_truncated : Truncated data (from the real data passed into the encoder)
    filename : Name of the file
    interval : Time between frams
    ylim : Limit in y axis
    """

    fig, ax = plt.subplots()

    line1_real, = ax.plot([], [], lw=1, color="#ff7f0e")
    line1_sim, = ax.plot([], [], lw=4, color="#ff7f0e", label="Gt")
    #line1_truncated, = ax.plot([], [], lw=2, color="#ff7f0e", linestyle= "--", label="truncated")

    line2_real, = ax.plot([], [], lw=1, color="#2ca02c")
    line2_sim, = ax.plot([], [], lw=4, color="#2ca02c", label="Kni")
    #line2_truncated, = ax.plot([], [], lw=2, color="#2ca02c", linestyle= "--", label="truncated")

    line3_real, = ax.plot([], [], lw=1, color="#d62728")
    line3_sim, = ax.plot([], [], lw=4, color="#d62728", label="Kr")
    #line3_truncated, = ax.plot([], [], lw=2, color="#d62728", linestyle= "--", label="truncated")

    line4_real, = ax.plot([], [], lw=1, color="#9467bd")
    line4_sim, = ax.plot([], [], lw=4, color="#9467bd", label="Hb")
    #line4_truncated, = ax.plot([], [], lw=2, color="#9467bd", linestyle= "--", label="truncated")

    blank_handle = ax.plot([], [], lw=1, color="lightsteelblue", label="True")
    shape_t, shape_x = data1_real.shape

    x = np.linspace(40, 100, shape_x)
    t = np.arange(shape_t)

    ax.set_xlim(np.min(x), np.max(x))
    if ylim is None:
        ymin = 0
        ymax = 1 
        """max(data1_real.max(), data1_sim.max(), data1_truncated.max(),
                   data2_real.max(), data2_sim.max(), data2_truncated.max(),
                   data3_real.max(), data3_sim.max(), data3_truncated.max(),
                   data4_real.max(), data4_sim.max(), data4_truncated.max())"""
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(*ylim)

    ax.set_xlabel("AP axis")
    ax.set_ylabel("Gene")
    ax.legend()
    title = ax.set_title("")

    def update(i):
        y_real1 = data1_real[i]
        y_sim1 = data1_sim[i]
        #y_truncated1 = data1_truncated[i]

        line1_real.set_data(x, y_real1)
        line1_sim.set_data(x, y_sim1)
        #line1_truncated.set_data(x, y_truncated1)

        y_real2 = data2_real[i]
        y_sim2 = data2_sim[i]
        #y_truncated2 = data2_truncated[i]

        line2_real.set_data(x, y_real2)
        line2_sim.set_data(x, y_sim2)
        #line2_truncated.set_data(x, y_truncated2)

        y_real3 = data3_real[i]
        y_sim3 = data3_sim[i]
        #y_truncated3 = data3_truncated[i]

        line3_real.set_data(x, y_real3)
        line3_sim.set_data(x, y_sim3)
        #line3_truncated.set_data(x, y_truncated3)

        y_real4 = data4_real[i]
        y_sim4 = data4_sim[i]
        #y_truncated4 = data4_truncated[i]

        line4_real.set_data(x, y_real4)
        line4_sim.set_data(x, y_sim4)
        #line4_truncated.set_data(x, y_truncated4)

        title.set_text(f"Time = {t[i]:.2f}")
        return line1_real, line1_sim, line2_real, line2_sim, line3_real, line3_sim, line4_real, line4_sim, title

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=interval,
        blit=True
    )

    writer = PillowWriter(fps=1000 // interval)
    anim.save(filename, writer=writer)

    plt.close(fig)

    return filename


@jax.jit
def gaussian_front(z1, z2, amp = 1, sigma = 1, X = 0, Y = 0):
    return amp * jnp.exp(- ( (z1 - X)**2 + (z2 - Y)**2 ) / (2*sigma**2) )

@jax.jit
def step_func(x):
    return 0.5 * (1 + x/jnp.sqrt((x**2)))

@jax.jit  
def Kr_evol(Kr, Hb, Gt, Kni, params):
    z1 = Kr - Gt
    z2 = Hb - Kni
    return step_func(z1) * (JAXflow_grad_mixture(z1, z2, params[0]) + JAXflow_rot_mixture(z1, z2, params[1]))[0]

@jax.jit
def Hb_evol(Kr, Hb, Gt, Kni, params):
    z1 = Kr - Gt
    z2 = Hb - Kni
    return step_func(z2) * (JAXflow_grad_mixture(z1, z2, params[0]) + JAXflow_rot_mixture(z1, z2, params[1]))[1]

@jax.jit
def Gt_evol(Kr, Hb, Gt, Kni, params):
    z1 = Kr - Gt
    z2 = Hb - Kni
    return -step_func(-z1) * (JAXflow_grad_mixture(z1, z2, params[0]) + JAXflow_rot_mixture(z1, z2, params[1]))[0]

@jax.jit
def Kni_evol(Kr, Hb, Gt, Kni, params):
    z1 = Kr - Gt
    z2 = Hb - Kni
    return -step_func(-z2) * (JAXflow_grad_mixture(z1, z2, params[0]) + JAXflow_rot_mixture(z1, z2, params[1]))[1]

