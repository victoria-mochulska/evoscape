# making a loss for trajectories instead of vectors (still need to be worked on)
# making batch of all combination of number of gaussian
# better vector from data with central difference, forward, backward 4th order 


GPU = True
CPU = False


import jax
import jax.numpy as jnp

from evoscape.jax.nath_utils.utilities import *

import time
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from functools import partial
import jax.lax as lax
from itertools import combinations



hihi=27

if GPU:
    gpu = 2
elif CPU:
    cpu = 20

times = 50

seed_number = int(time.time_ns())

jax.config.update("jax_enable_x64", False)
if CPU:
    jax.config.update('jax_num_cpu_devices', cpu)

mult = False


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.interpolate import griddata
import tqdm



def extract_gene_data(file_path):
    return np.loadtxt(file_path)

def retrieve_data(simulated_data):
    H1 = simulated_data[0]
    H2 = simulated_data[1]

    def reLU(x):
        return jnp.maximum(0,x)

    Kr = reLU(H1)
    Hb = reLU(H2)
    Gt = reLU(-H1)
    Kni = reLU(-H2)

    return Kr, Hb, Gt, Kni



# Load gene data
Gt_data = extract_gene_data("gene_data/Gt_3002.txt")
Kni_data = extract_gene_data("gene_data/Kni_3002.txt")
Hb_data = extract_gene_data("gene_data/Hb_3002.txt")
Kr_data = extract_gene_data("gene_data/Kr_3002.txt")

# Compute differences
H1 = Kr_data - Gt_data
H2 = Hb_data - Kni_data

def visu_latent_space(X, Y):
    limit = int(X.shape[0] * 0.60)

    X = X[:limit, :]
    Y = Y[:limit, :]

    print(X.shape)
    print(Y.shape)

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



def show_trajectory(trajectory):
    x_vals = trajectory[:,0]
    y_vals = trajectory[:,1]

    line, = plt.plot([], [], 'b-', lw=2)
    point, = plt.plot([], [], 'ro')

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        line.set_data(x_vals[:frame], y_vals[:frame])
        point.set_data([x_vals[frame-1]], [y_vals[frame-1]])
        return line, point

    ani = animation.FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=True, interval=0.01)

    plt.show()

#visu_latent_space(H1, H2)
limit_time = int(H1.shape[0] * 0.56)
limit_space = int(H1.shape[1]*0.40)

H1 = H1[:limit_time, limit_space:]
H2 = H2[:limit_time, limit_space:]
Gt_data = Gt_data[:limit_time, limit_space:]
Kni_data = Kni_data[:limit_time, limit_space:]
Hb_data = Hb_data[:limit_time, limit_space:]
Kr_data = Kr_data[:limit_time, limit_space:]


all_coords = np.array([H1, H2])


# Vector from central difference 4th order and the boundaries also have 4th order with forward and backward difference
vect_all = (-np.roll(all_coords, -2, axis=1) + 8 * np.roll(all_coords, -1, axis=1) - 8 * np.roll(all_coords, 1, axis=1) + np.roll(all_coords, 2, axis=1))/(12)
vect_all[:, -1, :] = (25 * all_coords[:, -1, :] - 48 * all_coords[:, -2, :] + 36 * all_coords[:, -3, :] - 16 * all_coords[:, -4, :] + 3 * all_coords[:, -5, :])/(12)
vect_all[:, -2, :] = (3 * all_coords[:, -1, :] + 10 * all_coords[:, -2, :] - 18 * all_coords[:, -3, :] + 6 * all_coords[:, -4, :] - all_coords[:, -5, :])/(12)

vect_all[:, 0, :] = (-25 * all_coords[:, 0, :] + 48 * all_coords[:, 1, :] - 36 * all_coords[:, 2, :] + 16 * all_coords[:, 3, :] - 3 * all_coords[:, 4, :])/(12)
vect_all[:, 1, :] = (-3 * all_coords[:, 0, :] - 10 * all_coords[:, 1, :] + 18 * all_coords[:, 2, :] - 6 * all_coords[:, 3, :] + all_coords[:, 4, :])/(12)

# coords: shape (2, 291, 930)
skip_time = 1  # skip in the time direction
taille = 50

X = all_coords[0,::skip_time,:].reshape(-1)
Y = all_coords[1,::skip_time,:].reshape(-1)
U = vect_all[0,::skip_time,:].reshape(-1)
V = vect_all[1,::skip_time,:].reshape(-1)



X_all = H1
Y_all = H2
U_all = vect_all[0,:,:]
V_all = vect_all[1,:,:]


percentage_time = 100
percentage_space = 10

skip_time = int(100/percentage_time)
skip_space = int(100/percentage_space)

vect_all_trimmed = vect_all[:,::skip_time, ::skip_space]

X_all_trimmed = H1[::skip_time,::skip_space]
Y_all_trimmed = H2[::skip_time,::skip_space]
U_all_trimmed = vect_all[0,::skip_time,::skip_space]
V_all_trimmed = vect_all[1,::skip_time,::skip_space]


@jax.jit
def sigmoid(x, amp, c, s):
    return amp/(jnp.exp(-c * (x - 2*s)))



@jax.jit
def JAXgaussian_function(X, Y, params):
    """
    Gaussian function with parameters for position, amplitude and size

    Parameters
    ----------
    X :
        X coordinate
    Y :
        Y coordinate
    params :
        Parameters for the gaussian, in order -> position X, position Y, amplitude, size1 and size2, theta
    """
    x0, y0, amp, size1, size2, theta = params
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    size1_squared = size1**2
    size2_squared = size2**2

    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    gaussian = amp * size1_squared * size2_squared * jnp.exp(-( (x_prime**2)/(2*size1_squared) + (y_prime**2)/(2*size2_squared)  ))

    return gaussian

@jax.jit
def JAXflow_grad(X, Y, params):
    x0, y0, amp, size1, size2, theta = params
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    size1_squared = size1**2
    size2_squared = size2**2
    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    front = amp * jnp.exp(-( (x_prime**2)/(2*size1_squared) + (y_prime**2)/(2*size2_squared)  ))

    return front * jnp.array([x_prime * size2_squared * cosine - y_prime * size1_squared * sine,
                               x_prime * size2_squared * sine + y_prime * size1_squared * cosine])

@jax.jit
def JAXflow_rot(X, Y, params):
    x0, y0, amp, size1, size2, theta = params
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    size1_squared = size1**2
    size2_squared = size2**2
    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    front = amp * jnp.exp(-( (x_prime**2)/(2*size1_squared) + (y_prime**2)/(2*size2_squared)  ))

    return -front * jnp.array([-x_prime * size2_squared * sine - y_prime * size1_squared * cosine,
                              x_prime * size2_squared * cosine - y_prime * size1_squared * sine])


@jax.jit
def JAXgaussian_mixture(X, Y, params):
    """
    Gaussian mixture function

    Parameters
    ----------
    X :
        X coordinate
    Y :
        Y coordinate
    params :
        Parameters for the gaussian, in order -> position X, position Y, amplitude, size1 and size2
    offset :
        Offset for the gaussian

    Returns
    -------
    The sum of the gaussians
    """

    gaussians = jax.vmap(JAXgaussian_function, in_axes = (None, None, 0))(X, Y, params)

    return jnp.sum(gaussians, axis=0)


@jax.jit
def JAXflow_grad_mixture(X, Y, params):
    grad_flows = jnp.sum(jax.vmap(JAXflow_grad, in_axes = (None, None, 0))(X, Y, params), axis=0)

    return grad_flows

@jax.jit
def JAXflow_rot_mixture(X, Y, params):
    rot_flows = jnp.sum(jax.vmap(JAXflow_rot, in_axes = (None, None, 0))(X, Y, params), axis=0)

    return rot_flows 


def make_trajectories(params, data, ndt=50):
    nt = len(data[0,:,0])
    Delta_t = (nt-1)/nt
    dt = Delta_t / ndt
    print(f"dt={dt}")

    input = data[:,0,:] # shape (2,N)

    # donne (2,N)
    def f(pos):
        pos_x = pos[0]
        pos_y = pos[1]
        return JAXflow_grad_mixture(pos_x, pos_y, params[0]) + JAXflow_rot_mixture(pos_x, pos_y, params[1])

    def sum(pos, _):    
        pos = pos + f(pos) * dt
        return pos, pos # ('carryover', 'accumulated')

    initial_carry = input

    final_pos, traj = lax.scan(sum, initial_carry, None, length=nt*ndt - 1)

    traj = jnp.concatenate(
    [input[None, :, :], traj],
    axis=0
    )

    return final_pos, jnp.swapaxes(traj[::ndt,:,:], axis1=0, axis2=1)




@jax.jit
def cost_func2(params, X, Y, Z):
    _, prediction = make_trajectories(params, Z)

    return jnp.mean(JAXnorm(prediction - Z)**2)

@jax.jit
def loss_magnitude_func(fit, data):
    magnitude_vec_data = JAXnorm(data)
    magnitude_vec_fit = JAXnorm(fit)
    loss_magnitude = (magnitude_vec_fit - magnitude_vec_data)**2
    return jnp.mean(loss_magnitude)

@jax.jit
def loss_angle_func(fit, data):
    eps = 1e-12
    dot_product = jnp.sum(fit * data, axis=0)
    magnitude_vec_data = JAXnorm(data)
    magnitude_vec_fit = JAXnorm(fit)
    loss_angle = -1 * (dot_product / (magnitude_vec_fit * magnitude_vec_data + eps)) + 1
    return jnp.mean(loss_angle)

@jax.jit
def loss_sigma_func(params):
    sigmas = jnp.concatenate([params[0][:, 3:5].reshape(-1), params[1][:, 3:5].reshape(-1)])
    sigmas_squared = sigmas**2
    loss_sigmas = sigmoid(sigmas_squared, amp, sharpness, sigma_max)
    return jnp.sum(loss_sigmas)



@jax.jit 
def cost_func(params, X, Y, Z):

    eps = 1e-12

    
    prediction_vec = JAXflow_grad_mixture(X, Y, params[0]) + JAXflow_rot_mixture(X, Y, params[1])

    dot_product = jnp.sum(prediction_vec * Z, axis=0)
    magnitude_vec_Z = JAXnorm(Z)
    magnitude_vec = JAXnorm(prediction_vec)

    cos_theta = -1 * (dot_product / (magnitude_vec * magnitude_vec_Z + eps)) + 1

    loss_magnitude = (magnitude_vec - magnitude_vec_Z)**2
    loss_angle = cos_theta


    abc = jnp.mean(loss_magnitude)
    deg = jnp.mean(loss_angle)

    sigmas = jnp.concatenate([params[0][:, 3:5].reshape(-1), params[1][:, 3:5].reshape(-1)])

    sigmas_squared = sigmas**2

    loss_sigmas = sigmoid(sigmas_squared, amp, sharpness, sigma_max)

    mean_loss_sigmas = jnp.sum(loss_sigmas)

    return 0.2*(abc + deg) + 0.8*(mean_loss_sigmas)


grad_cost = jax.grad(cost_func, argnums=(0))

@jax.jit
def gradient_descent(X, Y, Z, lr, init_params_grad, init_params_rot):
    params = (init_params_grad, init_params_rot)

    def gradient_step(t, params):
        grad_params = grad_cost(params, X, Y, Z)

        grad_grad, grad_rot = grad_params  # dépaquetage tuple

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        params = (new_params_grad, new_params_rot)

        return params

    params = jax.lax.fori_loop(0, MAXITER, gradient_step, params)

    loss = cost_func(params, X, Y, Z)

    return loss, params[0], params[1]


@jax.jit
def gradient_descent_small(X, Y, Z, lr, init_params_grad, init_params_rot, iter):
    params = (init_params_grad, init_params_rot)

    def gradient_step(t, params):
        grad_params = grad_cost(params, X, Y, Z)

        grad_grad, grad_rot = grad_params  # dépaquetage tuple

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        params = (new_params_grad, new_params_rot)

        return params

    params = jax.lax.fori_loop(0, iter, gradient_step, params)

    loss = cost_func(params, X, Y, Z)

    return loss, params[0], params[1]



@jax.jit
def batch_gradient_descent(X, Y, Z, lr, params_grad_batch, params_rot_batch):
    def single_run(params_grad_batch, params_rot_batch):
        return gradient_descent(X, Y, Z, lr, params_grad_batch, params_rot_batch)
    return jax.vmap(single_run)(params_grad_batch, params_rot_batch)


@jax.jit
def batch_gradient_descent_small(X, Y, Z, lr, params_grad_batch, params_rot_batch, iter):
    def single_run_small(params_grad_batch, params_rot_batch):
        return gradient_descent_small(X, Y, Z, lr, params_grad_batch, params_rot_batch, iter)
    return jax.vmap(single_run_small)(params_grad_batch, params_rot_batch)


def train_verbose(X, Y, Z, lr, params_grad_batch, params_rot_batch, verbose_dt=500):
    pbar = tqdm.tqdm(total=MAXITER)
    evol = []

    for i in range(MAXITER//verbose_dt):
        losses, params_grad_batch, params_rot_batch = batch_gradient_descent_small(X, Y, Z, lr, params_grad_batch, params_rot_batch, verbose_dt)
        loss, _, _ = select_best(losses, params_grad_batch, params_rot_batch)
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(losses)

    return losses, params_grad_batch, params_rot_batch, evol

# Function for initializing random initial parameters
def random_init(key, N, X, Y, max, min):
    """
    key : jax.random.PRNGKey
    N   : number of gaussian
    X,Y : meshgrid (shape: HxW)
    Z   : potential to approximate(shape: HxW)
    """

    subkeys = jax.random.split(key, 6)

    # spatial domain
    x_min, x_max = jnp.min(X), jnp.max(X)
    y_min, y_max = jnp.min(Y), jnp.max(Y)

    # random initialization of gaussians (x0, y0)
    x0s = jax.random.uniform(subkeys[0], shape=(N,), minval=x_min, maxval=x_max)
    y0s = jax.random.uniform(subkeys[1], shape=(N,), minval=y_min, maxval=y_max)


    # initialization of amplitudes
    amps = jax.random.uniform(subkeys[2], shape=(N,), minval=min, maxval=max)


    # initialization of size
    size1 = jax.random.uniform(subkeys[3], shape=(N,), minval=0.001, maxval = 5)
    size2 = jax.random.uniform(subkeys[4], shape=(N,), minval=0.001, maxval = 5)

    # initialization of rotation
    theta = jax.random.uniform(subkeys[5], shape=(N,), minval=0, maxval = 2*jnp.pi)

    # stacking everything as (N, 4)
    params = jnp.stack([x0s, y0s, amps, size1, size2, theta], axis=1)

    return params





def select_best(losses, params_grad, params_rot):
    is_finite = jnp.isfinite(losses)
    if jnp.any(is_finite):
        filtered_losses = jnp.where(is_finite, losses, jnp.inf)
        idx = jnp.argmin(filtered_losses)

        if params_grad == None:
            return filtered_losses[idx], None, params_rot[idx]
        elif params_rot == None:
            return filtered_losses[idx], params_grad[idx], None
        else: 
            return filtered_losses[idx], params_grad[idx], params_rot[idx]
    else:
        return None, jnp.inf, jnp.inf
    

def select_best_first_run(losses):
    losses = jnp.array(losses)
    is_finite = jnp.isfinite(losses)
    if jnp.any(is_finite):
        filtered_losses = jnp.where(is_finite, losses, jnp.inf)
        
        idx = jnp.argmin(filtered_losses)

        return filtered_losses[idx], idx


def vector_difference(vec1, vec2):
    eps = 1e-12
    
    dot_product = jnp.sum(vec1 * vec2, axis=0)

    magnitude_vec1 = JAXnorm(vec1)
    magnitude_vec2 = JAXnorm(vec2)

    cos_theta = -1 * (dot_product / (magnitude_vec1 * magnitude_vec2 + eps)) + 1

    loss_magnitude = jnp.mean((magnitude_vec1 - magnitude_vec2)**2)
    loss_angle = jnp.mean(cos_theta)

    return loss_angle + loss_magnitude


def window_function(x,y):
    return  np.exp(-(x**2+y**2)/1) #np.exp(-(x**2+y**2)/0.04)


def RMS_points(data, simulated_data):
    return jnp.sqrt(jnp.mean((data - simulated_data)**2))

def RMS_vectors(data, simulated_data):
    return jnp.sqrt(jnp.mean(JAXnorm(data - simulated_data)**2))

@jax.jit
def JAXflow_grad(X, Y, params):
    x0, y0, amp, size1, size2, theta = params
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    size1_squared = size1**2
    size2_squared = size2**2
    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    front = amp * jnp.exp(-( (x_prime**2)/(2*size1_squared) + (y_prime**2)/(2*size2_squared)  ))

    return front * jnp.array([x_prime * size2_squared * cosine - y_prime * size1_squared * sine,
                               x_prime * size2_squared * sine + y_prime * size1_squared * cosine])

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




####### SINGLE BATCH ##########
max_val = max(np.max(np.abs(H1)), np.max(np.abs(H2)))

L = max_val
npoints = 200 + 1

q = np.linspace(-max_val, max_val, npoints)
xx, yy = np.meshgrid(q, q, indexing='xy')

dx = xx[0, 1] - xx[0, 0]  # change along x-axis (horizontal direction)
dy = yy[1, 0] - yy[0, 0]  # change along y-axis (vertical direction)

# 2. Interpoler sur la grille
grid_u = griddata((X, Y), U, (xx, yy), method='linear')
grid_v = griddata((X, Y), V, (xx, yy), method='linear')


vector = jnp.array([grid_u, grid_v])
mask = ~jnp.isnan(vector)


vector_interpolated = jnp.nan_to_num(vector, nan=0.0)

# SYSTÈME À DÉCOMPOSER
vector_field = vector_interpolated * window_function(xx, yy)
"""noise = np.random.normal(0, scale=0.001, size=vector_field.shape)

vector_field += noise"""


#vector_field = np.array(vector_field)
grad_pot_recomp, rot_pot_recomp, vector_field_recomp, _, _, _, _ = helmholtz_decomp(xx, yy, vector_field, error_display=False, streamplots=False, fields=False, potentials=False)



orig_gradient_vector = -1 * JAXgradFinite(grad_pot_recomp, dx, dy)
orig_rotation_vector = JAXskew_gradFinite(rot_pot_recomp, dx, dy)



sigma_max = -((6*max_val)**2)/(2*jnp.log(0.05))
amp = 1
sharpness = jnp.log(1/0.05 - 1)/(sigma_max)



##### Initializing batch of random initial conditions ##########

from jax.sharding import Mesh
from jax.experimental import mesh_utils

if GPU:
    devices = jax.devices("gpu")[:gpu]

    mesh = Mesh(
        np.array(devices),
        ('i',)
    )
if CPU:
    mesh_devices = mesh_utils.create_device_mesh((cpu,))  # 1D mesh of 50
    mesh = Mesh(mesh_devices, ('i',))  # nomme l'axe 'i'    


from jax.sharding import PartitionSpec, NamedSharding

spec = PartitionSpec('i', None, None)  # on shard l'axe 0, pas les autres
sharding = NamedSharding(mesh, spec)



surface_grad = grad_pot_recomp
max_grad  = jnp.max(surface_grad) * 1.20
min_grad = jnp.min(surface_grad) * 1.20

surface_rot = rot_pot_recomp
max_rot = jnp.max(surface_rot) * 1.20
min_rot = jnp.min(surface_rot) * 1.20

seed_number = int(time.time_ns())

#### VECTORIZED TEST DESCENT ####

if mult:
    MAXITER = 60000
    learning_rate = 0.001

    if CPU:
        nbr_init_cond = cpu*times
    else:
        nbr_init_cond = gpu*times

    start_time = time.time()
    conditions = []



    gaussians = 2
    gaussians_squared = (gaussians)*(gaussians)


    seed = jax.random.PRNGKey(seed_number)


    seed_split = jax.random.split(seed, gaussians_squared*2)
    seed_idx = 0

    loss_first_run = []


    for nbr_gaussians_grad in range(1, gaussians + 1):
        for nbr_gaussians_rot in range(1, gaussians + 1):
            seeds_grad = jax.random.split(seed_split[seed_idx], num=nbr_init_cond)
            seeds_rot = jax.random.split(seed_split[gaussians_squared + seed_idx], num=nbr_init_cond)

            if nbr_gaussians_grad == 0:
                init_conditions_grad_sharded = 0
            else:
                init_conditions_grad = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds_grad, nbr_gaussians_grad, xx, yy, max_grad, min_grad)
                init_conditions_grad_sharded = jax.device_put(init_conditions_grad, sharding)

            if nbr_gaussians_rot == 0:
                init_conditions_rot_sharded = 0 
            else:
                init_conditions_rot = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds_rot, nbr_gaussians_rot, xx, yy, max_rot, min_rot)
                init_conditions_rot_sharded = jax.device_put(init_conditions_rot, sharding)

            conditions.append([init_conditions_grad_sharded, init_conditions_rot_sharded])

            seed_idx += 1


    for i in range(len(conditions)):
        print(i)

        cond0_zero = jnp.all(conditions[i][0] == 0)
        cond1_zero = jnp.all(conditions[i][1] == 0)

        if cond0_zero & cond1_zero:
            print("No gaussians")
            best_loss = 1111111111
        elif cond1_zero:
            print("Grad only")
            losses, params_grad, params_rot, evol = train_verbose(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, learning_rate, conditions[i][0], conditions[i][1])
            best_loss, best_params_grad, best_params_rot = select_best(losses, params_grad, params_rot)
        elif cond0_zero:
            print("Rot only")
            losses, params_grad, params_rot, evol = train_verbose(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, learning_rate, conditions[i][0], conditions[i][1])
            best_loss, best_params_grad, best_params_rot = select_best(losses, params_grad, params_rot)
        else:
            print(f"Both ! With {conditions[i][0].shape[1]} gradient module and {conditions[i][1].shape[1]} rotational module")
            losses, params_grad, params_rot, evol = train_verbose(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, learning_rate, conditions[i][0], conditions[i][1])
            best_loss, best_params_grad, best_params_rot = select_best(losses, params_grad, params_rot)

        loss_first_run.append(best_loss)


    losses = jnp.array(loss_first_run)
    is_finite = jnp.isfinite(losses)
    filtered_losses = jnp.where(is_finite, losses, jnp.inf)

    size = int(np.sqrt(len(filtered_losses)))
    mat = filtered_losses.reshape(size, size)


    # Exclure le premier élément pour calcul min/max
    flat_np = mat.flatten()
    vmin = flat_np[1:].min()
    vmax = flat_np[1:].max()

    # Normalisation
    norm = (mat - vmin) / (vmax - vmin)

    def value_to_rgb(val):
        # Dégradé du vert au rouge (val proche 0 → vert, proche 1 → rouge)
        r = int(255 * val)
        g = int(255 * (1 - val))
        b = 80
        return r, g, b

    for row in norm:
        line = ""
        for val, raw in zip(row, row * (vmax - vmin) + vmin):
            r, g, b = value_to_rgb(val)
            line += f"\033[48;2;{r};{g};{b}m {raw: .2e} \033[0m "
        print(line)




    best_loss, idx = select_best_first_run(loss_first_run)

    print("Best loss : ", best_loss)

    nbr_gaussians_grad = idx // (gaussians) + 1
    nbr_gaussians_rot = idx % (gaussians) + 1

    print("The best is : ")
    print("Phi gaussians : ", nbr_gaussians_grad)
    print("Psy gaussians : ", nbr_gaussians_rot)
    automatic = "no" #input("Do you want to choose the number of gaussians ? Answer (yes/no) : ")


    if automatic == "yes":
        nbr_gaussians_grad = int(input("How many gaussians for Phi ? Answer : "))
        nbr_gaussians_rot = int(input("How many gaussians for Psy ? Answer : "))

else:
    nbr_gaussians_grad = 3
    nbr_gaussians_rot = 3


print(nbr_gaussians_grad)
print(nbr_gaussians_rot)

#### VECTORIZED REAL DESCENT ####
MAXITER = 100000
learning_rate = 0.001

if CPU:
    nbr_init_cond = cpu*times
else:
    nbr_init_cond = gpu*times

seed = jax.random.PRNGKey(seed_number + 9191)


seed_split = jax.random.split(seed, 2)



seeds = jax.random.split(seed_split[0], num=nbr_init_cond)

init_conditions_grad = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds, nbr_gaussians_grad, xx, yy, max_grad, min_grad)
init_conditions_grad_sharded = jax.device_put(init_conditions_grad, sharding)
#init_conditions_grad_sharded = jnp.array(init_conditions_grad) #random_init(seeds, nbr_gaussians_grad, xx, yy, max_grad, min_grad)



seeds = jax.random.split(seed_split[1], num=nbr_init_cond)

init_conditions_rot = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds, nbr_gaussians_rot, xx, yy, max_rot, min_rot)
init_conditions_rot_sharded = jax.device_put(init_conditions_rot, sharding)
#init_conditions_rot_sharded = jnp.array(init_conditions_rot) #random_init(seeds, nbr_gaussians_rot, xx, yy, max_rot, min_rot)


#init_conditions_grad_sharded = jnp.array([[[0.0692349374294281, 0.37596389651298523, 0.61648029088974, 0.09235212206840515, 0.18534064292907715, 3.43951678276062], [0.09924217313528061, -0.07121173292398453, 0.6452233791351318, 0.44482502341270447, 0.29551708698272705, 1.1948328018188477]]])
#init_conditions_rot_sharded = jnp.array([[[-0.1448238044977188, 0.40780404210090637, -0.37811076641082764, 0.37827208638191223, 0.39845556020736694, 0.8899526000022888], [-0.23616063594818115, -0.08094899356365204, 0.25505736470222473, 0.3816242814064026, 0.4422702491283417, 6.235337734222412]]])

"""start_time = time.time()
losses, params_grad, params_rot = batch_gradient_descent(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, learning_rate, init_conditions_grad_sharded, init_conditions_rot_sharded)
print(f"Time without verbose :{time.time() - start_time:.2f}")
"""


start_time = time.time()
losses, params_grad, params_rot, evol = train_verbose(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, learning_rate, init_conditions_grad_sharded, init_conditions_rot_sharded)
print(f"Time with verbose :{time.time() - start_time:.2f}")

evol = jnp.array(evol)



print("Losses shape:", losses.shape)
print("Params grad shape:", params_grad.shape)
print("Params rot shape:", params_rot.shape)



best_loss, best_params_grad, best_params_rot = select_best(losses, params_grad, params_rot)



print("Best parameters for grad : ", best_params_grad.tolist())
print("----------------------------")
print("Best parameters for rot : ", best_params_rot.tolist())
print("----------------------------")
print("Best loss : ", best_loss)
print("----------------------------")



elapsed_time = time.time() - start_time

print(f"Temps écoulé : {elapsed_time:.2f} secondes")
print("----------------------------")


params_grad = best_params_grad
params_rot = best_params_rot
params = (params_grad, params_rot)

"""params_grad = jnp.array([[-0.08038115501403809, -0.6236693263053894, -3.743567503988743e-06, 4.119783401489258, 4.475943565368652, 1.0419814586639404], [-0.27228614687919617, 0.36665821075439453, 0.00029243878088891506, 3.936072587966919, 2.8592631816864014, 4.348583698272705]])
params_rot = jnp.array([[0.4752938151359558, 0.46080368757247925, 0.00025895307771861553, 3.668438196182251, 2.9819300174713135, 5.138655185699463], [-0.5304754376411438, -0.16358672082424164, -0.0013111373409628868, 0.011060850694775581, 0.16150766611099243, 0.9657031893730164], [0.38596272468566895, -0.11766429245471954, 2.300215419381857e-05, 4.814855575561523, 1.6128801107406616, 4.564448356628418]])
params = (params_grad, params_rot)"""

best_params_grad = params_grad
best_params_rot = params_rot

fit_grad = JAXgaussian_mixture(xx, yy, params_grad)
fit_rot = JAXgaussian_mixture(xx, yy, params_rot)


gradient_vector = JAXflow_grad_mixture(xx, yy, params[0])
rotation_vector = JAXflow_rot_mixture(xx, yy, params[1])


vector_field_gaussians = gradient_vector + rotation_vector


final, traj = make_trajectories(params, all_coords, ndt=100)

Kr_simul, Hb_simul, Gt_simul, Kni_simul = retrieve_data(traj)
Kr_truncated, Hb_truncated, Gt_truncated, Kni_truncated = retrieve_data(all_coords)



print(f"Forme de la trajectoire recomposée {traj.shape}")
print(f"Forme des trajectoires des données {all_coords.shape}")


fitted_vect_field = JAXflow_grad_mixture(X_all, Y_all, params[0]) + JAXflow_rot_mixture(X_all, Y_all, params[1])
fitted_vect_field_trimmed = fitted_vect_field[:,::skip_time, ::skip_space]
loss_magnitude = loss_magnitude_func(fitted_vect_field, vect_all)
loss_angle = loss_angle_func(fitted_vect_field, vect_all)
loss_sigma = loss_sigma_func(params)
total_loss = cost_func(params, X_all, Y_all, vect_all)

sigmas = jnp.concatenate([params[0][:, 3:5].reshape(-1), params[1][:, 3:5].reshape(-1)])
loss_sigmas = sigmoid(sigmas, amp, sharpness, sigma_max)

loss_traj = jnp.mean(JAXnorm(traj - all_coords)**2)


RMS_pts = RMS_points(all_coords, traj)/(jnp.max(all_coords) - jnp.min(all_coords))
max_norm = jnp.max(JAXnorm(vect_all))
RMS_vec = RMS_vectors(vect_all, fitted_vect_field)/(max_norm)

print("Loss magnitude : ", loss_magnitude)
print("Loss angle : ", loss_angle)
print("Loss sigma : ", loss_sigma)
print("Loss : ", total_loss)
print("Loss trajectoire : ", loss_traj)
print()
print("RMS of simulated points", RMS_pts)
print("RMS of vector", RMS_vec)

if len(evol) > 1:
    loss_evol = np.arange(0, MAXITER, 500)
    evol = evol.T


#### make gifs ####
show_gene_evol(Kr_data, Kr_simul, Kr_truncated, "gene_evol_kr.gif")
show_gene_evol(Gt_data, Gt_simul, Gt_truncated, "gene_evol_gt.gif")
show_gene_evol(Hb_data, Hb_simul, Hb_truncated, "gene_evol_hb.gif")
show_gene_evol(Kni_data, Kni_simul, Kni_truncated, "gene_evol_kni.gif")


#### in real space ####
npoints = 200 + 1
max_val = max(np.max(Gt_data), np.max(Hb_data))
real_space_q = np.linspace(0, max_val, npoints)
GT_vec_pos, HB_vec_pos = np.meshgrid(real_space_q, real_space_q, indexing='xy')

GT_vec = Gt_evol(0, HB_vec_pos, GT_vec_pos, 0, params)
HB_vec = Hb_evol(0, HB_vec_pos, GT_vec_pos, 0, params)
vec_HB_GT = jnp.array([GT_vec, HB_vec])

GT_vec_pos, KR_vec_pos = np.meshgrid(real_space_q, real_space_q, indexing='xy')

GT_vec = Gt_evol(KR_vec_pos, 0, GT_vec_pos, 0, params)
KR_vec = Kr_evol(KR_vec_pos, 0, GT_vec_pos, 0, params)
vec_KR_GT = jnp.array([GT_vec, KR_vec])




with PdfPages("figures/my_plots_" + str(hihi) + "_" + str(1) + "_" + str(1) + ".pdf") as pdf:

   # --- First page: text summary ---
    fig, ax = plt.subplots(figsize=(8.5, 11))  # A4-ish
    ax.axis("off")  # no axes

    text_lines = [
        "Best parameters for grad :",
        str(best_params_grad),
        "",
        "",
        "Best parameters for rot :",
        str(best_params_rot),
        "",
        "",
        f"Loss magnitude: {loss_magnitude}",
        f"Loss angle: {loss_angle}",
        f"Loss sigma: {loss_sigma}",
        f"Total Loss: {total_loss}",
        f"Loss trajectoire : {loss_traj}",
        f"RMS of simulated points: {RMS_pts}",
        f"RMS of vector: {RMS_vec}",
        "Sigmas:",
        str(sigmas),
        "Loss sigmas:",
        str(loss_sigmas),
    ]

    # put lines in figure
    y = 1.0
    for line in text_lines:
        ax.text(0.05, y, line, ha="left", va="top", fontsize=10, family="monospace")
        y -= 0.05

    pdf.savefig(fig)
    plt.close(fig)

    show_fields(xx, yy, gradient_vector, rotation_vector, "Curl free vector field from fit", "Divergence free vector field from fit")
    pdf.savefig()
    plt.close()
    show_fields(xx, yy, orig_gradient_vector, orig_rotation_vector, "Original curl free vector field", "Original divergence free vector field")
    pdf.savefig()
    plt.close()
    show_fields(xx, yy, vector_field_gaussians, vector_interpolated, "Vector field from Gaussian", "Original vector field")
    pdf.savefig()
    plt.close()
    show_streamplots(xx, yy, vector_field_gaussians, vector_interpolated, "Streamplot from Gaussian", "Original streamplot")
    pdf.savefig()
    plt.close()

    show_field(GT_vec_pos, HB_vec_pos, vec_HB_GT, "Hb versus Gt vector field in real space based on gaussians", nbr_arrows=30)
    pdf.savefig()
    plt.close()

    show_streamplot(GT_vec_pos, HB_vec_pos, vec_HB_GT, "Hb versus Gt streamplot in real space based on gaussians")
    pdf.savefig()
    plt.close()

    show_field(GT_vec_pos, KR_vec_pos, vec_KR_GT, "Kr versus Gt vector field in real space based on gaussians", nbr_arrows=30, nullclines=False)
    pdf.savefig()
    plt.close()

    show_streamplot(GT_vec_pos, KR_vec_pos, vec_KR_GT, "Kr versus Gt streamplot in real space based on gaussians", nullclines=False)
    pdf.savefig()
    plt.close()

    show_fields(X_all, Y_all, fitted_vect_field, vect_all, "Fitted vector field", "Original vector field", nbr_arrows=50)
    pdf.savefig()
    plt.close()


    """error_on_vect_field = error(fitted_vect_field_trimmed, vect_all_trimmed)
    #print(round(np.max(error_on_vect_field), 2))

    show_error_display(X_all_trimmed, Y_all_trimmed, error_on_vect_field, "Error between original vector field and gaussians")
    pdf.savefig()
    plt.close()"""

    show_potentials(xx, yy, fit_grad, fit_rot, "Fitted curl free potential", "Fitted divergence free potential")
    pdf.savefig()
    plt.close()


    fig = show_trajectories(all_coords[:,::skip_time,::skip_space], traj[:, ::skip_time,::skip_space])
    pdf.savefig()
    plt.close(fig)
    

    """fig = plt.figure(figsize=(6,6), dpi=300)
    plt.scatter(all_coords[0,::skip_time,::skip_space], all_coords[1,::skip_time,::skip_space], s=0.05)
    axis_limit = 2
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)
    plt.axis('equal')
    pdf.savefig()
    plt.close(fig)

    fig = plt.figure(figsize=(6,6), dpi=300)
    plt.scatter(traj[0, ::skip_time,::skip_space], traj[1, ::skip_time,::skip_space], s=0.05)
    axis_limit = 2
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)
    plt.axis('equal')
    pdf.savefig()
    plt.close(fig)"""

    """fig = plt.figure(figsize=(6,6), dpi=300)
    plt.quiver(all_coords[0], all_coords[1], dx, dy)
    plt.axis('equal')
    pdf.savefig()
    plt.close(fig)"""

    """fig = visu_latent_space(traj[0], traj[1])
    pdf.savefig()
    plt.close(fig)
    fig = visu_latent_space(H1, H2)
    pdf.savefig()
    plt.close(fig)"""

    if len(evol) > 1:
        fig = plt.figure(figsize=(6,6), dpi=300)
        if CPU:
            for loss in range(times*cpu):
                plt.semilogy(loss_evol, evol[loss], color= "grey")
        else:
            for loss in range(times*gpu):
                plt.semilogy(loss_evol, evol[loss], color= "grey")

        pdf.savefig()
        plt.close(fig)
    
    show_potentials(xx, yy, grad_pot_recomp, rot_pot_recomp, "HD curl free potential", "HD divergence free potential")
    pdf.savefig()
    plt.close()
