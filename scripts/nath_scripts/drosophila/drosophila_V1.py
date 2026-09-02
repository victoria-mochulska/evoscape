# Trying out my old code from summer 2025, trying to get a hang of it.
# worked

GPU = False
CPU = True

hihi=8

if GPU:
    gpu = 1
elif CPU:
    cpu = 30
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
if CPU:
    jax.config.update('jax_num_cpu_devices', cpu)
from evoscape.jax.nath_utils.utilities import *


import time
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from functools import partial



import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.interpolate import griddata




def extract_gene_data(file_path):
    return np.loadtxt(file_path)

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
    plt.show()

#visu_latent_space(H1, H2)
limit_time = int(H1.shape[0] * 0.56)
limit_space = int(H1.shape[1]*0.40)

H1 = H1[:limit_time, limit_space:]
H2 = H2[:limit_time, limit_space:]



all_coords = np.array([H1, H2])

# coords: shape (2, 291, 930)
skip_time = 100  # skip in the time direction
taille = 50

# Choose time indices to sample
time_indices = np.arange(0, all_coords.shape[1] - 1, skip_time)  # e.g., [0, 10, 20, ...]

# Initialize lists
X_list, Y_list, U_list, V_list = [], [], [], []

for t in time_indices:
    x0 = all_coords[0, t, :]
    y0 = all_coords[1, t, :]
    dx = all_coords[0, t+1, :] - x0
    dy = all_coords[1, t+1, :] - y0

    X_list.append(x0)
    Y_list.append(y0)
    U_list.append(dx)
    V_list.append(dy)

# Stack into single arrays
X = np.concatenate(X_list)
Y = np.concatenate(Y_list)
U = np.concatenate(U_list)
V = np.concatenate(V_list)


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

    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    gaussian = amp * jnp.exp(-( (x_prime**2)/(2*size1**2) + (y_prime**2)/(2*size2**2)  ))

    return gaussian


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


skewgradient_operator = skewgradient_gaussians(JAXgaussian_mixture)
gradient_operator = gradient_gaussians(JAXgaussian_mixture)
@jax.jit 
def cost_func(params, X, Y, Z, mask):

    alpha = 20
    eps = 1e-12

    prediction_vec_grad = gradient_operator(X, Y, params[0])
    prediction_vec_rot = skewgradient_operator(X, Y, params[1])
    
    prediction_vec = prediction_vec_grad + prediction_vec_rot

    dot_product = jnp.sum(prediction_vec * Z, axis=0)
    magnitude_vec_Z = JAXnorm(Z)
    magnitude_vec = JAXnorm(prediction_vec)

    cos_theta = -1 * (dot_product / (magnitude_vec * magnitude_vec_Z + eps)) + 1

    loss_magnitude = (magnitude_vec - magnitude_vec_Z)**2
    loss_angle = cos_theta

    loss_magnitude_mask = loss_magnitude * mask
    loss_angle_mask = loss_angle * mask

    abc = jnp.sum(loss_magnitude_mask) / jnp.sum(mask)
    deg = jnp.sum(loss_angle_mask) / jnp.sum(mask)

    sigmas = jnp.concatenate([params[0][:, 3:5].reshape(-1), params[1][:, 3:5].reshape(-1)])

    sigmas_squared = sigmas**2
    #mean_sigmas = jnp.mean(sigmas)

    loss_sigmas = sigmoid(sigmas_squared, amp, sharpness, sigma_max)

    mean_loss_sigmas = jnp.sum(loss_sigmas)

    return abc + deg + mean_loss_sigmas


@jax.jit 
def cost_func_grad_only(params, X, Y, Z, mask):
    alpha = 20
    eps = 1e-12

    prediction_vec_grad = gradient_operator(X, Y, params)
    
    prediction_vec = prediction_vec_grad

    dot_product = jnp.sum(prediction_vec * Z, axis=0)
    magnitude_vec_Z = JAXnorm(Z)
    magnitude_vec = JAXnorm(prediction_vec)

    cos_theta = -1 * (dot_product / (magnitude_vec * magnitude_vec_Z + eps)) + 1

    loss_magnitude = jnp.tanh((magnitude_vec - magnitude_vec_Z)**2)
    loss_angle = cos_theta

    loss_magnitude_mask = loss_magnitude * mask
    loss_angle_mask = loss_angle * mask

    abc = jnp.sum(loss_magnitude_mask) / jnp.sum(mask)
    deg = jnp.sum(loss_angle_mask) / jnp.sum(mask)

    return abc + deg


@jax.jit 
def cost_func_rot_only(params, X, Y, Z, mask):
    alpha = 20
    eps = 1e-12

    prediction_vec_rot = skewgradient_operator(X, Y, params)
    
    prediction_vec = prediction_vec_rot

    dot_product = jnp.sum(prediction_vec * Z, axis=0)
    magnitude_vec_Z = JAXnorm(Z)
    magnitude_vec = JAXnorm(prediction_vec)

    cos_theta = -1 * (dot_product / (magnitude_vec * magnitude_vec_Z + eps)) + 1

    loss_magnitude = jnp.tanh((magnitude_vec - magnitude_vec_Z)**2)
    loss_angle = cos_theta

    loss_magnitude_mask = loss_magnitude * mask
    loss_angle_mask = loss_angle * mask

    abc = jnp.sum(loss_magnitude_mask) / jnp.sum(mask)
    deg = jnp.sum(loss_angle_mask) / jnp.sum(mask)

    return abc + deg


grad_cost = jax.grad(cost_func, argnums=(0))
grad_cost_grad_only = jax.grad(cost_func_grad_only, argnums=(0))
grad_cost_rot_only = jax.grad(cost_func_rot_only, argnums=(0))

def gradient_descent(X, Y, Z, mask, lr, init_params_grad, init_params_rot):
    params = (init_params_grad, init_params_rot)

    def gradient_step(t, params):
        grad_params = grad_cost(params, X, Y, Z, mask)

        grad_grad, grad_rot = grad_params  # dépaquetage tuple

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        params = (new_params_grad, new_params_rot)

        return params

    params = jax.lax.fori_loop(0, MAXITER, gradient_step, params)

    loss = cost_func(params, X, Y, Z, mask)

    return loss, params[0], params[1]


def gradient_descent_grad_only(X, Y, Z, mask, lr, init_params_grad):
    params = init_params_grad

    def gradient_step(t, params):
        grad_params = grad_cost_grad_only(params, X, Y, Z, mask)
        
        grad_params_norm = jnp.max(jnp.abs(grad_params), axis=-1, keepdims=True)
        
        params = params - lr * grad_params / (grad_params_norm + 1e-12)

        return params

    params = jax.lax.fori_loop(0, MAXITER, gradient_step, params)

    loss = cost_func_grad_only(params, X, Y, Z, mask)

    return loss, params, None


def gradient_descent_rot_only(X, Y, Z, mask, lr, init_params_rot):
    params = init_params_rot

    def gradient_step(t, params):
        grad_params = grad_cost_rot_only(params, X, Y, Z, mask)
        
        grad_params_norm = jnp.max(jnp.abs(grad_params), axis=-1, keepdims=True)
        
        params = params - lr * grad_params / (grad_params_norm + 1e-12)

        return params

    params = jax.lax.fori_loop(0, MAXITER, gradient_step, params)

    loss = cost_func_rot_only(params, X, Y, Z, mask)

    return loss, None, params


@jax.jit
def batch_gradient_descent(X, Y, Z, mask, lr, params_grad_batch, params_rot_batch):
    def single_run(params_grad_batch, params_rot_batch):
        return gradient_descent(X, Y, Z, mask, lr, params_grad_batch, params_rot_batch)
    return jax.vmap(single_run)(params_grad_batch, params_rot_batch)


@jax.jit
def batch_gradient_descent_grad_only(X, Y, Z, mask, lr, params_grad_batch):
    def single_run(params_grad_batch):
        return gradient_descent_grad_only(X, Y, Z, mask, lr, params_grad_batch)
    return jax.vmap(single_run)(params_grad_batch)


@jax.jit
def batch_gradient_descent_rot_only(X, Y, Z, mask, lr, params_rot_batch):
    def single_run(params_rot_batch):
        return gradient_descent_rot_only(X, Y, Z, mask, lr, params_rot_batch)
    return jax.vmap(single_run)(params_rot_batch)


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


def reduced_region(X, Y, value_x, value_y, gradient_vector, rotation_vector, grad_pot, rot_pot):

    # Masques indépendants pour X et Y
    indices_x = np.where((X[0] >= -value_x) & (X[0] <= value_x))[0]
    indices_y = np.where((Y[:, 0] >= -value_y) & (Y[:, 0] <= value_y))[0]

    # Masquage des champs vectoriels (JAX)
    JAX_indices_x = jnp.array(indices_x)
    JAX_indices_y = jnp.array(indices_y)

    # gradient_vector: [2, H, W]
    # Apply slicing sur les deux axes
    gradient_vector_mask = gradient_vector[:, JAX_indices_y[:, None], JAX_indices_x]
    rotation_vector_mask = rotation_vector[:, JAX_indices_y[:, None], JAX_indices_x]

    grad_pot_mask = grad_pot[JAX_indices_y[:, None], JAX_indices_x]
    rot_pot_mask = rot_pot[JAX_indices_y[:, None], JAX_indices_x]

    # Masquage des grilles X, Y (NumPy)
    X_mask = X[np.ix_(indices_y, indices_x)]
    Y_mask = Y[np.ix_(indices_y, indices_x)]

    return X_mask, Y_mask, gradient_vector_mask, rotation_vector_mask, grad_pot_mask, rot_pot_mask


def clip(X, Y, value_x, value_y, vector):

    # Masques indépendants pour X et Y
    indices_x = np.where((X[0] >= -value_x) & (X[0] <= value_x))[0]
    indices_y = np.where((Y[:, 0] >= -value_y) & (Y[:, 0] <= value_y))[0]

    # Masquage des champs vectoriels (JAX)
    JAX_indices_x = jnp.array(indices_x)
    JAX_indices_y = jnp.array(indices_y)


    # gradient_vector: [2, H, W]
    # Apply slicing sur les deux axes
    vector_mask = vector[:, JAX_indices_y[:, None], JAX_indices_x]

    return vector_mask


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




#show_potential(xx, yy, window_function(xx, yy), "Window function")


####### SINGLE BATCH ##########

npoints = 99 + 1
max_val = max(np.max(np.abs(H1)), np.max(np.abs(H2)))

q = np.linspace(-max_val, max_val, npoints)
xx, yy = np.meshgrid(q, q, indexing='xy')

dx = xx[0, 1] - xx[0, 0]  # change along x-axis (horizontal direction)
dy = yy[1, 0] - yy[0, 0]  # change along y-axis (vertical direction)

# 2. Interpoler sur la grille
grid_u = griddata((X, Y), U, (xx, yy), method='linear')
grid_v = griddata((X, Y), V, (xx, yy), method='linear')


vector = jnp.array([grid_u, grid_v])
mask = ~jnp.isnan(vector)
print(jnp.sum(mask))


vector = jnp.nan_to_num(vector, nan=0.0)

# SYSTÈME À DÉCOMPOSER
vector_field = vector * window_function(xx, yy)
"""noise = np.random.normal(0, scale=0.001, size=vector_field.shape)

vector_field += noise"""


#vector_field = np.array(vector_field)
grad_pot_recomp, rot_pot_recomp, vector_field_recomp, _, _, _, _ = helmholtz_decomp(xx, yy, vector_field, error_display=False, streamplots=False, fields=False, potentials=False)


plt.show()

orig_gradient_vector = -1 * JAXgradFinite(grad_pot_recomp, dx, dy)
orig_rotation_vector = JAXskew_gradFinite(rot_pot_recomp, dx, dy)

region_x = max_val
region_y = max_val
xx_mask, yy_mask, orig_gradient_vector_mask, orig_rotation_vector_mask, grad_pot_recomp_mask, rot_pot_recomp_mask = reduced_region(xx, yy, region_x, region_y, orig_gradient_vector, orig_rotation_vector, grad_pot_recomp, rot_pot_recomp)

vector_field_mask = clip(xx, yy, region_x, region_y, vector_field)

print(xx_mask.shape)
print(yy_mask.shape)


sigma_max = -((6*max_val)**2)/(2*jnp.log(0.05))
amp = 1
sharpness = jnp.log(1/0.05 - 1)/(sigma_max)

##### Initializing batch of random initial conditions ##########

from jax.sharding import Mesh
from jax.experimental import mesh_utils

if GPU:
    mesh_devices = mesh_utils.create_device_mesh((gpu,))  # 1D mesh of 50
    mesh = Mesh(mesh_devices, ('i',))  # nomme l'axe 'i'
if CPU:
    mesh_devices = mesh_utils.create_device_mesh((cpu,))  # 1D mesh of 50
    mesh = Mesh(mesh_devices, ('i',))  # nomme l'axe 'i'    


from jax.sharding import PartitionSpec, NamedSharding

spec = PartitionSpec('i', None, None)  # on shard l'axe 0, pas les autres
sharding = NamedSharding(mesh, spec)




seed_number = 299

#### VECTORIZED TEST DESCENT ####
MAXITER = 6000
learning_rate = 0.01

nbr_init_cond = 30

start_time = time.time()
conditions = []

surface_grad = grad_pot_recomp
max_grad  = jnp.max(surface_grad) * 1.20
min_grad = jnp.min(surface_grad) * 1.20

surface_rot = rot_pot_recomp
max_rot = jnp.max(surface_rot) * 1.20
min_rot = jnp.min(surface_rot) * 1.20


gaussians = 2 - 1
gaussians_squared = gaussians*gaussians


seed = jax.random.PRNGKey(seed_number)


seed_split = jax.random.split(seed, gaussians_squared*2)
seed_idx = 0

loss_first_run = []


for nbr_gaussians_grad in range(0, gaussians + 1):
    for nbr_gaussians_rot in range(0, gaussians + 1):
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


"""for i in range(len(conditions)):
    print(i)

    cond0_zero = jnp.all(conditions[i][0] == 0)
    cond1_zero = jnp.all(conditions[i][1] == 0)

    if cond0_zero & cond1_zero:
        print("No gaussians")
        best_loss = 1111111111
    elif cond1_zero:
        print("Grad only")
        losses, params_grad, params_rot = batch_gradient_descent_grad_only(xx_mask, yy_mask, vector_field_mask, mask, learning_rate, conditions[i][0])
        best_loss, best_params_grad, best_params_rot = select_best(losses, params_grad, params_rot)
    elif cond0_zero:
        print("Rot only")
        losses, params_grad, params_rot = batch_gradient_descent_rot_only(xx_mask, yy_mask, vector_field_mask, mask, learning_rate, conditions[i][1])
        best_loss, best_params_grad, best_params_rot = select_best(losses, params_grad, params_rot)
    else:
        print("Both !")
        losses, params_grad, params_rot = batch_gradient_descent(xx_mask, yy_mask, vector_field_mask, mask, learning_rate, conditions[i][0], conditions[i][1])
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
"""
nbr_gaussians_grad = 1 #idx // (gaussians + 1)
nbr_gaussians_rot = 1 #idx % (gaussians + 1)

print("The best is : ")
print("Phi gaussians : ", nbr_gaussians_grad)
print("Psy gaussians : ", nbr_gaussians_rot)
automatic = "no" #input("Do you want to choose the number of gaussians ? Answer (yes/no) : ")


if automatic == "yes":
    nbr_gaussians_grad = int(input("How many gaussians for Phi ? Answer : "))
    nbr_gaussians_rot = int(input("How many gaussians for Psy ? Answer : "))


print(nbr_gaussians_grad)
print(nbr_gaussians_rot)

#### VECTORIZED REAL DESCENT ####
MAXITER = 20000
learning_rate = 0.005

nbr_init_cond = 30


seed = jax.random.PRNGKey(seed_number + 9191)


seed_split = jax.random.split(seed, 2)


start_time = time.time()

seeds = jax.random.split(seed_split[0], num=nbr_init_cond)

init_conditions_grad = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds, nbr_gaussians_grad, xx, yy, max_grad, min_grad)
init_conditions_grad_sharded = jax.device_put(init_conditions_grad, sharding)


seeds = jax.random.split(seed_split[1], num=nbr_init_cond)

init_conditions_rot = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds, nbr_gaussians_rot, xx, yy, max_rot, min_rot)
init_conditions_rot_sharded = jax.device_put(init_conditions_rot, sharding)

# mettre les 3 options #########
losses, params_grad, params_rot = batch_gradient_descent(xx_mask, yy_mask, vector_field_mask, mask, learning_rate, init_conditions_grad_sharded, init_conditions_rot_sharded)



print("Losses shape:", losses.shape)
print("Params grad shape:", params_grad.shape)
print("Params rot shape:", params_rot.shape)


"""print(params_grad)
print("----------------------------")
print(params_rot)
print("----------------------------")"""
print(losses)
print("----------------------------")


best_loss, best_params_grad, best_params_rot = select_best(losses, params_grad, params_rot)



print("Best parameters for grad : ", best_params_grad)
print("----------------------------")
print("Best parameters for rot : ", best_params_rot)
print("----------------------------")
print("Best loss : ", best_loss)
print("----------------------------")



elapsed_time = time.time() - start_time

print(f"Temps écoulé : {elapsed_time:.2f} secondes")
print("----------------------------")


params_grad = best_params_grad
params_rot = best_params_rot
params = (params_grad, params_rot)



#### Nouvelle grid


L = max_val
npoints = 200 + 1
q = np.linspace(-L, L, npoints)
xx, yy = np.meshgrid(q, q, indexing='xy')
dx = xx[0, 1] - xx[0, 0]  # change along x-axis (horizontal direction)
dy = yy[1, 0] - yy[0, 0]  # change along y-axis (vertical direction)



#### Inter polation    

grid_u = griddata((X, Y), U, (xx, yy), method='linear')
grid_v = griddata((X, Y), V, (xx, yy), method='linear')

vector = jnp.array([grid_u, grid_v])
vector = jnp.array([grid_u, grid_v])
mask = ~jnp.isnan(vector)
vector = jnp.nan_to_num(vector, nan=0.0)



# SYSTÈME À DÉCOMPOSER
vector_field = vector * window_function(xx, yy)
"""noise = np.random.normal(0, scale=0.001, size=vector_field.shape)

vector_field += noise"""

grad_pot_recomp, rot_pot_recomp, vector_field_recomp, _, _, _, _ = helmholtz_decomp(xx, yy, vector_field, error_display=False, streamplots=False, fields=False, potentials=False)

orig_gradient_vector = -1 * JAXgradFinite(grad_pot_recomp, dx, dy)
orig_rotation_vector = JAXskew_gradFinite(rot_pot_recomp, dx, dy)
xx_mask, yy_mask, orig_gradient_vector_mask, orig_rotation_vector_mask, grad_pot_recomp_mask, rot_pot_recomp_mask = reduced_region(xx, yy, region_x, region_y, orig_gradient_vector, orig_rotation_vector, grad_pot_recomp, rot_pot_recomp)
vector_field_mask = clip(xx, yy, region_x, region_y, vector_field)



fit_grad = JAXgaussian_mixture(xx, yy, params_grad)
fit_rot = JAXgaussian_mixture(xx, yy, params_rot)




gradient_vector = gradient_operator(xx, yy, params[0])
rotation_vector = skewgradient_operator(xx, yy, params[1])

gradient_vector_mask = gradient_operator(xx_mask, yy_mask, params[0])
rotation_vector_mask = skewgradient_operator(xx_mask, yy_mask, params[1])

vector_field_gaussians = gradient_vector + rotation_vector
vector_field_gaussians_mask = gradient_vector_mask + rotation_vector_mask


mask_mask = clip(xx, yy, region_x, region_y, mask)

###### RECALCULE DE LA LOSS ANGLE ET MAGNITUDE

eps = 1e-12


prediction_vec = vector_field_gaussians_mask

dot_product = jnp.sum(prediction_vec * vector_field_mask, axis=0)
magnitude_vec_Z = JAXnorm(vector_field_mask)
magnitude_vec = JAXnorm(prediction_vec)

cos_theta = -1 * (dot_product / (magnitude_vec * magnitude_vec_Z + eps)) + 1

loss_magnitude = jnp.tanh((magnitude_vec - magnitude_vec_Z)**2)
loss_angle = cos_theta

loss_magnitude_mask = loss_magnitude * mask_mask
loss_angle_mask = loss_angle * mask_mask

abc = jnp.sum(loss_magnitude_mask) / jnp.sum(mask)
deg = jnp.sum(loss_angle_mask) / jnp.sum(mask)

sigmas = jnp.concatenate([params[0][:, 3:5].reshape(-1), params[1][:, 3:5].reshape(-1)])
print(sigmas)
loss_sigmas = sigmoid(sigmas, amp, sharpness, sigma_max)

print(loss_sigmas)

mean_loss_sigmas = jnp.sum(loss_sigmas)


print("Loss magnitude : ", abc)
print("Loss angle : ", deg)
print("Loss sigma : ", mean_loss_sigmas)

print("Loss : ", cost_func(params, xx_mask, yy_mask, vector_field_mask, mask_mask))





found_gradient_vector_mask = clip(xx, yy, region_x, region_y, gradient_vector)
found_rotation_vector_mask = clip(xx, yy, region_x, region_y, rotation_vector)

vector_field_gaussians_mask = clip(xx, yy, region_x, region_y, vector_field_gaussians)


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
        f"Loss magnitude: {abc}",
        f"Loss angle: {deg}",
        f"Loss sigma: {mean_loss_sigmas}",
        f"Total Loss: {cost_func(params, xx_mask, yy_mask, vector_field_mask, mask_mask)}",
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

    show_fields(xx_mask, yy_mask, found_gradient_vector_mask, found_rotation_vector_mask, "Curl free vector field from fit", "Divergence free vector field from fit")
    pdf.savefig()
    plt.close()
    show_fields(xx_mask, yy_mask, orig_gradient_vector_mask, orig_rotation_vector_mask, "Original curl free vector field", "Original divergence free vector field")
    pdf.savefig()
    plt.close()
    show_fields(xx_mask, yy_mask, vector_field_gaussians_mask, vector_field_mask, "Vector field from Gaussian mask", "Original vector field mask")
    pdf.savefig()
    plt.close()
    show_streamplots(xx_mask, yy_mask, vector_field_gaussians_mask, vector_field_mask, "Streamplot from Gaussian mask", "Original streamplot mask")
    pdf.savefig()
    plt.close()
    show_fields(xx, yy, vector_field_gaussians, vector_field, "Vector field from Gaussian", "Original vector field")
    pdf.savefig()
    plt.close()
    show_streamplots(xx, yy, vector_field_gaussians, vector_field, "Streamplot from Gaussian", "Original streamplot")
    pdf.savefig()
    plt.close()
    error_on_vect_field = error(vector_field_mask, vector_field_gaussians_mask)
    #print(round(np.max(error_on_vect_field), 2))

    show_error_display(xx_mask, yy_mask, error_on_vect_field, "Error between original vector field and gaussians")
    pdf.savefig()
    plt.close()

    show_potentials(xx, yy, fit_grad, fit_rot, "Fitted curl free potential", "Fitted divergence free potential")
    pdf.savefig()
    plt.close()
    show_potentials(xx, yy, grad_pot_recomp, rot_pot_recomp, "HD curl free potential", "HD divergence free potential")
    pdf.savefig()
    plt.close()