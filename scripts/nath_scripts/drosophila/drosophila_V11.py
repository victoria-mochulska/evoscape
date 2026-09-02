# making 2 runs sequentially ==> first we make the run with only the vectors and then we otpimized the trajectories




import jax
import jax.numpy as jnp
from evoscape.jax.nath_utils.utilities import *
import time
from matplotlib.backends.backend_pdf import PdfPages
import json
import logging
from pathlib import Path
from datetime import datetime
import os
import argparse
from scipy.interpolate import griddata




job_id = os.getenv("SLURM_JOB_ID", "local")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

run_dir = Path("results") / f"run_{timestamp}_job{job_id}"
run_dir.mkdir(parents=True, exist_ok=True)


# Logging
logging.basicConfig(
    filename=run_dir / "log.txt",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


parser = argparse.ArgumentParser()
parser.add_argument("--GPU", action="store_true", default=True)
parser.add_argument("--CPU", action="store_true", default=False)
parser.add_argument("--nbr_gpu", type=int, default=1)
parser.add_argument("--nbr_cpu", type=int, default=20)
parser.add_argument("--times", type=int, default=1)
parser.add_argument("--precision64", action="store_true", default=False)

parser.add_argument("--percentage_time", type=int, default=100)
parser.add_argument("--percentage_space", type=int, default=100)

"""parser.add_argument("--mult", type=bool)
parser.add_argument("--lr", type=float)
parser.add_argument("--maxiter", type=int)
parser.add_argument("--gaussians", type=int)"""


parser.add_argument("--nbr_gaussians_grad", type=int, default=3)
parser.add_argument("--nbr_gaussians_rot", type=int, default=3)
parser.add_argument("--maxiter", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--verbose_dt", type=int, default=100)
parser.add_argument("--frac_height", type=int, default=10)
parser.add_argument("--frac_space", type=int, default=10)
parser.add_argument("--frac_time", type=int, default=4)
parser.add_argument("--percent", type=float, default=1)
parser.add_argument("--length", type=int, default=15)
parser.add_argument("--angle_importance", type=float, default=1)




args = parser.parse_args()
GPU = args.GPU
CPU = args.CPU
nbr_gpu = args.nbr_gpu
nbr_cpu = args.nbr_cpu
times = args.times
precision64 = args.precision64
percentage_time = args.percentage_time
percentage_space = args.percentage_space
nbr_gaussians_grad = args.nbr_gaussians_grad
nbr_gaussians_rot = args.nbr_gaussians_rot
MAXITER = args.maxiter
lr = args.lr 
verbose_dt = args.verbose_dt
frac_height = args.frac_height
frac_space = args.frac_space
frac_time = args.frac_time
percent = args.percent
length = args.length
angle_importance = args.angle_importance


params = vars(args)

with open(run_dir / "parameters.json", "w") as f:
    json.dump(params, f, indent=4)
logging.info("Parameters saved")


seed_number = int(time.time_ns())
logging.info(f"seed_number: {seed_number}")


if GPU:
    gpu = nbr_gpu
elif CPU:
    cpu = nbr_cpu



jax.config.update("jax_enable_x64", False)
if CPU:
    jax.config.update('jax_num_cpu_devices', cpu)

mult = False



###### EXTRACTING DATA ##################


# Load gene data
Gt_data = extract_gene_data("src/gene_data/Gt_3002.txt")
Kni_data = extract_gene_data("src/gene_data/Kni_3002.txt")
Hb_data = extract_gene_data("src/gene_data/Hb_3002.txt")
Kr_data = extract_gene_data("src/gene_data/Kr_3002.txt")
logging.info("Gene data extracted")

# Compute differences
H1 = Kr_data - Gt_data
H2 = Hb_data - Kni_data


#visu_latent_space(H1, H2, int(H1.shape[0] * 0.60))
limit_time = int(H1.shape[0] * 0.56)
limit_space = int(H1.shape[1]*0.40)

H1 = H1[:limit_time, limit_space:]
H2 = H2[:limit_time, limit_space:]
Gt_data = Gt_data[:limit_time, limit_space:]
Kni_data = Kni_data[:limit_time, limit_space:]
Hb_data = Hb_data[:limit_time, limit_space:]
Kr_data = Kr_data[:limit_time, limit_space:]


all_coords = np.array([H1, H2])


vect_all = make_derivative_4th_order(all_coords)


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


skip_time = int(100/percentage_time)
skip_space = int(100/percentage_space)

vect_all_trimmed = vect_all[:,::skip_time, ::skip_space]

X_all_trimmed = H1[::skip_time,::skip_space]
Y_all_trimmed = H2[::skip_time,::skip_space]
U_all_trimmed = vect_all[0,::skip_time,::skip_space]
V_all_trimmed = vect_all[1,::skip_time,::skip_space]


all_coords_trimmed = np.array([X_all_trimmed, Y_all_trimmed])













####### SINGLE BATCH ##########

max_val = max(np.max(np.abs(H1)), np.max(np.abs(H2)))

L = max_val*2
npoints = 600 + 1

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
logging.info("Helmholtz decomposition done")


orig_gradient_vector = -1 * JAXgradFinite(grad_pot_recomp, dx, dy)
orig_rotation_vector = JAXskew_gradFinite(rot_pot_recomp, dx, dy)





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
max_grad  = jnp.max(surface_grad) * 2
min_grad = jnp.min(surface_grad) * 2

surface_rot = rot_pot_recomp
max_rot = jnp.max(surface_rot) * 2
min_rot = jnp.min(surface_rot) * 2


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
                init_conditions_grad = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds_grad, nbr_gaussians_grad, X_all, Y_all, max_grad, min_grad)
                init_conditions_grad_sharded = jax.device_put(init_conditions_grad, sharding)

            if nbr_gaussians_rot == 0:
                init_conditions_rot_sharded = 0 
            else:
                init_conditions_rot = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds_rot, nbr_gaussians_rot, X_all, Y_all, max_rot, min_rot)
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




print(nbr_gaussians_grad)
print(nbr_gaussians_rot)

#### VECTORIZED REAL DESCENT ####

learning_rate = lr

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


#init_conditions_grad_sharded = jnp.array([[-0.23289351165294647, -0.43147459626197815, 0.19158291816711426, 0.2153414487838745, 0.33802297711372375, 5.484355926513672], [0.12188774347305298, 0.013387732207775116, 0.1152237132191658, 0.4090495705604553, 0.3055126667022705, 4.889540672302246], [-0.007605677470564842, 0.039299529045820236, -4.21609354019165, 0.061275988817214966, 0.0804225355386734, 0.4116712808609009]])
#init_conditions_rot_sharded = jnp.array([[-0.6409302353858948, -0.29425981640815735, 0.09684016555547714, 0.5276364088058472, 0.26812097430229187, 0.22394603490829468], [-0.4260116219520569, 0.30880239605903625, -2.0388145446777344, 0.2433188557624817, 0.09593580663204193, -0.5400583744049072], [-0.10509275645017624, 0.0829554945230484, 3.19108247756958, 0.09839922934770584, 0.09052365273237228, 3.1908345222473145]])


"""start_time = time.time()
losses, params_grad, params_rot = batch_gradient_descent(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, learning_rate, init_conditions_grad_sharded, init_conditions_rot_sharded)
print(f"Time without verbose :{time.time() - start_time:.2f}")
"""

threshold = 10*max_val

logging.info("Optimization started")
start_time = time.time()
#losses, params_grad, params_rot, evol = train_verbose(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, learning_rate, init_conditions_grad_sharded, init_conditions_rot_sharded, threshold, frac_height, MAXITER, verbose_dt)
losses, params_grad, params_rot, evol, params_grad_history, params_rot_history, loss_history = train_verbose_optax(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, learning_rate, init_conditions_grad_sharded, init_conditions_rot_sharded, threshold, frac_height, angle_importance, MAXITER, verbose_dt)
params_grad.block_until_ready()
print("First run done")
#losses, params_grad, params_rot, evol, params_grad_history, params_rot_history, loss_history = train_verbose_optax_traj_rand(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, lr, percent, params_grad, params_rot, jnp.swapaxes(all_coords_trimmed, 0, 2), seed, length, 100000, verbose_dt)
#losses, params_grad, params_rot, evol = train_verbose_optax_traj(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, all_coords_trimmed, learning_rate, init_conditions_grad_sharded, init_conditions_rot_sharded, threshold, frac_height, MAXITER, verbose_dt)
#losses, params_grad, params_rot, evol = train_verbose_optax_traj_frac(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, all_coords_trimmed, learning_rate, init_conditions_grad_sharded, init_conditions_rot_sharded, threshold, frac_height, frac_time, frac_space, MAXITER, verbose_dt)
#losses, params_grad, params_rot, evol, params_grad_history, params_rot_history, loss_history = train_verbose_optax_traj_rand(X_all_trimmed, Y_all_trimmed, vect_all_trimmed, lr, percent, init_conditions_grad_sharded, init_conditions_rot_sharded, jnp.swapaxes(all_coords_trimmed, 0, 2), seed, length, MAXITER, verbose_dt)
print(f"Time with verbose :{time.time() - start_time:.2f}")
logging.info("Optimization done")

evol = jnp.array(evol)

params_grad_l=np.array(params_grad_history)
params_rot_l=np.array(params_rot_history)
losses_l=np.array(loss_history)
save_info(params_grad_l, params_rot_l, losses_l, run_dir / "optimization_history.npz")
#write_info(params_grad_l, params_rot_l, losses_l, run_dir / "optimization_history.txt")





### sorting the candidates ######
idx_sort = jnp.argsort(losses, axis=0, descending=False)

ordered_losses = losses[idx_sort]
ordered_params_grad = params_grad[idx_sort]
ordered_params_rot = params_rot[idx_sort]

ordered_candidate_params_grad_history = params_grad_l[:, idx_sort, :, :]
best_candidate_params_grad_history = ordered_candidate_params_grad_history[:, 0, :, :]

ordered_candidate_params_rot_history = params_rot_l[:, idx_sort, :, :]
best_candidate_params_rot_history = ordered_candidate_params_rot_history[:, 0, :, :]



#write_ordered_result(ordered_params_grad, ordered_params_rot=, ordered_losses, run_dir / "ordered.txt")


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



#params_grad = jnp.array([[-0.23289351165294647, -0.43147459626197815, 0.19158291816711426, 0.2153414487838745, 0.33802297711372375, 5.484355926513672], [0.12188774347305298, 0.013387732207775116, 0.1152237132191658, 0.4090495705604553, 0.3055126667022705, 4.889540672302246], [-0.007605677470564842, 0.039299529045820236, -4.21609354019165, 0.061275988817214966, 0.0804225355386734, 0.4116712808609009]])
#params_rot = jnp.array([[-0.6409302353858948, -0.29425981640815735, 0.09684016555547714, 0.5276364088058472, 0.26812097430229187, 0.22394603490829468], [-0.4260116219520569, 0.30880239605903625, -2.0388145446777344, 0.2433188557624817, 0.09593580663204193, -0.5400583744049072], [-0.10509275645017624, 0.0829554945230484, 3.19108247756958, 0.09839922934770584, 0.09052365273237228, 3.1908345222473145]])
params = (params_grad, params_rot)


save_best_params(params_grad, params_rot, run_dir / "best_params.npz")


fit_grad = JAXgaussian_mixture(xx, yy, params_grad)
fit_rot = JAXgaussian_mixture(xx, yy, params_rot)
gradient_vector = JAXflow_grad_mixture(xx, yy, params[0])
rotation_vector = JAXflow_rot_mixture(xx, yy, params[1])
vector_field_gaussians = gradient_vector + rotation_vector


final, traj = make_trajectories(params, all_coords, ndt=100)

start_time = time.time()
#show_animated_potentials(xx, yy, best_candidate_params_grad_history, best_candidate_params_rot_history, "Best fit grad animated", "Best fit rot animated", run_dir / "animated_potentials.gif")
#show_animated_field(xx, yy, best_candidate_params_grad_history, best_candidate_params_rot_history, "Best fit vector field animated", run_dir / "animated_fields_combined.gif")
#show_animated_streamplot(xx, yy, best_candidate_params_grad_history, best_candidate_params_rot_history, "Best fit streamplot animated", run_dir / "animated_streamplots_combined.gif")
#show_trajectory(all_coords[:, :, ::int(100/10)], traj[:, :, ::int(100/10)], run_dir / "latent_space_trajectories.gif")
print(f"Time for animated stuff :{time.time() - start_time:.2f}")

Kr_simul, Hb_simul, Gt_simul, Kni_simul = retrieve_data(traj)
Kr_truncated, Hb_truncated, Gt_truncated, Kni_truncated = retrieve_data(all_coords)



print(f"Forme de la trajectoire recomposée {traj.shape}")
print(f"Forme des trajectoires des données {all_coords.shape}")


show_trajectory(all_coords[:, :, ::skip_space], traj[:, :, ::skip_space], run_dir / "latent_space_trajectories.gif")


fitted_vect_field = JAXflow_grad_mixture(X_all, Y_all, params[0]) + JAXflow_rot_mixture(X_all, Y_all, params[1])
fitted_vect_field_trimmed = fitted_vect_field[:,::skip_time, ::skip_space]
loss_magnitude = loss_magnitude_func(fitted_vect_field, vect_all)
loss_angle = loss_angle_func(fitted_vect_field, vect_all)
#loss_sigma = loss_sigma_func(params, amp, sharpness, sigma_max)
loss_sigma = loss_sigma_func(params, threshold, frac_height)
total_loss = cost_func(params, X_all, Y_all, vect_all, threshold, frac_height, angle_importance)

sigmas = jnp.concatenate([params[0][:, 3:5].reshape(-1), params[1][:, 3:5].reshape(-1)])
loss_sigmas = JAXsoftplus_barrier(sigmas, threshold)

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
print(f"Average iteration time {elapsed_time/MAXITER:.2f}")
if len(evol) > 1:
    loss_evol = np.arange(0, evol.shape[0]*verbose_dt, verbose_dt)
    evol = evol.T


#### make gifs ####
"""show_gene_evol(Gt_data, Gt_simul, Gt_truncated,
               Kni_data, Kni_simul, Kni_truncated,
               Kr_data, Kr_simul, Kr_truncated,
               Hb_data, Hb_simul, Hb_truncated, run_dir / "gene_evol.gif")"""



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

results = {
    "loss_magnitude": float(loss_magnitude),
    "loss_angle": float(loss_angle),
    "loss_sigma": float(loss_sigma),
    "total_loss": float(total_loss),
    "RMS_pts": float(RMS_pts),
    "RMS_vec": float(RMS_vec),
}
with open(run_dir / "results.json", "w") as f:
    json.dump(results, f, indent=4)

with PdfPages(run_dir / "myplots.pdf") as pdf:

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


    show_fields_with_modules(xx, yy, gradient_vector, rotation_vector, params_grad, params_rot, "Curl free vector field from fit", "Divergence free vector field from fit")
    pdf.savefig()
    plt.close()


    show_fields(xx, yy, orig_gradient_vector, orig_rotation_vector, "Original curl free vector field", "Original divergence free vector field")
    pdf.savefig()
    plt.close()

    show_fields(xx, yy, vector_field_gaussians, vector_interpolated, "Vector field from Gaussian", "Original vector field")
    pdf.savefig()
    plt.close()


    show_streamplots_with_modules(xx, yy, vector_field_gaussians, vector_interpolated, params_grad, params_rot, "Streamplot from Gaussian", "Original streamplot")
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



