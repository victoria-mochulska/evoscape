# making a loss for trajectories instead of vectors

import sys
sys.path.insert(0, "../helm_decomp_project_venv")

GPU = True
CPU = False


import jax
import jax.numpy as jnp

from src.utilities import *
from trajectory_loss_functions import *

import time


MAXITER = 10000

if GPU:
    gpu = 1

elif CPU:
    cpu = 40

times = 1

nbr_init_cond = 200


jax.config.update("jax_enable_x64", False)
if CPU:
    jax.config.update('jax_num_cpu_devices', cpu)
else:
    pass

#jax.config.update("jax_log_compiles", True)




import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.interpolate import griddata
import tqdm


seed_number = int(time.time_ns())


# Load gene data
Gt_data = extract_gene_data("gene_data/Gt_3002.txt")
Kni_data = extract_gene_data("gene_data/Kni_3002.txt")
Hb_data = extract_gene_data("gene_data/Hb_3002.txt")
Kr_data = extract_gene_data("gene_data/Kr_3002.txt")

# Compute differences
H1 = Kr_data - Gt_data
H2 = Hb_data - Kni_data

#visu_latent_space(H1, H2)
limit_time = int(H1.shape[0] * 0.56)
limit_space = int(H1.shape[1]*0.40)

H1 = H1[:limit_time, limit_space:]
H2 = H2[:limit_time, limit_space:]



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


percentage_time = 100
percentage_space = 100

skip_time = int(100/percentage_time)
skip_space = int(100/percentage_space)

vect_all_trimmed = vect_all[:,::skip_time, ::skip_space]

X_all_trimmed = H1[::skip_time,::skip_space]
Y_all_trimmed = H2[::skip_time,::skip_space]
U_all_trimmed = vect_all[0,::skip_time,::skip_space]
V_all_trimmed = vect_all[1,::skip_time,::skip_space]



all_coords = all_coords[:,::skip_time, ::skip_space]



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
    pass
    #mesh_devices = mesh_utils.create_device_mesh((gpu,))  # 1D mesh of 50
    #mesh = Mesh(mesh_devices, ('i',))  # nomme l'axe 'i'    
if CPU:
    mesh_devices = mesh_utils.create_device_mesh((cpu,))  # 1D mesh of 50
    mesh = Mesh(mesh_devices, ('i',))  # nomme l'axe 'i'    

devices = jax.devices("gpu")[:gpu]

mesh = Mesh(
    np.array(devices),
    ('i',)
)

from jax.sharding import PartitionSpec, NamedSharding

spec = PartitionSpec('i', None, None)  # on shard l'axe 0, pas les autres
sharding = NamedSharding(mesh, spec)
print(mesh)
print(spec)





#### VECTORIZED TEST DESCENT ####

#nbr_init_cond = cpu*times

start_time = time.time()
conditions = []

surface_grad = grad_pot_recomp
max_grad  = jnp.max(surface_grad) * 1.20
min_grad = jnp.min(surface_grad) * 1.20

surface_rot = rot_pot_recomp
max_rot = jnp.max(surface_rot) * 1.20
min_rot = jnp.min(surface_rot) * 1.20




seed = jax.random.PRNGKey(seed_number + 9191)


grad_gaussian = 1
rot_gaussian = 2

seed_split = jax.random.split(seed, 2)

seeds = jax.random.split(seed_split[0], num=nbr_init_cond)

init_conditions_grad = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds, grad_gaussian, xx, yy, max_grad, min_grad)
init_conditions_grad_sharded = jax.device_put(init_conditions_grad, sharding)


seeds = jax.random.split(seed_split[1], num=nbr_init_cond)

init_conditions_rot = jax.vmap(random_init, in_axes=(0, None, None, None, None, None))(seeds, rot_gaussian, xx, yy, max_rot, min_rot)
init_conditions_rot_sharded = jax.device_put(init_conditions_rot, sharding)


params = (init_conditions_grad_sharded, init_conditions_rot_sharded)

changed_coords = jnp.swapaxes(all_coords, 0, 2)

#print(params.shape)

"""
start = time.time()
params_opt, evol = ALLtrajec2(params, changed_coords, changed_coords[:,0,:], MAXITER, verbose_dt=100, ndt=10)
print("temps descente :",time.time()-start)
params_grad, params_rot = params_opt

loss_idx = jnp.argmin(multiple_gaussians_comparisons_all_points(params_opt, changed_coords[:, 1:, :], changed_coords[:, 0, :], ndt=50))
best_params_grad = params_grad[loss_idx]
best_params_rot = params_rot[loss_idx]
best_params = (best_params_grad, best_params_rot)

_, traj = make_trajectories((best_params_grad, best_params_rot), all_coords, ndt=100)

fig = show_trajectories(all_coords, traj, dpi=450)
plt.savefig(f"trajectory_loss_functions/pic2.png")
plt.close(fig)

error_pts = RMS_points(all_coords, traj)
print("RMS error", error_pts)


gradient_vector = JAXflow_grad_mixture(xx, yy, best_params_grad)
rotation_vector = JAXflow_rot_mixture(xx, yy, best_params_rot)


vector_field_gaussians = gradient_vector + rotation_vector

show_fields(xx, yy, vector_field_gaussians, vector_interpolated, "Vector field from Gaussian", "Original vector field")
plt.savefig(f"trajectory_loss_functions/field.png")
plt.close()
show_streamplots(xx, yy, vector_field_gaussians, vector_interpolated, "Streamplot from Gaussian", "Original streamplot")
plt.savefig(f"trajectory_loss_functions/streamplot.png")
plt.close()

print("Finished")"""

"""
start = time.time()
params_opt, evol = second_verbose_descent(params, changed_coords, changed_coords[:,0,:], MAXITER, verbose_dt=5)
print("temps descente :",time.time()-start)
params_grad, params_rot = params_opt"""























"""start = time.time()
time2 = changed_coords.shape[1]
initial_points = changed_coords[:, :-1, :]
first_points = changed_coords[:, 0, :]
params_opt, evol = ONE_PT_trajec(params, changed_coords[:, 1:, :], time2, initial_points, first_points, MAXITER, ndt=5, verbose_dt=5, lr = 0.001)
print("temps descente :",time.time()-start)
params_grad, params_rot = params_opt

loss_idx = jnp.argmin(multiple_gaussians_comparisons_one_point(params_opt, changed_coords[:, 1:, :], time2, initial_points, first_points))

best_params_grad = params_grad[loss_idx]
best_params_rot = params_rot[loss_idx]
print(best_params_grad.tolist())
print(best_params_rot.tolist())
print(compare_trajectories_one_point((best_params_grad, best_params_rot), changed_coords[:, 1:, :], time2, initial_points, first_points))

time2 = changed_coords.shape[1]
initial_points = changed_coords[:, :-1, :]
predictions_trajectories = all_trajectories_mult_times((best_params_grad, best_params_rot), initial_points, time2)
predictions_trajectories = jnp.concatenate(
[changed_coords[:, 0:1, :], predictions_trajectories],
axis=1
)

fig = show_trajectories(all_coords, jnp.swapaxes(predictions_trajectories, 0, 2), dpi=450)
plt.savefig(f"trajectory_loss_functions/pic1.png")
plt.close(fig)
best_params = (best_params_grad, best_params_rot)
_, traj = make_trajectories(best_params, all_coords, ndt=100)
fig = show_trajectories(all_coords, traj, dpi=450)
plt.savefig(f"trajectory_loss_functions/pic2.png")
plt.close(fig)

gradient_vector = JAXflow_grad_mixture(xx, yy, best_params_grad)
rotation_vector = JAXflow_rot_mixture(xx, yy, best_params_rot)


vector_field_gaussians = gradient_vector + rotation_vector

show_fields(xx, yy, vector_field_gaussians, vector_interpolated, "Vector field from Gaussian", "Original vector field")
plt.savefig(f"trajectory_loss_functions/field.png")
plt.close()
show_streamplots(xx, yy, vector_field_gaussians, vector_interpolated, "Streamplot from Gaussian", "Original streamplot")
plt.savefig(f"trajectory_loss_functions/streamplot.png")
plt.close()

print("Finished")"""








"""




start = time.time()
time2 = changed_coords.shape[1]
Delta_t = (time2-1)/time2

frac_space = 10
frac_time = 4
params_opt, evol = FRAC_PT_trajec(params, changed_coords, time2, frac_time, frac_space, MAXITER, ndt=10, verbose_dt=100, lr = 0.001)
print("temps descente :",time.time()-start)
params_grad, params_rot = params_opt
shape_space, shape_time, coord = changed_coords.shape
interval_of_trajectories = int((100/frac_space))
length_of_mini_traj = int(shape_time * (frac_time/100))

initial_points, mini_traj = give_mini_traj(changed_coords, interval_of_trajectories, length_of_mini_traj)
loss_idx = jnp.argmin(multiple_gaussians_comparisons_frac_point(params_opt, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt=50))

best_params_grad = params_grad[loss_idx]
best_params_rot = params_rot[loss_idx]
print(best_params_grad.tolist())
print(best_params_rot.tolist())


best_params = (best_params_grad, best_params_rot)
_, traj = make_trajectories(best_params, all_coords, ndt=100)
fig = show_trajectories(all_coords, traj, dpi=450)
plt.savefig(f"trajectory_loss_functions/pic2.png")
plt.close(fig)

error_pts = RMS_points(all_coords, traj)
print("RMS error", error_pts)
gradient_vector = JAXflow_grad_mixture(xx, yy, best_params_grad)
rotation_vector = JAXflow_rot_mixture(xx, yy, best_params_rot)


vector_field_gaussians = gradient_vector + rotation_vector

show_fields(xx, yy, vector_field_gaussians, vector_interpolated, "Vector field from Gaussian", "Original vector field")
plt.savefig(f"trajectory_loss_functions/field.png")
plt.close()
show_streamplots(xx, yy, vector_field_gaussians, vector_interpolated, "Streamplot from Gaussian", "Original streamplot")
plt.savefig(f"trajectory_loss_functions/streamplot.png")
plt.close()

print("Finished")




"""
























"""
start = time.time()
time2 = changed_coords.shape[1]
initial_points = changed_coords[:, 0, :]
params_opt, evol = LAST_PT_trajec(params, changed_coords[:, -1, :], time2, initial_points, MAXITER, ndt=5 , verbose_dt=10, lr = 0.0001)
print("temps descente :",time.time()-start)
params_grad, params_rot = params_opt
loss_idx = jnp.argmin(multiple_gaussians_comparisons_last_point(params_opt, changed_coords[:, -1, :], time2, initial_points))
best_params_grad = params_grad[loss_idx]
best_params_rot = params_rot[loss_idx]
print(best_params_grad.tolist())
print(best_params_rot.tolist())
print(compare_end((best_params_grad, best_params_rot), changed_coords[:, -1, :], time2, initial_points))

time2 = changed_coords.shape[1]
initial_points = changed_coords[:, 0, :]
predictions_trajectories = all_end((best_params_grad, best_params_rot), initial_points, time2)

_, traj = make_trajectories((best_params_grad, best_params_rot), all_coords, ndt=100)

fig = show_trajectories(all_coords, traj, dpi=450)
plt.savefig(f"trajectory_loss_functions/pic2.png")
plt.close(fig)

gradient_vector = JAXflow_grad_mixture(xx, yy, best_params_grad)
rotation_vector = JAXflow_rot_mixture(xx, yy, best_params_rot)


vector_field_gaussians = gradient_vector + rotation_vector

show_fields(xx, yy, vector_field_gaussians, vector_interpolated, "Vector field from Gaussian", "Original vector field")
plt.savefig(f"trajectory_loss_functions/field.png")
plt.close()
show_streamplots(xx, yy, vector_field_gaussians, vector_interpolated, "Streamplot from Gaussian", "Original streamplot")
plt.savefig(f"trajectory_loss_functions/streamplot.png")
plt.close()

print("Finished")

"""





























start = time.time()
time2 = changed_coords.shape[1]
percent = 5
length = 10
params_opt, evol = RAND_trajec(params, changed_coords, time2, percent, length, seed, MAXITER, ndt=10, verbose_dt=25, lr = 0.001)
print("temps descente :",time.time()-start)



params_grad, params_rot = params_opt
Delta_t = (time2-1)/time2
data_okay = changed_coords[:, :-length ,:]
shape_space, shape_time, coord = data_okay.shape
num_pts_iter =  int((shape_space * shape_time) * (percent/100))
indices, initial_points, ends = choose_point(seed, changed_coords, data_okay, num_pts_iter, length)
loss_idx = jnp.argmin(multiple_gaussians_comparisons_part(params, ends, initial_points, length, Delta_t, ndt=50))
best_params_grad = params_grad[loss_idx]
best_params_rot = params_rot[loss_idx]

print(best_params_grad.tolist())
print(best_params_rot.tolist())

time2 = changed_coords.shape[1]
initial_points = changed_coords[:, 0, :]
final, predictions_trajectories = all_end_part((best_params_grad, best_params_rot), initial_points, time2, Delta_t)
print(predictions_trajectories.shape)
fig = show_trajectories(all_coords, jnp.swapaxes(predictions_trajectories, 0, 2), dpi=450)
plt.savefig(f"trajectory_loss_functions/pic1.png")
plt.close(fig)


_, traj = make_trajectories((best_params_grad, best_params_rot), all_coords, ndt=100)

fig = show_trajectories(all_coords, traj, dpi=450)
plt.savefig(f"trajectory_loss_functions/pic2.png")
plt.close(fig)

error_pts = RMS_points(all_coords, traj)
print("RMS error", error_pts)


gradient_vector = JAXflow_grad_mixture(xx, yy, best_params_grad)
rotation_vector = JAXflow_rot_mixture(xx, yy, best_params_rot)


vector_field_gaussians = gradient_vector + rotation_vector

show_fields(xx, yy, vector_field_gaussians, vector_interpolated, "Vector field from Gaussian", "Original vector field")
plt.savefig(f"trajectory_loss_functions/field.png")
plt.close()
show_streamplots(xx, yy, vector_field_gaussians, vector_interpolated, "Streamplot from Gaussian", "Original streamplot")
plt.savefig(f"trajectory_loss_functions/streamplot.png")
plt.close()

print("Finished")