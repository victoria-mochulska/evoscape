import jax
import jax.numpy as jnp
from src.utilities import *
from functools import partial
import jax.lax as lax
import tqdm






################### SIMULATING N TIME POINT FOR ALL/N TIMES AND COMPARING ALL/N #############################


#@partial(jax.jit, static_argnames=("interval_of_trajectories", "length_of_mini_traj"))
def give_mini_traj(data, interval_of_trajectories, length_of_mini_traj):

    initial_points = data[::interval_of_trajectories, ::length_of_mini_traj, :]
    if data.shape[1] % length_of_mini_traj != 0:
        initial_points= initial_points[:, :-1 ,:]
        mini_traj_base = data[::interval_of_trajectories, :-(data.shape[1]%length_of_mini_traj), :]
    else:
        mini_traj_base = data[::interval_of_trajectories, :, :]

    initial_points = initial_points.reshape(-1,2)


    mini_traj = mini_traj_base.reshape(initial_points.shape[0], length_of_mini_traj, mini_traj_base.shape[2])

    return initial_points, mini_traj[:, -1, :]

# params is a tuple of (2) of [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("length_of_mini_traj", "Delta_t", "ndt"))
def one_trajectory_frac(params, initial_point, length_of_mini_traj, Delta_t, ndt=50):
    dt = Delta_t / ndt

    # donne (2,N)
    def f(pos):
        pos_x = pos[0]
        pos_y = pos[1]
        return JAXflow_grad_mixture(pos_x, pos_y, params[0]) + JAXflow_rot_mixture(pos_x, pos_y, params[1]) #JAXflow(pos_x, pos_y, params)

    def inner_step(_, pos):
        return pos + f(pos) * dt
    
    pos = lax.fori_loop(0, ndt * length_of_mini_traj, inner_step, initial_point)

    return pos


@partial(jax.jit, static_argnames=("length_of_mini_traj", "Delta_t", "ndt"))
def all_traj_frac(params, initial_points, length_of_mini_traj, Delta_t, ndt=50):
    return jax.vmap(one_trajectory_frac, in_axes=(None, 0, None, None, None))(params, initial_points, length_of_mini_traj, Delta_t, ndt)


# parmas is still a tuple of (2) with shape [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("length_of_mini_traj", "Delta_t", "ndt"))
def compare_mini_traj(params, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt=50):
    predicted_ends = all_traj_frac(params, initial_points, length_of_mini_traj, Delta_t, ndt)

    diff = predicted_ends - mini_traj
    norm = diff[:,0]**2+diff[:,1]**2
    return jnp.mean(norm)

grad_cost3 = jax.grad(compare_mini_traj, argnums=(0))

@partial(jax.jit, static_argnames=("length_of_mini_traj", "Delta_t", "ndt"))
def multiple_gaussians_comparisons_frac_point(params, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt=50):
    return jax.vmap(compare_mini_traj, in_axes=(0, None, None, None, None, None))(params, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt)

@partial(jax.jit, static_argnames=("length_of_mini_traj" , "Delta_t", "ndt", "iter", "lr"))
def mini_descent(params, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt=50, iter=100, lr=0.005):
    def gradient_step(t, params):
        grad_grad, grad_rot = grad_cost3(params, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt)

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        params = (new_params_grad, new_params_rot)

        return params

    params = jax.lax.fori_loop(0, iter, gradient_step, params)

    return params

@partial(jax.jit, static_argnames=("length_of_mini_traj", "Delta_t", "ndt", "iter", "lr"))
def batch_mini_descent(params, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt=50, iter=100, lr=0.005):
    return jax.vmap(mini_descent, in_axes=(0, None, None, None, None, None, None, None))(params, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt, iter, lr)


def FRAC_PT_trajec(params, data, time, frac_time, frac_space, maxiter, ndt=50, verbose_dt=100, lr=0.005):
    pbar = tqdm.tqdm(total=maxiter)
    evol = []
    Delta_t = (time-1)/time

    shape_space, shape_time, coord = data.shape
    interval_of_trajectories = int((100/frac_space))
    length_of_mini_traj = int(shape_time * (frac_time/100))

    initial_points, mini_traj = give_mini_traj(data, interval_of_trajectories, length_of_mini_traj)
    
    print(initial_points.shape)
    print(mini_traj.shape)

    for i in range(maxiter//verbose_dt):
        params = batch_mini_descent(params, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt, verbose_dt, lr)
        loss = jnp.min(multiple_gaussians_comparisons_frac_point(params, initial_points, mini_traj, length_of_mini_traj, Delta_t, ndt))
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(loss)

    return params, evol
