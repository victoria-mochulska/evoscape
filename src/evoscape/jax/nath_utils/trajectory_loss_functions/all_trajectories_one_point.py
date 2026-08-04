
import jax
import jax.numpy as jnp
from src.utilities import *
from functools import partial
import jax.lax as lax









################### SIMULATING 1 TIME POINT FOR ALL TIMES AND COMPARING EVERYTHING #############################
# work but the result is not that good.



# params is a tuple of (2) of [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("Delta_t", "ndt"))
def one_trajectory_time(params, initial_point, Delta_t, ndt=50):
    dt = Delta_t / ndt

    # donne (2,N)
    def f(pos):
        pos_x = pos[0]
        pos_y = pos[1]
        return JAXflow_grad_mixture(pos_x, pos_y, params[0]) + JAXflow_rot_mixture(pos_x, pos_y, params[1]) #JAXflow(pos_x, pos_y, params)

    def inner_step(_, pos):
        return pos + f(pos) * dt
    
    pos = lax.fori_loop(0, ndt, inner_step, initial_point)

    return pos

# params is a tuple of (2) of [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("time", "ndt"))
def one_trajectory_mult_times(params, initial_points_time, time, ndt=50):
    Delta_t = (time-1)/time
    return jax.vmap(one_trajectory_time, in_axes=(None, 0, None, None))(params, initial_points_time, Delta_t, ndt) # need (2) and will give (T, 2)

@partial(jax.jit, static_argnames=("time", "ndt"))
def all_trajectories_mult_times(params, initial_points_space, time, ndt=50):
    return jax.vmap(one_trajectory_mult_times, in_axes=(None, 0, None, None))(params, initial_points_space, time, ndt)  #need init points = (T, 2) will give (N, T, 2)


# parmas is still a tuple of (2) with shape [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("time","ndt"))
def compare_trajectories_one_point(params, data, time, initial_points, first_points, ndt=50):
    predictions_trajectories = all_trajectories_mult_times(params, initial_points, time, ndt)

    diff = predictions_trajectories - data
    norm = diff[:,:,0]**2+diff[:,:,1]**2
    return jnp.mean(norm)

grad_cost2 = jax.grad(compare_trajectories_one_point, argnums=(0))

@partial(jax.jit, static_argnames=("time","ndt"))
def multiple_gaussians_comparisons_one_point(params, data, time, initial_points, first_points, ndt=50):
    return jax.vmap(compare_trajectories_one_point, in_axes=(0, None, None, None, None, None))(params, data, time, initial_points, first_points, ndt)

@partial(jax.jit, static_argnames=("time", "ndt", "iter", "lr"))
def third_descent(params, data, time, initial_points, first_points, ndt=50, iter=100, lr=0.005):
    def gradient_step(t, params):
        grad_grad, grad_rot = grad_cost2(params, data, time, initial_points, first_points, ndt)

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        params = (new_params_grad, new_params_rot)

        return params

    params = jax.lax.fori_loop(0, iter, gradient_step, params)

    return params

@partial(jax.jit, static_argnames=("time", "ndt", "iter", "lr"))
def batch_third_descent(params, data, time, initial_points, first_points, ndt=50, iter=100, lr=0.005):
    return jax.vmap(third_descent, in_axes=(0, None, None, None, None, None, None, None))(params, data, time, initial_points, first_points, ndt, iter, lr)


def ONE_PT_trajec(params, data, time, initial_points, first_points, maxiter, ndt=50, verbose_dt=100, lr=0.005):
    pbar = tqdm.tqdm(total=maxiter)
    evol = []

    for i in range(maxiter//verbose_dt):
        params = batch_third_descent(params, data, time, initial_points, first_points, ndt, verbose_dt, lr)
        loss = jnp.min(multiple_gaussians_comparisons_one_point(params, data, time, initial_points, first_points, ndt))
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(loss)

    return params, evol


