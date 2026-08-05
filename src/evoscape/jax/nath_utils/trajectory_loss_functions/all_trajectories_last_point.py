import jax
import jax.numpy as jnp
from evoscape.jax.nath_utils.utilities import *
from functools import partial
import jax.lax as lax







################### SIMULATING ALL TRAJECTORIES (WITH LOOP) AND COMPARING LAST POINT ONLY #############################
# Doesn't really work, really slow and has Nan

# params is a tuple of (2) of [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("Delta_t", "ndt"))
def one_trajectory_loop(params, initial_point, Delta_t, ndt=50):
    dt = Delta_t / ndt

    # donne (2,N)
    def f(pos):
        pos_x = pos[0]
        pos_y = pos[1]
        return JAXflow_grad_mixture(pos_x, pos_y, params[0]) + JAXflow_rot_mixture(pos_x, pos_y, params[1]) #JAXflow(pos_x, pos_y, params)

    def inner_step(_, pos):
        return pos + f(pos) * dt
    
    pos = lax.fori_loop(0, ndt * Delta_t, inner_step, initial_point)

    return pos


@partial(jax.jit, static_argnames=("Delta_t", "ndt"))
def all_end(params, initial_points, Delta_t, ndt=50):
    return jax.vmap(one_trajectory_loop, in_axes=(None, 0, None, None))(params, initial_points, Delta_t, ndt)  #need init points = (T, 2) will give (N, T, 2)


# parmas is still a tuple of (2) with shape [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("Delta_t","ndt"))
def compare_end(params, ends, Delta_t, initial_points, ndt=50):
    predicted_ends = all_end(params, initial_points, Delta_t, ndt)

    diff = predicted_ends - ends
    norm = diff[:,0]**2+diff[:,1]**2
    return jnp.mean(norm)

grad_cost3 = jax.grad(compare_end, argnums=(0))

@partial(jax.jit, static_argnames=("Delta_t","ndt"))
def multiple_gaussians_comparisons_last_point(params, ends, Delta_t, initial_points, ndt=50):
    return jax.vmap(compare_end, in_axes=(0, None, None, None, None))(params, ends, Delta_t, initial_points, ndt)

@partial(jax.jit, static_argnames=("Delta_t", "ndt", "iter", "lr"))
def fourth_descent(params, ends, Delta_t, initial_points, ndt=50, iter=100, lr=0.005):
    def gradient_step(t, params):
        grad_grad, grad_rot = grad_cost3(params, ends, Delta_t, initial_points, ndt)

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        params = (new_params_grad, new_params_rot)

        return params

    params = jax.lax.fori_loop(0, iter, gradient_step, params)

    return params

@partial(jax.jit, static_argnames=("Delta_t", "ndt", "iter", "lr"))
def batch_fourth_descent(params, ends, Delta_t, initial_points, ndt=50, iter=100, lr=0.005):
    return jax.vmap(fourth_descent, in_axes=(0, None, None, None, None, None, None))(params, ends, Delta_t, initial_points, ndt, iter, lr)


def LAST_PT_trajec(params, ends, time, initial_points, maxiter, ndt=50, verbose_dt=100, lr=0.005):
    pbar = tqdm.tqdm(total=maxiter)
    evol = []
    Delta_t = (time-1)

    for i in range(maxiter//verbose_dt):
        params = batch_fourth_descent(params, ends, Delta_t, initial_points,  ndt, verbose_dt, lr)
        loss = jnp.mean(multiple_gaussians_comparisons_last_point(params, ends, Delta_t, initial_points, ndt))
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(loss)

    return params, evol


