import jax
import jax.numpy as jnp
from evoscape.jax.nath_utils.utilities import *
from functools import partial
import jax.lax as lax

















################### SIMULATING ALL TRAJECTORIES FOR ALL POINTS AND COMPARING EVERYTHING #############################
# working

@partial(jax.jit, static_argnames=("time", "ndt"))
def one_trajectory1(params, initial_points, time, ndt=50):
    nt = time
    Delta_t = (nt-1)/nt
    dt = Delta_t / ndt

    # donne (2,N)
    def f(pos):
        pos_x = pos[0]
        pos_y = pos[1]
        return JAXflow_grad_mixture(pos_x, pos_y, params[0]) + JAXflow_rot_mixture(pos_x, pos_y, params[1]) #JAXflow(pos_x, pos_y, params) #

    def sum(pos, _):    
        pos = pos + f(pos) * dt
        return pos, pos # ('carryover', 'accumulated')


    final_pos, traj = lax.scan(jax.remat(sum), initial_points, None, length=nt*ndt - 1)

    traj = jnp.concatenate(
    [initial_points[None, :], traj],
    axis=0
    )

    return final_pos, traj[::ndt,:]

# params is a tuple of (2) of [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("time", "ndt"))
def one_trajectory_all_points(params, initial_points, time, ndt=50):
    nt = time
    Delta_t = (nt-1)/nt
    dt = Delta_t / ndt

    # donne (2,N)
    def f(pos):
        pos_x = pos[0]
        pos_y = pos[1]
        return JAXflow_grad_mixture(pos_x, pos_y, params[0]) + JAXflow_rot_mixture(pos_x, pos_y, params[1]) #JAXflow(pos_x, pos_y, params)

    def inner_step(_, pos):
        return pos + f(pos) * dt

    def outer_step(pos, _):

        pos = lax.fori_loop(
            0,
            ndt,
            inner_step,
            pos
        )

        return pos, pos

    final_pos, traj = lax.scan(
        outer_step,
        initial_points,
        None,
        length=nt - 1
    )

    traj = jnp.concatenate(
    [initial_points[None, :], traj],
    axis=0
    )

    return final_pos, traj


# parmas is still a tuple of (2) with shape [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("time", "ndt"))
def make_trajectories_all_points(params, init_points, time, ndt=50):
    return jax.vmap(one_trajectory_all_points, in_axes=(None, 0, None, None))(params, init_points, time, ndt)  #need init points = (N, 2) will give (N, T, 2)

# parmas is still a tuple of (2) with shape [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("ndt",))
def compare_trajectories_all_points(params, data, initial_points, ndt=50):
    time = data.shape[1]
    _, predictions_trajectories = make_trajectories_all_points(params, initial_points, time, ndt)

    diff = predictions_trajectories - data
    norm = diff[:,:,0]**2+diff[:,:,1]**2
    return jnp.mean(norm)

grad_cost = jax.grad(compare_trajectories_all_points, argnums=(0))


# parmas is still a tuple of (2) with shape [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("ndt",))
def multiple_grad(params, data, initial_points, ndt=50):
    return jax.vmap(grad_cost, in_axes=(0, None, None, None))(params, data, initial_points, ndt)


@partial(jax.jit, static_argnames=("ndt",))
def gradient_descent_small(params, data, initial_points, ndt=50, iter=100, lr=0.005):
    def gradient_step(t, params):
        grad_grad, grad_rot = multiple_grad(params, data, initial_points, ndt)

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        params = (new_params_grad, new_params_rot)

        return params

    params = jax.lax.fori_loop(0, iter, gradient_step, params)

    return params

def ALLtrajec1(params, data, initial_points, maxiter, ndt=50, verbose_dt=100, lr=0.005):
    pbar = tqdm.tqdm(total=maxiter)
    evol = []

    for i in range(maxiter//verbose_dt):
        params = gradient_descent_small(params, data, initial_points, ndt, verbose_dt, lr)
        loss = jnp.min(multiple_gaussians_comparisons_all_points(params, data, initial_points, ndt))
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(loss)
        print(i)

    return params, evol



@partial(jax.jit, static_argnames=("ndt",))
def second_descent(params, data, initial_points, ndt=50, iter=100, lr=0.005):
    def gradient_step(t, params):
        grad_grad, grad_rot = grad_cost(params, data, initial_points, ndt)

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        params = (new_params_grad, new_params_rot)

        return params

    params = jax.lax.fori_loop(0, iter, gradient_step, params)

    return params

@partial(jax.jit, static_argnames=("ndt",))
def batch_second_descent(params, data, initial_points, ndt=50, iter=100, lr=0.005):
    return jax.vmap(second_descent, in_axes=(0, None, None, None, None, None))(params, data, initial_points, ndt, iter, lr)

def ALLtrajec2(params, data, initial_points, maxiter, ndt=50, verbose_dt=100, lr=0.005):
    pbar = tqdm.tqdm(total=maxiter)
    evol = []

    for i in range(maxiter//verbose_dt):
        params = batch_second_descent(params, data, initial_points, ndt, verbose_dt, lr)
        loss = jnp.min(multiple_gaussians_comparisons_all_points(params, data, initial_points, ndt))
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(loss)
        print(i)

    return params, evol

@partial(jax.jit, static_argnames=("ndt",))
def multiple_gaussians_comparisons_all_points(params, data, initial_points, ndt=50):
    return jax.vmap(compare_trajectories_all_points, in_axes=(0, None, None, None))(params, data, initial_points, ndt)


