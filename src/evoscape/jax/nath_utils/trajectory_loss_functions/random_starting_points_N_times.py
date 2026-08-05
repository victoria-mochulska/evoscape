import jax
import jax.numpy as jnp
from evoscape.jax.nath_utils.utilities import *
from functools import partial
import jax.lax as lax





################### CHOOSING M RANDOM PLACE SIMULATING N TIME POINT ON A RANDOM POINT TIMES AND COMPARING ALL/N #############################  
# Fixed N point time, CHOSE M RANDOM PLACE AND TIME AND SIMULATING FOR N. Then compare the point

# return an array of len(key) indices, points and end
@partial(jax.jit, static_argnames=("num_pts_iter", "length_traj"))
def choose_point(key, data, data_okay, num_pts_iter, length_traj):
    shape_space, shape_time, coord = data_okay.shape

    flat_indices = jax.random.choice(key, shape_space*shape_time, shape=(num_pts_iter,), replace=False)
    space_coords, time_coords = jnp.unravel_index(flat_indices, (shape_space, shape_time))
    time_indices = time_coords[:, None] + jnp.arange(length_traj)[None, :]
    initial_points = data[space_coords, time_coords, :]
    traj = data[space_coords[:, None], time_indices, :]
    return (space_coords, time_coords), initial_points, traj




# params is a tuple of (2) of [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("length_traj", "Delta_t", "ndt"))
def one_trajectory_loop_part(params, initial_points, length_traj, Delta_t, ndt=50):

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
        length=length_traj-1
    )

    traj = jnp.concatenate(
    [initial_points[None, :], traj],
    axis=0
    )

    return final_pos, traj


# params is a tuple of (2) of [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("length_traj", "Delta_t", "ndt"))
def one_trajectory_loop_part_time(params, initial_points, time_coord, length_traj, Delta_t, frames, ndt=50):
    dt = Delta_t / ndt

    grad_params, rot_params = params

    def outer_step(carry, frame):
        frame = time_coord + frame

        pos = carry

        # parameters for this frame
        grad_t = grad_params[:, frame, :]
        rot_t  = rot_params[:, frame, :]

        def inner_step(_, pos):
            pos_x = pos[0]
            pos_y = pos[1]

            v = (
                JAXflow_grad_mixture(pos_x, pos_y, grad_t)
                + JAXflow_rot_mixture(pos_x, pos_y, rot_t)
            )

            return pos + dt * v

        pos = lax.fori_loop(
            0,
            ndt,
            inner_step,
            pos,
        )

        return pos, pos

    final_pos, traj = lax.scan(
        outer_step,
        initial_points,
        frames,
    )

    traj = jnp.concatenate(
        [initial_points[None], traj],
        axis=0,
    )

    return final_pos, traj


@partial(jax.jit, static_argnames=("length_traj", "Delta_t", "ndt"))
def all_end_part(params, initial_points, length_traj, Delta_t, ndt=50):
    return jax.vmap(one_trajectory_loop_part, in_axes=(None, 0, None, None, None))(params, initial_points, length_traj, Delta_t, ndt)  #need init points = (T, 2) will give (N, T, 2)


# parmas is still a tuple of (2) with shape [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("length_traj", "Delta_t","ndt"))
def compare_end_part(params, traj, initial_points, length_traj, Delta_t, ndt=50):


    _, predictions_trajectories = all_end_part(params, initial_points, length_traj, Delta_t, ndt)

    diff = predictions_trajectories - traj
    norm = jnp.linalg.norm(diff, axis=-1)
    return jnp.mean(norm)

@partial(jax.jit, static_argnames=("length_traj", "Delta_t", "ndt"))
def all_end_part_time(params, initial_points, time_coords, length_traj, Delta_t, frames, ndt=50):
    frames = jnp.arange(length_traj - 1)

    return jax.vmap(
        one_trajectory_loop_part_time,
        in_axes=(None, 0, 0, None, None, None, None)
    )(params, initial_points, time_coords, length_traj, Delta_t, frames, ndt)
    #return jax.vmap(one_trajectory_loop_part_time, in_axes=(None, 0, 0, None, None, None))(params, initial_points, time_coords, length_traj, Delta_t, ndt)  #need init points = (T, 2) will give (N, T, 2)


# parmas is still a tuple of (2) with shape [N, 6] and [M, 6]
@partial(jax.jit, static_argnames=("length_traj", "Delta_t","ndt"))
def compare_end_part_time(params, traj, initial_points, time_coords, length_traj, Delta_t, ndt=50):


    _, predictions_trajectories = all_end_part_time(params, initial_points, time_coords, length_traj, Delta_t, ndt)

    diff = predictions_trajectories - traj
    norm = jnp.linalg.norm(diff, axis=-1)
    return jnp.mean(norm)

grad_cost4 = jax.grad(compare_end_part, argnums=(0))

@partial(jax.jit, static_argnames=("length_traj", "Delta_t","ndt"))
def multiple_gaussians_comparisons_part(params, traj, initial_points, length_traj, Delta_t, ndt=50):
    return jax.vmap(compare_end_part, in_axes=(0, None, None, None, None, None))(params, traj, initial_points, length_traj, Delta_t, ndt)

@partial(jax.jit, static_argnames=("num_pts_iter" ,"length_traj", "Delta_t", "ndt", "iter", "lr"))
def fifth_descent(params, data, data_okay, keys, num_pts_iter, length_traj, Delta_t, ndt=50, iter=100, lr=0.005):
    init_state = (params[0], params[1])
    def gradient_step(t, params):
        key = keys[t]

        indices, initial_points, traj = choose_point(key, data, data_okay, num_pts_iter, length_traj)
        
        grad_grad, grad_rot = grad_cost4(params, traj, initial_points, length_traj, Delta_t, ndt)

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        return (new_params_grad, new_params_rot)

    params = jax.lax.fori_loop(0, iter, gradient_step, init_state)

    return params

@partial(jax.jit, static_argnames=("num_pts_iter", "length_traj", "Delta_t", "ndt", "iter", "lr"))
def batch_fifth_descent(params, data, data_okay, keys, num_pts_iter, length_traj, Delta_t, ndt=50, iter=100, lr=0.005):
    return jax.vmap(fifth_descent, in_axes=(0, None, None, None, None, None, None, None, None, None))(params, data, data_okay, keys, num_pts_iter, length_traj, Delta_t, ndt, iter, lr)


def RAND_trajec(params, data, time, percent_data, length_traj, base_key, maxiter, ndt=50, verbose_dt=100, lr=0.005):
    pbar = tqdm.tqdm(total=maxiter)
    evol = []
    Delta_t = (time-1)/time

    data_okay = data[:, :-length_traj ,:]

    shape_space, shape_time, coord = data_okay.shape
    print("shapes", shape_space, shape_time)
    num_pts_iter =  int((shape_space * shape_time) * (percent_data/100))
    print("num pts", num_pts_iter)

    keys = jax.random.split(base_key, num=maxiter)
    verbose_keys = jax.random.split(base_key, num=maxiter//verbose_dt)


    for i in range(maxiter//verbose_dt):
        params = batch_fifth_descent(params, data, data_okay, keys, num_pts_iter, length_traj, Delta_t, ndt, verbose_dt, lr)
        indices, initial_points, traj = choose_point(verbose_keys[i], data, data_okay, num_pts_iter, length_traj)
        loss = jnp.min(multiple_gaussians_comparisons_part(params, traj, initial_points, length_traj, Delta_t, ndt=50))
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(loss)

    return params, evol

