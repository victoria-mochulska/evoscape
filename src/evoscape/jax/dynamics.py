from functools import partial

import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrnd

from jax import jit, vmap


# Initialization function used in the fitness function, we let the model choose where the cells should start in optimization
def init_cell(key, n, init_cond, noise):
    key, subkey = jrnd.split(key)
    return key, init_cond[:, None] + noise * jrnd.normal(subkey, shape=(2, n))


# Base flow function to work with
# @jit
def _flow(q_flat, xs, ys, sig, a, Js, A0, x0):
    x, y = q_flat
    xr = x[None] - xs[:, None]
    yr = y[None] - ys[:, None]
    r = jnp.sqrt(xr ** 2 + yr ** 2)
    nonzero_sig = jnp.where(sig == 0, 1, sig)
    w = a[:, None] * jnp.exp(-0.5 * (r / nonzero_sig[:, None]) ** 2)
    dx = Js[:, 0, 0, None] * xr + Js[:, 0, 1, None] * yr
    dy = Js[:, 1, 0, None] * xr + Js[:, 1, 1, None] * yr

    dX = A0 * (-(x - x0[0]) ** 3) + jnp.sum(w * dx, axis=0)
    dY = A0 * (-(y - x0[1]) ** 3) + jnp.sum(w * dy, axis=0)
    derivs = jnp.stack((dX, dY), axis=0)

    return derivs


# Function to compute a and s for a given time t and a given regime
# @partial(jit, static_argnames=("regime",))
def get_current_par(t, dynamic, regime):
    a = dynamic.module.a
    s = dynamic.module.s

    def compute(i):
        return regime(t, a[i], s[i])

    s_t, a_t = vmap(compute)(jnp.arange(a.shape[0]))
    return a_t * s_t ** 2, s_t, a_t


# Get_flow depending on time: the use of get_current_par is recquired !
# @partial(jit, static_argnames=("regime",))
def get_flow(t, coordinate, dynamic, static, regime):
    a0 = static.A0
    xs = dynamic.module.x
    ys = dynamic.module.y
    _, sig, a = get_current_par(t, dynamic, regime)
    x0 = static.x0
    Js = static.module.J
    return _flow(coordinate, xs, ys, sig, a, Js, a0, x0)


# Base integration function to work with
# @partial(jit, static_argnames=("nt", "ndt", "get_states", "regime"))
def _integrate(key, y0, t0, tf, nt, ndt, noise, dynamic, static, regime, get_states):
    dt = (tf - t0) / (nt - 1) / ndt
    sqrt_dt = jnp.sqrt(dt)

    def outer_step(carry, _):
        key, t, y = carry
        key, subkey = jrnd.split(key)
        etas = jrnd.normal(subkey, (ndt,) + y.shape, dtype=y.dtype)

        def inner_step(carry, eta):
            t, y = carry
            deriv = get_flow(t, y, dynamic, static, regime)
            y = y + deriv * dt + noise * eta * sqrt_dt
            t = t + dt
            return (t, y), None

        (t, y), _ = lax.scan(inner_step, (t, y), etas)
        s = get_states(t, y, dynamic, static, regime)
        return (key, t, y), (y, s)

    state0 = get_states(t0, y0, dynamic, static, regime)
    (key_final, _, _), (ys, states) = lax.scan(outer_step, (key, t0, y0), None, length=nt - 1)

    traj = jnp.concatenate([y0[None], ys], axis=0).transpose(1, 2, 0)
    states = jnp.concatenate([state0[None], states], axis=0).T
    return key_final, traj, states


# Definition of get_states with integration of regime function which was previously defined.
# @partial(jit, static_argnames=("measure", "prob_threshold", "abs_threshold", "regime"))
def get_cell_states(t, coordinate, dynamic, static, regime, measure="gaussian", prob_threshold=0.0, abs_threshold=0.0):
    """
    Return cell states given cell coordinates. Assignent based on a chosen distance measure, can depend on time or signals.
    :param t: float, timepoint,
    :param coordinate: array of shape (2, n) where n is the number of cells
        (optional, can use the current coordinates stored in landscape)
    :param measure: 'dist' - base on Euclidean distance to modules, same as get_cell_states_static.
        'gaussian' - based on a gaussian mixture model, taking into account time-dependent module size.
    :return: states - array of length n of ints
    """
    states = None
    n_modules = len(static.module.J)
    if measure == "dist":
        dist = jnp.linalg.norm(coordinate.T[:, :, None] - jnp.array((dynamic.module.x, dynamic.module.y)), axis=1)
        states = jnp.argmin(dist, axis=1)
    elif measure == "gaussian":
        _, st, at = get_current_par(t, dynamic, regime)
        dist = jnp.sum(
            (coordinate.T[:, :, None] - jnp.array([dynamic.module.x, dynamic.module.y])) ** 2,
            axis=1,
        )

        mask = (st == 0) | (at == 0)
        prob_main = jnp.where(
            mask[None, :],
            0.0,
            jnp.exp(-dist / (2 * st ** 2)) / (st ** 2),
        )

        abs_col = jnp.full((prob_main.shape[0], 1), abs_threshold)
        prob = jnp.concatenate([prob_main, abs_col], axis=1)
        prob = (prob.T / jnp.sum(prob, axis=1)).T
        if abs_threshold == 0:
            prob = prob.at[:, -1].set(prob_threshold)
        states = jnp.argmax(prob, axis=1)
        states = jnp.where(states == n_modules, -1, states)
    return states


#  Check/update
# @partial(jit, static_argnames=("regime",))
def state_probs(t, coordinate, dynamic, static, regime):
    _, st, at = get_current_par(t, dynamic, regime)
    dist = jnp.sum(
        (coordinate.T[:, :, None] - jnp.array([dynamic.module.x, dynamic.module.y])) ** 2,
        axis=1,
    )

    mask = (st == 0) | (at == 0)
    nonzero_st = jnp.where(mask, 1.0, st)
    gaussian_values = jnp.where(
        mask[None, :],
        0.0,
        jnp.exp(-dist / (2 * nonzero_st ** 2)) / (nonzero_st ** 2),
    )
    sum_values = jnp.sum(gaussian_values, axis=1, keepdims=True)
    safe_sum = jnp.where(sum_values == 0, 1.0, sum_values)
    probs = gaussian_values / safe_sum

    return probs


#
# @partial(jit, static_argnames=("regime",))
def compute_potentials(t, q, dynamic, static, regime):
    q = jnp.asarray(q)
    grid_shape = q.shape[1:]
    q_flat = q.reshape(2, -1)
    x, y = q_flat

    _, sig_list, a_list = get_current_par(t, dynamic, regime)
    xs = dynamic.module.x
    ys = dynamic.module.y
    Js = static.module.J
    A0 = static.A0
    x0 = static.x0

    if xs.shape[0] == 0:
        pot = A0 / 4 * ((x - x0[0]) ** 4 + (y - x0[1]) ** 4)
        return pot.reshape(grid_shape), jnp.zeros(grid_shape)

    sign = jnp.where(Js[:, 0, 0] != 0, Js[:, 0, 0], Js[:, 1, 0])
    curl = jnp.where(Js[:, 0, 0] == 0, 1.0, 0.0)

    xr = x[None] - xs[:, None]
    yr = y[None] - ys[:, None]

    r = jnp.sqrt(xr ** 2 + yr ** 2)
    nonzero_sig = jnp.where(sig_list == 0, 1.0, sig_list)
    w = a_list[:, None] * jnp.exp(-0.5 * (r / nonzero_sig[:, None]) ** 2)

    coefs = sign * (1 - curl) * sig_list ** 2
    coefs_rot = sign * curl * sig_list ** 2
    pot = jnp.sum(w * coefs[:, None], axis=0) + A0 / 4 * ((x - x0[0]) ** 4 + (y - x0[1]) ** 4)
    pot_rot = jnp.sum(w * coefs_rot[:, None], axis=0)

    return pot.reshape(grid_shape), pot_rot.reshape(grid_shape)
