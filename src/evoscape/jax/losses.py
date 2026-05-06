from functools import partial

import jax.numpy as jnp

from jax import jit
from jax.scipy.special import kl_div, entr

from .dynamics import _integrate, init_cell
from .types import LandscapeDynamic, LandscapeStatic


# Here is different fitness functions to choose from

@partial(jit, static_argnames=("get_states", "n", "cell_noise", "nt", "ndt", "regime"))
def origin_leading_fitness(dynamic: LandscapeDynamic, static: LandscapeStatic, n, cell_noise, t0, tf, nt, ndt, noise, key, regime, get_states):
    key, q_flat = init_cell(key, n, dynamic.init_cond, noise=cell_noise)
    _, traj, _ = _integrate(key, q_flat, t0, tf, nt, ndt, noise, dynamic, static, regime, get_states)

    return jnp.sum(traj[:, :, -1] ** 2, axis=(0, 1)) / n


@partial(jit, static_argnames=("loss_params", "n", "cell_noise", "nt", "ndt", "get_states", "regime"))
def biseparating_fitness(dynamic: LandscapeDynamic, static: LandscapeStatic, n, cell_noise, t0, tf, nt, ndt, noise, key, regime, get_states, loss_params):
    key, q_flat = init_cell(key, n, dynamic.init_cond, noise=cell_noise)
    _, traj, _ = _integrate(key, q_flat, t0, tf, nt, ndt, noise, dynamic, static, regime, get_states)

    temperature, lam, q_prob = loss_params
    final_coord = traj[:, :, -1]

    module_coord = jnp.stack((dynamic.module.x, dynamic.module.y))

    dist_squared = jnp.sum((final_coord.T[:, :, None] - module_coord) ** 2, axis=1)

    weights = jnp.exp(-dist_squared / temperature)

    P = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-8)

    p_j = jnp.mean(P, axis=0)

    loss_uniform = jnp.mean(
        kl_div(p_j, jnp.array(q_prob)) + kl_div(jnp.array(q_prob), p_j)
    )

    return (
        loss_uniform
        + 0.0 * lam
        + 0.0 * jnp.sum(entr(P))
    )
