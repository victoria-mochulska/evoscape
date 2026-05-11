from functools import partial

import jax.numpy as jnp

from jax import jit
from jax.scipy.special import kl_div, entr

from .dynamics import _integrate, init_cell
from .types import LandscapeDynamic, LandscapeStatic


# Here is different fitness functions to choose from

def origin_leading_fitness(traj, states, dynamic: LandscapeDynamic, loss_params=None):
    return jnp.sum(traj[:, :, -1] ** 2, axis=(0, 1))


def biseparating_fitness(traj, states, dynamic: LandscapeDynamic, loss_params=None):
    temperature, lam, q_prob = loss_params

    final_coord = traj[:, :, -1]
    module_coord = jnp.stack((dynamic.module.x, dynamic.module.y))

    dist_squared = jnp.sum((final_coord.T[:, :, None] - module_coord) ** 2, axis=1)
    weights = jnp.exp(-dist_squared / temperature)
    P = weights / jnp.sum(weights, axis=1, keepdims=True)

    p_prob = jnp.mean(P, axis=0)

    loss_uniform = jnp.mean(
        kl_div(p_prob, jnp.array(q_prob)) + kl_div(jnp.array(q_prob), p_prob)
    )

    return (
        loss_uniform
        + 0.0 * lam
        + 0.0 * jnp.sum(entr(P))
    )

