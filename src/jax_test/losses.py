import jax.numpy as jnp

from jax import jit
from .dynamics import _integrate
from .utils import init_cell
from .landscape_pytree_class import LandscapeDynamic, LandscapeStatic

# Here is different fitness functionx to choose from

@jit(static_argnames=('get_states','n','nt','ndt', 'regime'))
def origin_leading_fitness(dynamic : LandscapeDynamic, static : LandscapeStatic, n, t0, tf, nt, ndt, noise, key, regime, get_states):

    key, q_flat = init_cell(key,n,dynamic.init_cond,noise=noise)
    _, traj, _=_integrate(key, q_flat, t0, tf, nt, ndt, noise, dynamic, static, regime, get_states)
    
    return jnp.sum(traj[:,:,-1]**2,axis=(0,1))/n


@jit(static_argnames=('loss_params','n', 'nt','ndt', 'get_states', 'regime'))
def biseparating_fitness(dynamic : LandscapeDynamic,static : LandscapeStatic, n, t0, tf, nt, ndt, noise, key, regime, get_states, loss_params):
    key, q_flat = init_cell(key,n,dynamic.init_cond,noise=noise)
    _, traj, _=_integrate(key, q_flat, t0, tf, nt, ndt, noise, dynamic, static, regime, get_states)

    T1, T2, λ, μ = loss_params
    final_coord = traj[:,:,-1]

    module_coord = jnp.stack((dynamic.module.x,dynamic.module.y))
    m = module_coord.shape[1]
    dist_squared = jnp.sum((final_coord.T[:, :, None] - module_coord)**2, axis=1)

    weights = jnp.exp(-dist_squared / T1)

    P = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-8)

    p_i = jnp.mean(P, axis=0)

    loss_uniform = jnp.sum((p_i - 1/m)**2)

    loss_cell = -jnp.mean(jnp.max(jnp.log(weights + 1e-8), axis=1))

    dist_modules = jnp.sum((module_coord.T[:, :, None] - module_coord)**2, axis=1)

    loss_module = jnp.sum(jnp.exp(-dist_modules / T2))

    return (
        loss_cell/n +
        # λ*loss_module - 
        μ*loss_uniform
        )