from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import kl_div, entr

from .dynamics import _integrate, init_cell
from .types import LandscapeDynamic, LandscapeStatic

from .utils import rescale


# Here are different fitness functions to choose from

def origin_leading_fitness(traj, states, dynamic: LandscapeDynamic, loss_params=None):
    return jnp.sum(traj[:, :, -1] ** 2, axis=(0, 1))


def biseparating_fitness(traj, states, dynamic: LandscapeDynamic, loss_params=None):
    temperature, mu, lam, q_prob = loss_params

    final_coord = traj[:, :, -1]
    module_coord = jnp.stack((dynamic.module.x, dynamic.module.y))

    dist_squared = jnp.sum((final_coord.T[:, :, None] - module_coord) ** 2, axis=1)
    weights = jnp.exp(-dist_squared / temperature)
    P = weights / jnp.sum(weights, axis=1, keepdims=True)

    p_prob = jnp.mean(P, axis=0)

    loss_uniform = jnp.mean(
        kl_div(p_prob, jnp.array(q_prob))
    )

    uniform = jnp.ones_like(P) / P.shape[1]

    loss_particle = -jnp.mean(
    jnp.sum(kl_div(P, uniform), axis=1)
    )

    return (
        mu * loss_uniform + lam* loss_particle
    )

# Faut jiter ?
def drosophile_fitness(traj, states, decoded_traj, dynamic, decoder, data, fitness_params):
    #traj of shape (2,n,nt)
    #states of shape(n_module,n,nt)
    #decoded_traj of shape (4,n,nt)
    #data of shape (4,N,T)
    mu, lam = fitness_params
    decoded_traj = rescale(decoded_traj,data.shape[1],data.shape[2])
    loss_traj = jnp.mean((decoded_traj-traj)**2)
    loss_line = jnp.mean(jnp.sum(jnp.diff(traj[:,:,0],axis=1),axis=0))

    return (mu*loss_traj + lam*loss_line
    #+rho*loss_encoder_decoder    
    ) 


# ==========================================================
# Loss to compare point clouds
# ==========================================================
 
 
# Kernels

def rbf_kernel(X: jnp.ndarray, Y: jnp.ndarray, bandwidth: float = 1.0) -> jnp.ndarray:
    """RBF kernel (Gaussian) : k(x,y) = exp(-||x-y||² / (2σ²))."""
    # X : (n, d), Y : (m, d) 
    # returns : (n, m)
    diff = X[:, None, :] - Y[None, :, :]          # (n, m, d)
    sq_dist = jnp.sum(diff ** 2, axis=-1)          # (n, m)
    return jnp.exp(-sq_dist / (2.0 * bandwidth ** 2))
 
 
def multiscale_rbf_kernel(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    bandwidths: tuple = (0.5, 1.0, 2.0, 5.0),
) -> jnp.ndarray:
    """Sum of RBF kernels with multiple scales"""
    diff = X[:, None, :] - Y[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    K = sum(jnp.exp(-sq_dist / (2.0 * bw ** 2)) for bw in bandwidths)
    return K / len(bandwidths)
 

def linear_kernel(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """linear kernel : k(x,y) = <x,y>."""
    return X @ Y.T
 
 
# ---------------------------------------------------------------------------
# Biased mmd
# ---------------------------------------------------------------------------
 
def mmd(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    kernel: Callable = rbf_kernel,
    **kernel_kwargs,
) -> jnp.ndarray:
    """
    MMD²(X,Y) = E[k(x,x')] - 2 E[k(x,y)] + E[k(y,y')]
 
    Args:
        X : (n, d) — samples from P
        Y : (m, d) — samples from Q
        kernel : kernel function
        **kernel_kwargs : kernel parameters
 
    Returns:
        mmd2 : float (MMD²)
    """
    K_XX = kernel(X, X, **kernel_kwargs)
    K_YY = kernel(Y, Y, **kernel_kwargs)
    K_XY = kernel(X, Y, **kernel_kwargs)
 
    mmd2 = K_XX.mean() - 2.0 * K_XY.mean() + K_YY.mean()
    return mmd2

def mmd_traj(
    model_traj: jnp.ndarray,
    target_traj: jnp.ndarray,
    kernel: Callable = rbf_kernel,
    **kernel_kwargs,
) -> jnp.ndarray:
    """
    model_traj : array (d, n_cells, timepoints)
    target_traj : array (d, n_cells, timepoints)

    we need to apply mmd to each sub array of size (d, n_cells)
    
    """

    model_traj_trans = model_traj.transpose(2, 1, 0) # (timepoints, n_cells, d)
    target_traj_trans = target_traj.transpose(2, 1, 0) # (timepoints, n_cells, d)

    vmmd = vmap(mmd)

    return jnp.sum(vmmd(model_traj_trans, target_traj_trans))