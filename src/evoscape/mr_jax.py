from jax.tree_util import tree_map

import jax
import jax.numpy as jnp

def mr_const_jax(t, module_params):
    
    xs = module_params["xs"]
    ys = module_params["ys"]
    sig_list = module_params["sig_list"]
    a_list = module_params["a_list"]

    return xs, ys, sig_list, a_list

def mr_sigmoid_jax(t, module_params, t0 = 0., tau = 1.):
    xs = module_params["xs"]
    ys = module_params["ys"]
    a_array = module_params["a_list"]
    sig_array = module_params["sig_list"]

    tanh = jnp.tanh((t - t0) / (2. * tau))
    a_t = a_array[:,0] + (a_array[:,1] - a_array[:,0]) / 2. * (1 + tanh)
    s_t = sig_array[:,0] + (sig_array[:,1] - sig_array[:,0]) / 2. * (1 + tanh)

    return xs, ys, s_t, a_t

def mr_piecewise_jax(t, module_params, t_list):
    xs = module_params["xs"]
    ys = module_params["ys"]
    a_array = module_params["a_list"]
    sig_array = module_params["sig_list"]

    idx = jnp.sum(t > t_list)

    a_t = a_array[:, idx]
    s_t = sig_array[:, idx]

    return xs, ys, s_t, a_t


