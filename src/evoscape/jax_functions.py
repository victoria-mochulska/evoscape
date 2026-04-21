
from functools import partial
import jax.lax as lax
import jax.random as jrnd
import jax.numpy as jnp
from jax import jit
import jax

@partial(jit, static_argnames=['return_potentials'])
def _flow(t, q_flat, module_params, module_infos, return_potentials):
    """
    parameters :
    q_flat : the cells coordinates (2,n)

    xs, ys : coordinates of the different modules (m,), (m,)
    sig_list : array containing the sigmas of the gaussian clusters so it should be (m,). 
    a_list : array containing the intensity of the gaussian clusters. Same as before, the size is (m,).

    sign : array containing signs of the modules (m,) 
    curl : array containing 1 if there is curl or 0 if there is not (m,)
    A0, x0 : float
    Js : jacobians of the modules

    return_potentials : bool

    returns :
    derivs : (2,n) array containing the derivative of each particle
    """

    ## converting the pytree into variables
    xs = module_params["xs"]
    ys = module_params["ys"]
    sig_list = module_params["sig_list"]
    a_list = module_params["a_list"]

    sign = module_infos["sign"]
    curl = module_infos["curl"]
    Js = module_infos["Js"]
    A0 = module_infos["A0"]
    x0 = module_infos["x0"] 

    nb_cells = q_flat.shape[1]
    a = a_list[:,None] * jnp.ones((1, nb_cells))
    sig = sig_list[:,None] * jnp.ones((1, nb_cells))

    ## the code after this remains the same
    x, y = q_flat[0], q_flat[1]
    xr = x[None] - xs[:,None]
    yr = y[None] - ys[:,None]

    r = jnp.sqrt(xr ** 2 + yr ** 2)
    w = a * jnp.exp(-0.5 * (r / sig) ** 2)

    dx = Js[:,:, 0, 0] * xr + Js[:,:, 0, 1] * yr
    dy = Js[:, :,1, 0] * xr + Js[:, :,1, 1] * yr

    dX = A0 * (-(x - x0[0]) ** 3) + jnp.sum(w * dx, axis=0)
    dY = A0 * (-(y - x0[1]) ** 3) + jnp.sum(w * dy, axis=0)
    derivs = jnp.stack((dX, dY), axis=0)

    if return_potentials:
        coefs = sign * (1-curl) * sig ** 2
        coefs_rot = (sign * curl) * sig ** 2
        pot = jnp.sum(w * coefs[:, None], axis=0) + A0 / 4 * ((x - x0[0]) ** 4 + (y - x0[1]) ** 4)
        pot_rot = jnp.sum(w * coefs_rot[:, None], axis=0)
        return derivs, pot, pot_rot
    return derivs


@partial(jit, static_argnames=['nt', 'ndt', 'get_states'])
def _integrate(key, y0, t0, tf, nt, ndt, noise, module_infos, module_params, get_states):
    """
    key : jax key, to create random numbers with jax
    tf, t0 : floats
    nt : number of points of a trajectory, int
    ndt : number of steps performed between points of the trajectory, int
    get_states(t, y) : a function that returns a (n,) array containing the state of each cell
    y0 : array (2, n) containing all the particles (also named q_flat sometimes)

    returns :
    traj : (2, n, nt) array : coordinate[i] of the cell[j] at datapoint[k]
    states : (n, nt) array : state of cell[i] at datapoint[k]
    """

    dt = (tf - t0) / (nt - 1) / ndt
    sqrt_dt = jnp.sqrt(dt)

    def outer_step(carry, _):
        key, t, y = carry
        key, subkey = jrnd.split(key)
        etas = jrnd.normal(subkey, (ndt,) + y.shape, dtype=y.dtype)

        def inner_step(carry, eta):
            t, y = carry
            deriv = _flow(t, y, module_params, module_infos, return_potentials=False)

            y = y + deriv * dt + noise * eta * sqrt_dt
            t = t + dt
            return (t, y), None

        (t, y), _ = lax.scan(inner_step, (t, y), etas)
        s = get_states(t, y)
        return (key, t, y), (y, s)

    state0 = get_states(t0, y0)
    (key_final, _, _), (ys, states) = lax.scan(outer_step, (key, t0, y0), None, length=nt - 1)

    traj = jnp.concatenate([y0[None], ys], axis=0).transpose(1, 2, 0)
    states = jnp.concatenate([state0[None], states], axis=0).T
    return key_final, traj, states

@jit
def state_probs(module_params, q):
    """
    returns :
    probs : (n, m) array containing the probability of cell[i] to be in state[j]
    """

    xs = module_params["xs"]
    ys = module_params["ys"]
    sig_list = module_params["sig_list"]
    a_list = module_params["a_list"]

    ## This code does not consider the case where a or sig is zero

    x, y = q[0], q[1]

    gaussian_values = jnp.exp(((x[:,None] - xs)**2 + (y[:,None] - ys)**2) / 2*sig_list**2) * a_list/(jnp.sqrt(2*jnp.pi)*sig_list)
    sum_values = jnp.sum(gaussian_values, axis=1)
    probs = gaussian_values / sum_values[:,None]

    return probs

