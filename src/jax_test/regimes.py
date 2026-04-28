import jax.numpy as jnp
from .landscape_pytree_class import LandscapeDynamic, LandscapeStatic
from jax.tree_util import Partial
def mr_const(t, a, s):
    a = jnp.atleast_1d(a)
    s = jnp.atleast_1d(s)
    return jnp.broadcast_to(s[0], t.shape), jnp.broadcast_to(a[0], t.shape)

def mr_sigmoid(t, a, s, t_list,tau):
    t = jnp.asarray(t)
    a = jnp.atleast_1d(a)
    s = jnp.atleast_1d(s)
    tanh = jnp.tanh((t[:, None] - t_list[0]) / 2. / tau)
    a_t = a[0] + (a[1] - a[0]) / 2. * (1 + tanh)
    s_t = s[0] + (s[1] - s[0]) / 2. * (1 + tanh)
    return s_t, a_t

def mr_piecewise(t, a, s, t_list):
    a = jnp.atleast_1d(a)
    s = jnp.atleast_1d(s)
    t = jnp.atleast_1d(t)[:, None]

    t_list = jnp.asarray(t_list)

    mask = t >= t_list

    idx = jnp.sum(mask,axis=1)

    idx = jnp.clip(idx, 0, a.shape[0] - 1)

    return s[idx], a[idx]


def mr_current_regime(t,t_list):

    t = jnp.atleast_1d(t)[:, None]

    t_list = jnp.asarray(t_list)

    mask = t >= t_list

    # convert boolean mask → index of last True
    # (cumulative sum trick)
    idx = jnp.sum(mask,axis=1)

    idx = jnp.minimum(idx, t_list.shape[0])

    return idx

def mr_linear_2signals(t, a, s,signal0, signal1):
    a = jnp.atleast_1d(a)
    s = jnp.atleast_1d(s)
    signal0= signal0(t)
    signal1 = signal1(t)
    a_t = a[0] + a[1]*signal0 + a[2]*signal1
    s_t = s[0] + s[1]*signal0 + s[2]*signal1
    # Clip parameters to the allowed range
    # Negative a allowed
    a_t = jnp.maximum(a_t, 0.0)
    s_t = jnp.maximum(s_t, 0.2)
    # a_t = np.minimum(a_t, 16.0)
    # s_t = np.minimum(s_t, 1.5)
    a_t = jnp.minimum(a_t, 20.0)
    s_t = jnp.minimum(s_t, 1.2)
    return s_t, a_t

regimes = (
    mr_const,
    mr_sigmoid,
    mr_piecewise,
    mr_linear_2signals
)

def wrapped_regime(static: LandscapeStatic, signal_param=None):
    regime_number = static.regime
    if regime_number == 0:
        return regimes[regime_number]
    t_list = static.morphogen_times

    # All modules are supposed to have the same tau
    if regime_number == 1:
        return Partial(regimes[regime_number], t_list = t_list, tau= static.module.tau[0])
    
    if regime_number == 2:
         return Partial(regimes[regime_number], t_list = t_list)
    
    if regime_number == 3:
        signal0, signal1 = signal_param
        return Partial(regimes[regime_number], t_list = t_list, signal0=signal0,signal1=signal1)
    else:
        NotImplemented
    
