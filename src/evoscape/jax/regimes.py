import jax.numpy as jnp

from jax.tree_util import Partial

from .types import LandscapeStatic, AnisotropicLandscapeStatic


def mr_const(t, a, s):
    t = jnp.asarray(t)
    a = jnp.atleast_1d(a)
    s = jnp.atleast_1d(s)
    return jnp.broadcast_to(s[0], t.shape), jnp.broadcast_to(a[0], t.shape)


def mr_sigmoid(t, a, s, t_list, tau):
    t = jnp.asarray(t)
    a = jnp.atleast_1d(a)
    s = jnp.atleast_1d(s)
    tanh = jnp.tanh((t - t_list[0]) / 2. / tau)
    a_t = a[0] + (a[1] - a[0]) / 2. * (1 + tanh)
    s_t = s[0] + (s[1] - s[0]) / 2. * (1 + tanh)
    return s_t, a_t


def mr_piecewise(t, a, s, t_list):
    a = jnp.atleast_1d(a)
    s = jnp.atleast_1d(s)
    t = jnp.asarray(t)
    t_list = jnp.asarray(t_list)

    idx = jnp.sum(t > t_list)
    idx = jnp.clip(idx, 0, a.shape[0] - 1)

    return s[idx], a[idx]


def mr_current_regime(t, t_list):
    t = jnp.asarray(t)
    t_list = jnp.asarray(t_list)

    idx = jnp.sum(t >= t_list)
    idx = jnp.minimum(idx, t_list.shape[0])

    return idx


def mr_linear_2signals(t, a, s, signal0, signal1):
    a = jnp.atleast_1d(a)
    s = jnp.atleast_1d(s)
    signal0 = signal0(t)
    signal1 = signal1(t)
    a_t = a[0] + a[1] * signal0 + a[2] * signal1
    s_t = s[0] + s[1] * signal0 + s[2] * signal1
    a_t = jnp.maximum(a_t, 0.0)
    s_t = jnp.maximum(s_t, 0.2)
    a_t = jnp.minimum(a_t, 20.0)
    s_t = jnp.minimum(s_t, 1.2)
    return s_t, a_t


regimes = (
    mr_const,
    mr_sigmoid,
    mr_piecewise,
    mr_linear_2signals,
)


def wrapped_regime(static: LandscapeStatic, signal_param=None):
    """
    All functions need to be wrapped with Partial (the one from jax.tree_util, not functools.partial)
    That way, they can be seen as pytrees by jax. This is important because that way we can declare
    regimes as nnx.data in landscape_flax (nnx.data would throw an error if we didn't wrap with nnx.data)
    
    """
    regime_number = static.regime_id
    if regime_number == 0:
        return Partial(regimes[regime_number])
    t_list = static.morphogen_times

    # All modules are supposed to have the same tau
    if regime_number == 1:
        return Partial(regimes[regime_number], t_list=t_list, tau=static.module.tau[0])

    if regime_number == 2:
        return Partial(regimes[regime_number], t_list=t_list)

    if regime_number == 3:
        signal0, signal1 = signal_param
        return Partial(regimes[regime_number], signal0=signal0, signal1=signal1)
    else:
        raise NotImplementedError





def mr_const_aniso(t, a, sx, sy, th):
    t = jnp.asarray(t)
    a = jnp.atleast_1d(a)
    sx = jnp.atleast_1d(sx)
    sy = jnp.atleast_1d(sy)
    th = jnp.atleast_1d(th)
    return jnp.broadcast_to(sx[0], t.shape), jnp.broadcast_to(sy[0], t.shape), jnp.broadcast_to(a[0], t.shape), jnp.broadcast_to(th[0], t.shape)


def mr_sigmoid_aniso(t, a, sx, sy, th, t_list, tau):
    t = jnp.asarray(t)
    a = jnp.atleast_1d(a)
    sx = jnp.atleast_1d(sx)
    sy = jnp.atleast_1d(sy)
    th = jnp.atleast_1d(th)
    tanh = jnp.tanh((t - t_list[0]) / 2. / tau)
    a_t = a[0] + (a[1] - a[0]) / 2. * (1 + tanh)
    sx_t = sx[0] + (sx[1] - sx[0]) / 2. * (1 + tanh)
    sy_t = sy[0] + (sy[1] - sy[0]) / 2. * (1 + tanh)
    th_t = th[0] + (th[1] - th[0]) / 2. * (1 + tanh)

    return sx_t, sy_t, a_t, th_t


def mr_piecewise_aniso(t, a, sx, sy, th, t_list):
    a = jnp.atleast_1d(a)
    sx = jnp.atleast_1d(sx)
    sy = jnp.atleast_1d(sy)
    th = jnp.atleast_1d(th)
    t = jnp.asarray(t)
    t_list = jnp.asarray(t_list)

    idx = jnp.sum(t > t_list)
    idx = jnp.clip(idx, 0, a.shape[0] - 1)

    return sx[idx], sy[idx], a[idx], th[idx]


def mr_current_regime_aniso(t, t_list):
    t = jnp.asarray(t)
    t_list = jnp.asarray(t_list)

    idx = jnp.sum(t >= t_list)
    idx = jnp.minimum(idx, t_list.shape[0])

    return idx


def mr_linear_2signals_aniso(t, a, sx, sy, th, signal0, signal1):
    a = jnp.atleast_1d(a)
    sx = jnp.atleast_1d(sx)
    sy = jnp.atleast_1d(sy)
    th = jnp.atleast_1d(th)
    signal0 = signal0(t)
    signal1 = signal1(t)
    a_t = a[0] + a[1] * signal0 + a[2] * signal1
    sx_t = sx[0] + sx[1] * signal0 + sx[2] * signal1
    sy_t = sy[0] + sy[1] * signal0 + sy[2] * signal1
    th_t = th[0] + th[1] * signal0 + th[2] * signal1
    a_t = jnp.maximum(a_t, 0.0)
    sx_t = jnp.maximum(sx_t, 0.2)
    sy_t = jnp.maximum(sy_t, 0.2)
    a_t = jnp.minimum(a_t, 20.0)
    sx_t = jnp.minimum(sx_t, 1.2)
    return sx_t, sy_t, a_t, th_t


regimes_aniso = (
    mr_const_aniso,
    mr_sigmoid_aniso,
    mr_piecewise_aniso,
    mr_linear_2signals_aniso,
)


def wrapped_regime(static: AnisotropicLandscapeStatic, signal_param=None):
    """
    All functions need to be wrapped with Partial (the one from jax.tree_util, not functools.partial)
    That way, they can be seen as pytrees by jax. This is important because that way we can declare
    regimes as nnx.data in landscape_flax (nnx.data would throw an error if we didn't wrap with nnx.data)
    
    """
    regime_number = static.regime_id
    if regime_number == 0:
        return Partial(regimes_aniso[regime_number])
    t_list = static.morphogen_times

    # All modules are supposed to have the same tau
    if regime_number == 1:
        return Partial(regimes_aniso[regime_number], t_list=t_list, tau=static.module.tau[0])

    if regime_number == 2:
        return Partial(regimes_aniso[regime_number], t_list=t_list)

    if regime_number == 3:
        signal0, signal1 = signal_param
        return Partial(regimes_aniso[regime_number], signal0=signal0, signal1=signal1)
    else:
        raise NotImplementedError
