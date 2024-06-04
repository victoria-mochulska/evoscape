import numpy as np


@np.vectorize
def const_v(t, V0, **kwargs):
    return V0


def mr_sigmoid_v(t, V0, V1, t0, tau, **kwargs):
    return V0 + (V1 - V0) / 2 * (1 + np.tanh((t - t0) / 2. / tau))


def mr_piecewise_v(t, V0, V1, V2, t0, t1, **kwargs):
    return np.piecewise(t, [t < t0, (t >= t0) & (t < t1), t >= t1], [V0, V1, V2])


def mr_sigmoid(t, a, s, t0, tau, **kwargs):
    if a.size == 2:
        a_t = a[0] + (a[1] - a[0]) / 2. * (1 + np.tanh((t - t0) / 2. / tau))
        s_t = s[0] + (s[1] - s[0]) / 2. * (1 + np.tanh((t - t0) / 2. / tau))
    else:
        raise NotImplementedError
    return s_t, a_t


def mr_piecewise(t, a, s, t0, t1, **kwargs):
    a_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), t >= t1], list(a))
    s_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), t >= t1], list(s))
    return s_t, a_t
