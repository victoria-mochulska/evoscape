import numpy as np


@np.vectorize
def mr_const(t, a, s, **kwargs):
    return s[0], a[0]


def mr_sigmoid(t, a, s, t0, tau, **kwargs):
    if a.size == 2:
        a_t = a[0] + (a[1] - a[0]) / 2. * (1 + np.tanh((t - t0) / 2. / tau))
        s_t = s[0] + (s[1] - s[0]) / 2. * (1 + np.tanh((t - t0) / 2. / tau))
    else:
        raise NotImplementedError
    return s_t, a_t


def mr_piecewise(t, a, s, t0, t1, **kwargs):
    if a.size == 2:
        a_t = np.piecewise(t, [t < t0, t >= t0], list(a))
        s_t = np.piecewise(t, [t < t0, t >= t0], list(s))
    elif a.size == 3:
        a_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), t >= t1], list(a))
        s_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), t >= t1], list(s))
    else:
        raise NotImplementedError
    return s_t, a_t
