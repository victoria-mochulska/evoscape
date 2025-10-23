import numpy as np


@np.vectorize
def mr_const(t, a, s, **kwargs):
    return s[0], a[0]


def mr_sigmoid(t, a, s, t0, tau, **kwargs):
    t = np.asarray(t)
    if a.size == 2:
        tanh = np.tanh((t[..., None] - t0) / 2. / tau)
        a_t = a[0] + (a[1] - a[0]) / 2. * (1 + tanh)
        s_t = s[0] + (s[1] - s[0]) / 2. * (1 + tanh)
    else:
        raise NotImplementedError
    return s_t, a_t


def mr_piecewise(t, a, s, t0, t1=None, t2=None, t3=None, **kwargs):
    t = float(t)
    if a.size == 2:
        a_t = np.piecewise(t, [t < t0, t >= t0], list(a))
        s_t = np.piecewise(t, [t < t0, t >= t0], list(s))
    elif a.size == 3:
        a_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), t >= t1], list(a))
        s_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), t >= t1], list(s))
    elif a.size == 4:
        a_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), (t >= t1) & (t < t2), t >= t2], list(a))
        s_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), (t >= t1) & (t < t2), t >= t2], list(s))
    elif a.size == 5:
        a_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), (t >= t1) & (t < t2), (t >= t2) & (t < t3), t >= t3], list(a))
        s_t = np.piecewise(t, [t < t0, (t >= t0) & (t < t1), (t >= t1) & (t < t2), (t >= t2) & (t < t3), t >= t3], list(s))
    else:
        raise NotImplementedError
    return s_t, a_t

def mr_current_regime(t, t0, t1=None, t2=None, t3=None):
    """Return an index indicating the current time:
    0: t < t0, 1: t0 ≤ t < t1, ..., n: t ≥ last threshold
    """
    thresholds = [th for th in [t0, t1, t2, t3] if th is not None]
    conds = [t < thresholds[0]] if thresholds else [np.ones_like(t, dtype=bool)]

    for i in range(len(thresholds) - 1):
        conds.append((t >= thresholds[i]) & (t < thresholds[i + 1]))

    if thresholds:
        conds.append(t >= thresholds[-1])

    return np.select(conds, list(range(len(conds))))


# ___________________________________________________________________________________________________________________

def mr_linear_2signals(t, a, s, t0, t1, **kwargs):
    signal1= t0(t)
    signal2 = t1(t)
    a_t = a[0] + a[1]*signal1 + a[2]*signal2
    s_t = s[0] + s[1]*signal1 + s[2]*signal2
    # Clip parameters to the allowed range
    # Negative a allowed
    a_t = np.maximum(a_t, 0.0)
    s_t = np.maximum(s_t, 0.2)
    # a_t = np.minimum(a_t, 16.0)
    # s_t = np.minimum(s_t, 1.5)
    a_t = np.minimum(a_t, 20.0)
    s_t = np.minimum(s_t, 1.2)
    return s_t, a_t

