import numpy as np

# timepoint 0 = D1.5, 1 = D2, 3 = D3, 5 = D4, 7 = D5
delta = 3.  # timepoint duration (a.u.)

def no_chir(t):
    t = np.asarray(t)
    return np.zeros_like(t, dtype=float)

def chir_2_3(t):
    t = np.asarray(t)
    return np.where((1. * delta <= t) & (t < 3. * delta), 1.0, 0.0)

def chir_2_4(t):
    t = np.asarray(t)
    return np.where((1. * delta <= t) & (t < 5. * delta), 1.0, 0.0)

def chir_2_5(t):
    t = np.asarray(t)
    return np.where(t >= 1. * delta, 1.0, 0.0)

def fgf_no_pd(t):
    t = np.asarray(t)
    return np.where(t < 3. * delta, 1.0, 0.9)

def fgf_0_3(t):
    t = np.asarray(t)
    return np.where(t < 3. * delta, 1.0, 0.)

def fgf_0_4(t):
    t = np.asarray(t)
    return np.where(t < 5. * delta, 1.0, 0.)

def fgf_0_5(t):
    t = np.asarray(t)
    return np.ones_like(t, dtype=float)