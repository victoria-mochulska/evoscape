import jax
import jax.numpy as jnp
import numpy as np
from .helmholtz_decomp import *
from .module_gaussian import *
from .visualizations import *
from .gradient_descent import *
from .drosophila import *


def heteroclinic_flip_vectors(x, y, a=0, b=0):
    dx = 4*x**3 + 3*x**2 - 2*y**2 - 2*x + a 
    dy =  4*y**3 - 4*x*y + b
    return jnp.array([-dx, -dy])


def heteroclinic_flip_potential(x, y, a=0, b=0):
    return x**4 + y**4 + x**3 - 2*x*y**2 - x**2 + a*x + b*y 


def dual_cusp_potential(x, y, a=0, b=0):
    return y**4 + x**4 + x**3 + x**2 - 4*x*y**2 + b*y - a*x

def dual_cusp_vectors(x, y, a=0, b=0):
    dx = 4*x**3 + 3*x**2 + 2*x - 4*y**2 - a
    dy = 4*y**3 - 8*x*y + b
    return jnp.array([-dx, -dy])


def triple_cusp_potential(x, y, a=0, b=0):
    return y**6 - y**4 + y**2 + 4*x**4 - x**2 - 2*x*y**2 - a*x + b*y

def triple_cusp_vectors(x, y, a=0, b=0):
    dx = 16*x**3 - 2*x - 2*y**2 - a
    dy = 6*y**5 - 4*y**3 + 2*y - 4*x*y + b
    return jnp.array([-dx, -dy])


def fold_vectors(x, a=1):
    dx = 3*x**2 + x
    dy = 0
    return jnp.array([-dx, -dy])

def fold_potential(x, a=1):
    return x**3


def elliptic_umbilic_vectors(x, y, a=0, b=0):
    dx = 3*x**2 - 2*y**2 + 0.4*(2*x) + a + (x**3)
    dy = -4*x*y + 0.4*(2*y) + b + (y**3)
    return jnp.array([-dx, -dy])

def elliptic_umbilic_potential(x, y, a=0, b=0):
    return x**3 - 2*x*y**2 + 0.4*(x**2 + y**2) + a*x + b*y + 0.25*(x**4 + y**4)



def make_fake_data(xx, yy, vector_func, potential_func, params, nbr_cells, length, dt, initial_position_lim_x, initial_position_lim_y):
    x_min_lim, x_max_lim = initial_position_lim_x 
    y_min_lim, y_max_lim = initial_position_lim_y


    x_init_coords = np.random.uniform(x_min_lim, x_max_lim, size=nbr_cells)
    y_init_coords = np.random.uniform(y_min_lim, y_max_lim, size=nbr_cells)

    init_coords = np.array([x_init_coords, y_init_coords]).T
    final_pos, traj = jax.vmap(make_trajectories_personnal, in_axes=(None, None, 0, None, None) )(vector_func, params, init_coords, length, dt)
    return jnp.swapaxes(traj, axis1=0, axis2=2), potential_func(xx, yy, *params)


