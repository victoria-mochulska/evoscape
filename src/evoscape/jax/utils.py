import jax.random as jrd
import jax.numpy as jnp
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import io
import numpy as np
from evoscape.landscape_visuals import visualize_landscape
from evoscape.jax.converters import pytree_to_landscape

# Initialization function used in the fitness function, we let the model choose where the cells should start in optimization
def init_cell(key,n,init_cond,noise):                    
    key,subkey = jrd.split(key)
    return  key,init_cond[:,None]+noise*jrd.normal(subkey, shape=(2, n))

def init_cell_circle(n, center, r):
    angles = jnp.linspace(0, 2*jnp.pi, n)

    x = jnp.cos(angles)*r + center[0]
    y = jnp.sin(angles)*r + center[1]

    return jnp.stack([x,y])


def make_movie_landscape(dynamics, static, xx, yy, n_frames):
    frames = []
    indices = np.linspace(0, len(dynamics)-1, n_frames, dtype=int)
    selected = [dynamics[i] for i in indices]
    for dynamic in selected:
        l = pytree_to_landscape(dynamic, static)
        fig = visualize_landscape(l, xx, yy, regime=0, color_scheme='fp_types')
        plt.close(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
    imageio.mimsave("animation.gif", frames, duration=10.)