import jax.random as jrd
import jax.numpy as jnp
import jax.image as jimg
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import io
import numpy as np
import pandas as pd
from jax import jit, vmap
from evoscape.landscape_visuals import visualize_landscape
from evoscape.jax.converters import pytree_to_landscape

# Initialization function used in the fitness function, we let the model choose where the cells should start in optimization
def init_cell(key,n,init_cond,noise):                    
    key,subkey = jrd.split(key)
    return  key,init_cond+noise*jrd.normal(subkey, shape=(2, n))

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
    imageio.mimsave("../temp/figures/animation.gif", frames, duration=10.)

@jit
def rescale(A, N, T):
    return vmap(lambda x: jimg.resize(x, (N, T), method="linear"))(A)

def get_drosophile_data(pathfile):
    df =pd.read_csv(pathfile, sep =",")
    #Hardcoded 
    data = np.zeros((4,930,291))
    for _, row in df.iterrows():
        t = row["time"]
        x = row["x"]
        data[0][x][t] = row["Gt_data"]
        data[1][x][t] = row["Kni_data"]
        data[2][x][t] = row["Hb_data"]
        data[3][x][t] = row["Kr_data"]
    
    return jnp.array(data)

def get_facs_data(pathfile_to_conditioned_facs_data):
    df =pd.read_csv(pathfile_to_conditioned_facs_data, sep =",")
    
    genes = ["TBX6", "BRA", "CDX2", "SOX2", "SOX1"]

    df = df.sort_values("timepoint").reset_index(drop=True)

    unique_timepoints = np.sort(df["timepoint"].unique())

    n_time = df["timepoint"].nunique()

    n_genes = 5

    values = df[genes].to_numpy()

    # (cells, time, genes)
    values = values.reshape(-1, n_time, n_genes)

    # (genes, cells, time)
    values = np.transpose(values, (2, 0, 1))

    x = jnp.array(values)

    return x, unique_timepoints



    

