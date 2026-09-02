import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
from src.utilities import *
import time
from matplotlib.backends.backend_pdf import PdfPages
import json
import logging
from pathlib import Path
from datetime import datetime
import os
import argparse
from scipy.interpolate import griddata

np.random.seed(42)



npoints = 200 + 1
dim = 2.8
q = np.linspace(-dim, dim, npoints)
xx, yy = np.meshgrid(q, q, indexing='xy')
num = 10
trajs = []
pots = []
start_time = time.time()

#heteroclinic
for i in range(num):
    params = (np.random.uniform(-2, 2), np.random.uniform(-1, 1))
    traj, pot = make_fake_data(xx, yy, heteroclinic_flip_vectors, heteroclinic_flip_potential, params, 558, 200, 0.008, (-dim, dim), (-dim, dim))
    trajs.append(traj)
    pots.append(pot)

    #fig = show_trajectory(traj, traj, "other/fake_data/traj" + str(i) +".gif")
    fig = show_trajectories(traj, traj, title=f"a = {params[0]:.2f} et b = {params[1]:.2f}", save=True, filename="other/fake_data/heteroclinic_flip/traj/traj" + str(i) + ".png")
    plt.close(fig)
    fig = show_potential(xx, yy, pot, title=f"Potentiel " + str(i) + f" | a = {params[0]:.2f} et b = {params[1]:.2f}", save=True, filename="other/fake_data/heteroclinic_flip/pot/pot" + str(i) + ".png")
    plt.close(fig)
    #fig = show_trajectory(traj, traj, title=f"a = {params[0]:.2f} et b = {params[1]:.2f}", filename="other/fake_data/heteroclinic_flip/gif/traj" + str(i) +".gif", fps=10, frame_step=5, linewidth=2, point_size=32)
    print(traj.shape)
    np.savetxt("other/fake_data/heteroclinic_flip/data/heteroclinic_flip_" + str(i) + "_gene_X.txt", traj[0])
    np.savetxt("other/fake_data/heteroclinic_flip/data/heteroclinic_flip_" + str(i) + "_gene_Y.txt", traj[1])
    print(f"Time :{time.time() - start_time:.2f}")


#umbilic
for i in range(num):
    params = (np.random.uniform(-2, 2), np.random.uniform(-2.5, 2.5))
    traj, pot = make_fake_data(xx, yy, elliptic_umbilic_vectors, elliptic_umbilic_potential, params, 558, 200, 0.008, (-dim, dim), (-dim, dim))
    trajs.append(traj)
    pots.append(pot)

    #fig = show_trajectory(traj, traj, "other/fake_data/traj" + str(i) +".gif")
    fig = show_trajectories(traj, traj, title=f"a = {params[0]:.2f} et b = {params[1]:.2f}", save=True, filename="other/fake_data/elliptic_umbilic/traj/traj" + str(i) + ".png")
    plt.close(fig)
    fig = show_potential(xx, yy, pot, title=f"Potentiel " + str(i) + f" | a = {params[0]:.2f} et b = {params[1]:.2f}", save=True, filename="other/fake_data/elliptic_umbilic/pot/pot" + str(i) + ".png")
    plt.close(fig)
    #fig = show_trajectory(traj, traj, title=f"a = {params[0]:.2f} et b = {params[1]:.2f}", filename="other/fake_data/elliptic_umbilic/gif/traj" + str(i) +".gif", fps=10, frame_step=5, linewidth=2, point_size=32)
    np.savetxt("other/fake_data/elliptic_umbilic/data/elliptic_umbilic_" + str(i) + "_gene_X.txt", traj[0])
    np.savetxt("other/fake_data/elliptic_umbilic/data/elliptic_umbilic_" + str(i) + "_gene_Y.txt", traj[1])

    print(f"Time :{time.time() - start_time:.2f}")


#dual cusp
for i in range(num):
    params = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))
    traj, pot = make_fake_data(xx, yy, dual_cusp_vectors, dual_cusp_potential, params, 558, 200, 0.008, (-dim, dim), (-dim, dim))
    trajs.append(traj)
    pots.append(pot)

    #fig = show_trajectory(traj, traj, "other/fake_data/traj" + str(i) +".gif")
    fig = show_trajectories(traj, traj, title=f"a = {params[0]:.2f} et b = {params[1]:.2f}", save=True, filename="other/fake_data/dual_cusp/traj/traj" + str(i) + ".png")
    plt.close(fig)
    fig = show_potential(xx, yy, pot, title=f"Potentiel " + str(i) + f" | a = {params[0]:.2f} et b = {params[1]:.2f}", save=True, filename="other/fake_data/dual_cusp/pot/pot" + str(i) + ".png")
    plt.close(fig)
    #fig = show_trajectory(traj, traj, title=f"a = {params[0]:.2f} et b = {params[1]:.2f}", filename="other/fake_data/dual_cusp/gif/traj" + str(i) +".gif", fps=10, frame_step=5, linewidth=2, point_size=32)
    np.savetxt("other/fake_data/dual_cusp/data/dual_cusp_" + str(i) + "_gene_X.txt", traj[0]) 
    np.savetxt("other/fake_data/dual_cusp/data/dual_cusp_" + str(i) + "_gene_Y.txt", traj[1])   
  
 
    print(f"Time :{time.time() - start_time:.2f}")


#triple cusp
for i in range(num):
    params = (np.random.uniform(-1.5, 1.5), np.random.uniform(-2, 2))
    traj, pot = make_fake_data(xx, yy, triple_cusp_vectors, triple_cusp_potential, params, 558, 200, 0.008, (-2, 2), (-2, 2))
    trajs.append(traj)
    pots.append(pot)

    #fig = show_trajectory(traj, traj, "other/fake_data/traj" + str(i) +".gif")
    fig = show_trajectories(traj, traj, title=f"a = {params[0]:.2f} et b = {params[1]:.2f}", save=True, filename="other/fake_data/triple_cusp/traj/traj" + str(i) + ".png")
    plt.close(fig)
    fig = show_potential(xx, yy, pot, title=f"Potentiel " + str(i) + f" | a = {params[0]:.2f} et b = {params[1]:.2f}", save=True, filename="other/fake_data/triple_cusp/pot/pot" + str(i) + ".png")
    plt.close(fig)
    #fig = show_trajectory(traj, traj, title=f"a = {params[0]:.2f} et b = {params[1]:.2f}", filename="other/fake_data/triple_cusp/gif/traj" + str(i) +".gif", fps=10, frame_step=5, linewidth=2, point_size=32)
    np.savetxt("other/fake_data/triple_cusp/data/triple_cusp_" + str(i) + "_gene_X.txt", traj[0])   
    np.savetxt("other/fake_data/triple_cusp/data/triple_cusp_" + str(i) + "_gene_Y.txt", traj[1])    
 
    print(f"Time :{time.time() - start_time:.2f}")


print(f"Time :{time.time() - start_time:.2f}")
 