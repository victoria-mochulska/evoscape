
import matplotlib.pyplot as plt
import numpy as np
import os
# import warnings

from evoscape.population_class import Population
from evoscape.landscapes import Somitogenesis_Landscape
from evoscape.modules import Node, UnstableNode, Center, NegCenter
from evoscape.morphogen_regimes import *
from evoscape.landscape_visuals import *


# ___________________________________________
save_dir = 'saved_files_somites_1/'


time_pars = (0., 50., 201)
morphogen_times = (0.,)

#  Computation parameters
N = 100  # population size (e.g. 16 laptop, 100 computing server)
n_sim = 30  #
ndt = 50   # integration steps per traj. timestep
ncells = 50
ngens = 30  #

# ___________________________________________
par_limits = {
    'x': (-2., 2.),
    'y': (-2., 2.),
    'a': (0., 4.),
    's': (0.2, 1.5),
}

par_choice_values = {
    'tau': (5.,)
}

landscape_pars = {
    'A0': 0.005,
    'init_cond': (0., 0.),
    'regime': mr_sigmoid,
    'n_regimes': 2,
    'morphogen_times': morphogen_times,
    'used_fp_types': (Node, UnstableNode, Center, NegCenter),
    'immutable_pars_list': ['tau',],
}

prob_pars = {
    'prob_add': 0.15,
    'prob_drop': 0.15,
    'prob_shuffle': 0.
    # the rest is mutation of parameters
}

fitness_pars = {
    'ncells': ncells,
    'time_pars': time_pars,
    'init_state': (0., 0.),
    't0_shift': 1.,  # shift (delay) of the time of transition between 2 neighbor cells
    'noise': 0.0,
    'low_value': -1.,
    'high_value': 1.,
    'penalty_weight': 0.1,
    't_stable': 5,  # how many timepoints should be at steady state
    'ndt': ndt,
}

L = 3.
npoints = 201
q = np.linspace(-L, L, npoints)
xx, yy = np.meshgrid(q, q, indexing='xy')

# ________________________________________________________________________


if __name__ == '__main__':
    print('N = ', N)
    for sim in range(n_sim):
        print('# '+str(sim))
        #  Starting with 2 random nodes, then any modules can be added or deleted
        start_module_list = [Node.generate(par_limits, par_choice_values, n_regimes=2) for i in range(2)]

        P = Population(N, Somitogenesis_Landscape, landscape_pars, prob_pars,
                         fitness_pars, par_limits, par_choice_values, start_module_list=start_module_list)

        fitness_traj, timecode, results_dir = P.evolve_parallel(ngens, fitness_pars, save_dir, save_each=1)

        # Fitness plot
        fig = plt.figure(figsize=(4, 3))
        plt.plot(fitness_traj, lw=2, c='steelblue')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best fitness', fontsize=12)
        plt.gca().set_ylim(bottom=0, top=ncells//2)
        fig.savefig(results_dir + '/result_fitness_traj.png', bbox_inches='tight')
        plt.close(fig)

        with open(results_dir + '/result_fitness_traj.npy', 'wb') as f:
            np.save(f, fitness_traj)

        landscape = P.landscape_list[0]
        # kymograph
        fig = plt.figure()
        plt.imshow(P.landscape_list[0].result, cmap='Blues')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Space', fontsize=15)
        fig.savefig(results_dir + '/result_kymograph.png', bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        plt.plot(P.landscape_list[0].result[:, -1], lw=2, alpha=0.6)
        plt.title('Final pattern', fontsize=15)
        plt.xlabel('Space', fontsize=15)
        fig.savefig(results_dir + '/result_pattern.png', bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        plt.plot(P.landscape_list[0].result[20, :], lw=2, alpha=0.6)
        plt.title('Single-cell dynamics', fontsize=15)
        plt.xlabel('Time', fontsize=15)
        fig.savefig(results_dir + '/result_dynamics.png', bbox_inches='tight')
        plt.close(fig)

        #  plot the landscape at 3 timepoints
        times = np.array((-10, 0., 10.))
        figures = visualize_all(landscape, xx, yy, times, density=0.45, plot_traj=False, color_scheme='fp_types')
        for i in range(len(figures)):
            figures[i].savefig(results_dir + '/result_landscape_' + str(i) + '.png')
            plt.close(figures[i])

        log_filename = save_dir + '/' + landscape.__class__.__name__ + '/optimization_log.csv'
        if not os.path.exists(log_filename):
            with open(log_filename, 'a') as f:
                f.write('#Timecode\tFitness\n')

        with open(log_filename, 'a') as f:
            f.write('\t'.join([timecode, str(P.landscape_list[0].fitness)]) + '\n')


