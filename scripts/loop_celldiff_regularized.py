import random
import os

from evoscape.population_class import Population
from evoscape.landscapes.landscape_dataset_fitness import CellDiff_Dataset_Landscape
from evoscape.modules.module_class import Node
from evoscape.morphogen_regimes import *
from evoscape.landscape_visuals import *
from evoscape.helper_functions import plot_compare_cell_proportions, get_cell_data

# _____________________________________________________________________________
save_dir = 'saved_files_rev_1/'

#  Hyperparameters
delta = 3.
noise = 0.2

#  Computation parameters
N = 200  # population size (e.g. 16 laptop, 200 computing server)
n_sim = 10
ndt = 200   # 200
ncells = 300
ngens0 = 101
ngens = 301 # 301

L = 5.

#  Priors
par_limits = {
    'x': (-L, L),
    'y': (-L, L),
    'a': (0., 16.),
    's': (0.1, 1.5),
}

par_choice_values = {}

# _______________________________________________________________________________

# Regime 1 - FGF, no Chir
# Regime 2 - Chir + FGF
# Regime 3 - Chir (+PD)
# Regime 4 - Chir + end. FGF
# Regime 5 - endogenous FGF (90% FGF)

# Module 0 (Epi): a2-4, s2-4 = 0
# Module 1 (Tr):  a0, a2-4, s0, s2-4 = 0
# Module 2 (CE):  a0, s0 = 0
# Module 3 (PN):  a0, s0 = 0
# Module 4 (M):   a0, s0 = 0
# Module 5 (AN):  a0-3, s0-3 = 0

# ________________________________________________________________________________

filenames = ('NoCh.txt', 'Ch2-3.txt', 'Ch2-4.txt', 'Ch2-5.txt',
             'Ch2-5_FGF0-3.txt', 'Ch2-5_FGF0-4.txt', 'Ch2-5_FGF0-5.txt')

col_labels = ['EPI', 'Tr', 'CE', 'PN', 'M', 'AN']
row_labels = ['Day 1.5', 'Day 2', 'Day 2.5', 'Day 3', 'Day 3.5', 'Day 4', 'Day 4.5', 'Day 5']
col_colors = ['indianred', 'tab:orange', 'gold', 'tab:green', 'tab:blue', 'tab:purple']
cell_dataset = []

for filename in filenames:
    cell_data = get_cell_data(filename, remove_cols=(), col_order=(0, 1, 5, 3, 4, 2))
    cell_data = np.insert(cell_data, 0, cell_data[0], axis=0)
    for row in cell_data:
        row *= 1. / np.sum(row)  # rescale everything to sum up to 1
    cell_dataset.append(cell_data)

#  Make a subset of data: first 4 timepoints and 3 cell states
filename = 'Ch2-5_FGF0-3.txt'
cell_data = get_cell_data(filename)
cell_data = np.insert(cell_data, 0, cell_data[0], axis=0)  ## s

cell_data_4 = cell_data[:4]
row_labels_4 = ['Day 1.5', 'Day 2', 'Day 2.5', 'Day 3']
cell_data_4[-1, 2] = 87
cell_data_4[-1, -1] = 0
cell_data_4 = cell_data_4[:, :3]
col_labels_4 = col_labels[:3]
for row in cell_data_4:
    row *= 1. / np.sum(row)
cell_dataset_0 = (cell_data_4,)


morphogen_times_0 = ((delta * 1, delta * 3, delta*100, delta*100),)  # Signal is changing at timepoint 1 and timepoint 3
time_pars_0 = (0., delta * 3, 4)

#  Morphogen changing times for all experiments
morphogen_times = ((delta*3, delta*3, delta*3, delta*3),  # No Chir  (D3, D3, D3, D3)
                   (delta*1, delta*3, delta*3, delta*3),   # Chir 2-3  (D2, D3, D3, D3)
                   (delta*1, delta*3, delta*3, delta*5),  # Chir 2-4  (D2, D3, D3, D4)
                   (delta*1, delta*3, delta*3, delta*100),  # Chir 2-5    (D2, D3, D3, -)
                   (delta*1, delta*3, delta*100, delta*100),    # Ch 2-5 FGF 0-3  (D2, D3, -, -)
                   (delta*1, delta*5, delta*100, delta*100),    # Ch 2-5 FGF 0-4  (D2, D4, -, -)
                   (delta*1, delta*100, delta*100, delta*100),    # Ch 2-5 FGF 0-5    (D2, -, -, -)
                   )
time_pars = (0., delta * 7, 8)

# ___________________________________________________________________________
landscape_pars_celldiff = {
    'A0': 0.005,
    'init_cond': (0., 0.),
    'regime': mr_piecewise,
    'n_regimes': 5,  # !
    'morphogen_times': morphogen_times_0[0],
    'used_fp_types': [Node],
    'immutable_pars_list': [],
}

prob_pars_celldiff = {
    'prob_add': 0.,
    'prob_drop': 0.,
    'prob_shuffle': 0.
}


if __name__ == '__main__':
    print('N = ', N)
    for sim in range(n_sim):

        # Set up for optimizing the first 4 timepoints
        fitness_pars_celldiff = {
            'ncells': ncells,  #
            'cell_data': cell_dataset_0,
            'init_state': 0,
            'attractor_states': (2,),
            'non_attractor_states': (),
            'noise': noise,
            'penalty_weight': .1,
            'time_pars': time_pars_0,
            'morphogen_times': morphogen_times_0,
            'ndt': ndt,  # integration steps per time point
        }

        # Set up population
        # Start with 3 modules (red, orange, yellow)
        start_module_list = [
            random.choice(landscape_pars_celldiff['used_fp_types']).generate(par_limits, par_choice_values,
                                                                             n_regimes=landscape_pars_celldiff[
                                                                                 'n_regimes']) for i in
            range(cell_dataset_0[0].shape[1])]

        start_module_list[2].par_limits = {'x': (-1., 1.), 'y': (-1., 1.)}  # Constrain yellow to be around the center

        # ___________________________________________________________________________________________________________
        #   Set some parameters to 0:
        # Module 0 (Epi): a2-4, s2-4 = 0
        # Module 1 (Tr):  a0, a2-4, s0, s2-4 = 0
        # Module 2 (CE):  a0, s0 = 0
        start_module_list[0].a[2:] = 0.
        start_module_list[0].s[2:] = 0.
        start_module_list[0].set_immutable_idx([2,3,4])

        start_module_list[1].a[0] = 0.
        start_module_list[1].s[0] = 0.
        start_module_list[1].a[2:] = 0.
        start_module_list[1].s[2:] = 0.
        start_module_list[1].set_immutable_idx([0,2,3,4])

        start_module_list[2].a[0] = 0.
        start_module_list[2].s[0] = 0.
        start_module_list[2].set_immutable_idx([0])
        # ____________________________________________________________________________________________________________

        P = Population(N, CellDiff_Dataset_Landscape, landscape_pars_celldiff, prob_pars_celldiff,
                       fitness_pars_celldiff, par_limits, par_choice_values,
                       start_module_list=start_module_list)

        print('# '+str(sim))
        fitness_traj, timecode1, results_dir = P.evolve_parallel(ngens0, fitness_pars_celldiff, save_dir, save_each=50)
        # print('Done')

        fig = plt.figure(figsize=(4, 3))
        plt.plot(fitness_traj, lw=2, c='steelblue')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best fitness', fontsize=12)
        plt.ylim((-4, 0))
        fig.savefig(results_dir + '/result_fitness_traj.png', bbox_inches='tight')
        plt.close(fig)

        landscape = P.landscape_list[0]

        #  Plot result Vs target proportions
        fig = plot_compare_cell_proportions(cell_data_4, landscape.result[0], col_labels_4, col_colors,
                                            row_labels_4)
        fig.savefig(results_dir + '/result_proportions.png', bbox_inches='tight')
        plt.close(fig)

        npoints = 201
        q = np.linspace(-L, L, npoints)
        xx, yy = np.meshgrid(q, q, indexing='xy')
        times = np.array((0., delta * 1.1))
        figures = visualize_all(landscape, xx, yy, times, density=0.45, plot_traj=False, color_scheme='order')

        for i in range(len(figures)):
            figures[i].savefig(results_dir + '/result_landscape_' + str(i) + '.png')
            plt.close(figures[i])

        landscape.morphogen_times = fitness_pars_celldiff['morphogen_times'][0]
        n = 30
        landscape.init_cells(n, 0, noise)
        fig = get_and_plot_traj(landscape, 0, delta * 3, 11, L, noise, frozen=False)
        # fig.show()
        fig.savefig(results_dir + '/result_cell_trajectories.png', bbox_inches='tight')
        plt.close(fig)

        fig = plot_cells(landscape, L)
        # fig.show()
        fig.savefig(results_dir + '/result_final_state.png', bbox_inches='tight')
        plt.close(fig)

        # Set up for the full optimization
        # To each landscape, add three randomly generated modules (green, blue, purple)

        for landscape in P.landscape_list:
            module_list = landscape.module_list
            for i in range(3):
                module_list.append(random.choice(
                    landscape_pars_celldiff['used_fp_types']).generate(par_limits,par_choice_values,
                                                                       n_regimes=landscape_pars_celldiff['n_regimes']))

        # ______________________________________
        # Set some parameters to 0
        # Module 3 (PN):  a0, s0 = 0
        # Module 4 (M):   a0, s0 = 0
        # Module 5 (AN):  a0-3, s0-3 = 0
        for landscape in P.landscape_list:
            module_list = landscape.module_list
            module_list[3].a[0] = 0.
            module_list[3].s[0] = 0.
            module_list[3].set_immutable_idx([0])

            module_list[4].a[0] = 0.
            module_list[4].s[0] = 0.
            module_list[4].set_immutable_idx([0])

            module_list[5].a[0:4] = 0.
            module_list[5].s[0:4] = 0.
            module_list[5].set_immutable_idx([0, 1, 2, 3])

        # ____________________________________________________________________________________________________________

        # Reset fitness
        for landscape in P.landscape_list:
            landscape.fitness = -np.inf

        fitness_pars_celldiff = {
            'ncells': ncells,  #
            'cell_data': cell_dataset,  # full dataset
            'init_state': 0,
            'attractor_states': (),
            'non_attractor_states': (),
            'noise': noise,
            'penalty_weight': 0.,
            'time_pars': time_pars,
            'morphogen_times': morphogen_times,
            'ndt': ndt,
        }

        fitness_traj, timecode2, results_dir = P.evolve_parallel(ngens, fitness_pars_celldiff, save_dir, save_each=50)
        # print('Done')

        # Fitness plot
        fig = plt.figure(figsize=(4, 3))
        plt.plot(fitness_traj, lw=2, c='steelblue')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best fitness', fontsize=12)
        plt.gca().set_ylim(top=0)
        fig.savefig(results_dir + '/result_fitness_traj.png', bbox_inches='tight')
        plt.close(fig)

        landscape = P.landscape_list[0]

        #  Plot result Vs target proportions
        for k in range(len(cell_dataset)):
            fig = plot_compare_cell_proportions(cell_dataset[k], landscape.result[k], col_labels, col_colors,
                                                row_labels=None)
            fig.savefig(results_dir + '/result_proportions_' + str(k) + '.png', bbox_inches='tight')
            plt.close(fig)

        # Plot the landscape (all regimes)
        landscape.morphogen_times = (delta*1, delta*3, delta*5, delta*7)
        times = np.arange(0, delta*10, delta*2)

        npoints = 201
        q = np.linspace(-L, L, npoints)
        xx, yy = np.meshgrid(q, q, indexing='xy')
        figures = visualize_all(landscape, xx, yy, times, density=0.45, plot_traj=False, color_scheme='order')
        for i in range(len(figures)):
            figures[i].savefig(results_dir + '/result_landscape_' + str(i) + '.png')
            plt.close(figures[i])

        #  ________________________________________________________________________________________________________

        # Plot trajectories from several experiments
        experiments = range(7)
        n = 50

        for exp in experiments:
            landscape.morphogen_times = fitness_pars_celldiff['morphogen_times'][exp]
            landscape.init_cells(n, 0, noise)
            fig = get_and_plot_traj(landscape, 0, delta * 7, 51, L, noise, frozen=False)
            fig.savefig(results_dir + '/result_cell_trajectories_'+str(exp)+'.png', bbox_inches='tight')
            plt.close(fig)

            fig = plot_cells(landscape, L)
            fig.savefig(results_dir + '/result_final_state_'+str(exp)+'.png', bbox_inches='tight')
            plt.close(fig)

        plt.close('all')

        log_filename = save_dir + '/' + landscape.__class__.__name__ + '/optimization_log.csv'
        if not os.path.exists(log_filename):
            with open(log_filename, 'a') as f:
                f.write('# Main timecode\tInit timecode\tFitness\n')

        with open(log_filename, 'a') as f:
            f.write('\t'.join([timecode2, timecode1, str(P.landscape_list[0].fitness)]) + '\n')
