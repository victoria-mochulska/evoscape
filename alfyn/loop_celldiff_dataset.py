import random
import os
import warnings

from landscapes.population_class import Population
from landscapes.landscapes.landscape_dataset_fitness import CellDiff_Dataset_Landscape
from landscapes.modules.module_class import Node
from landscapes.morphogen_regimes import *
from landscapes.landscape_visuals import *
from landscapes.helper_functions import plot_compare_cell_proportions, get_cell_data

warnings.simplefilter('ignore')

# _____________________________________________________________________________
save_dir = 'saved_files_2/'

#  Hyperparameters
day = 2.
noise = 0.2

#  Computation parameters
N = 200  # population size
n_sim = 200
ndt = 200   # 200
ncells = 300

#  Priors
par_limits_1 = {
    'x': (-2., 2.),
    'y': (-2., 2.),
    'a': (0., 9.),
    's': (0.1, 1.5),
}

par_limits_2 = {
    'x': (-4., 4.),
    'y': (-4., 4.),
    'a': (0., 16.),
    's': (0.1, 1.5),
}

# _______________________________________________________________________________

# Regime 1 - FGF, no Chir               (red bar)
# t0
# Regime 2 - Chir + FGF                (purple+red bar)
# t1
# Regime 3 - Chir (+PD)                (purple+green)

# #   Training data  # #

#  Initial training: regimes 1, 2, 3

#  Ch 2-5 FGF 0-3:
#  regime 1 until day 2, regime 1 day 2-3, regime 3 day 3-5
# t0 = day 2, t1 = day 3

#  Ch 2-5 FGF 0-4:
#  regime 1 until day 2, regime 1 day 2- 4, regime 3 day 4-5
# t0 = day 2, t1 = day 4

#  Ch 2-5 FGF 0-5:
#  regime 1 until day 2, regime 1 day 2-5
# t0 = day 2, t1 = day 5

# ________________________________________________________________________________

#  Loading the dataset of 3 experiments
filenames = ('Ch2-5_FGF0-3.txt', 'Ch2-5_FGF0-4.txt', 'Ch2-5_FGF0-5.txt')
cell_dataset = []
for filename in filenames:
    cell_data = get_cell_data(filename)
    cell_data = np.insert(cell_data, 0, cell_data[0], axis=0)  ## s
    col_labels = ['EPI', 'Tr', 'CE', 'PN', 'M']
    row_labels = ['D1.5', 'D2', 'D2.5', 'D3', 'D3.5', 'D4', 'D4.5', 'D5']
    col_colors = ['indianred', 'tab:orange', 'gold', 'tab:green', 'tab:blue']

    for row in cell_data:
        row *= 1. / np.sum(row)  # rescale everything to sum up to 1

    cell_dataset.append(cell_data)

#  Make a subset of data: first 4 timepoints and 3 cell states
filename = 'Ch2-5_FGF0-3.txt'
cell_data = get_cell_data(filename)
cell_data = np.insert(cell_data, 0, cell_data[0], axis=0)  ## s
col_labels = ['EPI', 'Tr', 'CE', 'PN', 'M']
col_colors = ['indianred', 'tab:orange', 'gold', 'tab:green', 'tab:blue']

cell_data_4 = cell_data[:4]
row_labels_4 = ['D1.5', 'D2', 'D2.5', 'D3']
cell_data_4[-1, 2] = 87
cell_data_4[-1, -1] = 0
cell_data_4 = cell_data_4[:, :3]
col_labels_4 = col_labels[:3]

for row in cell_data_4:
    row *= 1. / np.sum(row)

cell_dataset_0 = (cell_data_4,)

# In[]:

if __name__ == '__main__':
    print('N = ', N)
    for sim in range(n_sim):

        # Set up for optimizing the first 4 timepoints

        time_pars_0 = (0., day * 3, 4)
        morphogen_times_0 = ((day * 1, day * 3),)  # Signal is changing at timepoint 1 and timepoint 3

        par_limits = par_limits_1

        par_choice_values = {}

        landscape_pars_celldiff = {
            'A0': 0.005,
            'init_cond': (0., 0.),
            'regime': mr_piecewise,
            'n_regimes': 3,  # !
            'morphogen_times': morphogen_times_0[0],
            'used_fp_types': [Node],
            'immutable_pars_list': [],
        }

        prob_pars_celldiff = {
            'prob_add': 0.,
            'prob_drop': 0.,
            'prob_shuffle': 0.
        }

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

        P = Population(N, CellDiff_Dataset_Landscape, landscape_pars_celldiff, prob_pars_celldiff,
                       fitness_pars_celldiff, par_limits, par_choice_values,
                       start_module_list=start_module_list)

        print('# '+str(sim))
        fitness_traj, timecode1, results_dir = P.evolve_parallel(101, fitness_pars_celldiff, save_dir, save_each=50)
        # print('Done')

        fig = plt.figure(figsize=(4, 3))
        plt.plot(fitness_traj, lw=2, c='steelblue')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best fitness', fontsize=12)
        plt.ylim((-4, 0))
        # fig.show()
        fig.savefig(results_dir + '/result_fitness_traj.png', bbox_inches='tight')
        plt.close(fig)

        #  Plot result Vs target proportions
        fig = plot_compare_cell_proportions(cell_data_4, P.landscape_list[0].result[0], col_labels_4, col_colors,
                                            row_labels_4)
        # fig.show()
        fig.savefig(results_dir + '/result_proportions.png', bbox_inches='tight')
        plt.close(fig)

        landscape = P.landscape_list[0]
        # print(landscape)

        L = 4.
        npoints = 201
        q = np.linspace(-L, L, npoints)
        xx, yy = np.meshgrid(q, q, indexing='xy')
        times = np.array((0., day * 1.1))
        figures = visualize_all(landscape, xx, yy, times, density=0.45, plot_traj=False, color_scheme='order')

        for i in range(len(figures)):
            figures[i].savefig(results_dir + '/result_landscape_' + str(i) + '.png')
            plt.close(figures[i])

        landscape.morphogen_times = fitness_pars_celldiff['morphogen_times'][0]
        n = 30
        landscape.init_cells(n, 0, 0.1)
        fig = get_and_plot_traj(landscape, 0, day * 3, 11, L, 0.1, frozen=False)
        # fig.show()
        fig.savefig(results_dir + '/result_cell_trajectories.png', bbox_inches='tight')
        plt.close(fig)

        fig = plot_cells(landscape, L)
        # fig.show()
        fig.savefig(results_dir + '/result_final_state.png', bbox_inches='tight')
        plt.close(fig)

        # In[]:

        # Set up for the full optimization
        # To each landscape, add two randomly generated modules (green and blue)

        par_limits = par_limits_2
        P.par_limits = par_limits

        for landscape in P.landscape_list:
            start_module_list = landscape.module_list
            start_module_list.append(random.choice(landscape_pars_celldiff['used_fp_types']).generate(par_limits,
                                                                                                      par_choice_values,
                                                                                                      n_regimes=
                                                                                                      landscape_pars_celldiff[
                                                                                                          'n_regimes']))
            start_module_list.append(random.choice(landscape_pars_celldiff['used_fp_types']).generate(par_limits,
                                                                                                      par_choice_values,
                                                                                                      n_regimes=
                                                                                                      landscape_pars_celldiff[
                                                                                                          'n_regimes']))

        # Reset fitness
        for landscape in P.landscape_list:
            landscape.fitness = -np.inf

        time_pars = (0., day * 7, 8)

        #   2 morphogen changing times for each of the 3 experiments
        morphogen_times = ((day * 1, day * 3), (day * 1, day * 5), (day * 1, day * 7))

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

        fitness_traj, timecode2, results_dir = P.evolve_parallel(301, fitness_pars_celldiff, save_dir, save_each=50)
        # print('Done')

        fig = plt.figure(figsize=(4, 3))
        plt.plot(fitness_traj, lw=2, c='steelblue')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best fitness', fontsize=12)
        plt.gca().set_ylim(top=0)
        fig.savefig(results_dir + '/result_fitness_traj.png', bbox_inches='tight')
        plt.close(fig)

        #  Plot result Vs target proportions
        for k in range(len(cell_dataset)):
            fig = plot_compare_cell_proportions(cell_dataset[k], P.landscape_list[0].result[k], col_labels, col_colors,
                                                row_labels=None)
            fig.show()
            fig.savefig(results_dir + '/result_proportions_' + str(k) + '.png', bbox_inches='tight')
            plt.close(fig)

        landscape = P.landscape_list[0]

        L = 4.
        npoints = 201
        q = np.linspace(-L, L, npoints)
        xx, yy = np.meshgrid(q, q, indexing='xy')
        times = np.array((0., day * 2, day * 10))
        figures = visualize_all(landscape, xx, yy, times, density=0.45, plot_traj=False, color_scheme='order')

        for i in range(len(figures)):
            figures[i].savefig(results_dir + '/result_landscape_' + str(i) + '.png')
            plt.close(figures[i])

        #  Plot trajectories of the first experiment
        landscape.morphogen_times = fitness_pars_celldiff['morphogen_times'][0]
        n = 50
        landscape.init_cells(n, 0, 0.1)
        fig = get_and_plot_traj(landscape, 0, day * 7, 51, L, 0.1, frozen=False)
        # fig.show()
        fig.savefig(results_dir + '/result_cell_trajectories_1.png', bbox_inches='tight')
        plt.close(fig)

        fig = plot_cells(landscape, L)
        # fig.show()
        fig.savefig(results_dir + '/result_final_state_1.png', bbox_inches='tight')
        plt.close(fig)

        #  Plot trajectories of the third experiment
        landscape.morphogen_times = fitness_pars_celldiff['morphogen_times'][2]
        n = 50
        landscape.init_cells(n, 0, 0.1)
        fig = get_and_plot_traj(landscape, 0, day * 7, 51, L, 0.1, frozen=False)
        # fig.show()
        fig.savefig(results_dir + '/result_cell_trajectories_2.png', bbox_inches='tight')
        plt.close(fig)

        fig = plot_cells(landscape, L)
        # fig.show()
        fig.savefig(results_dir + '/result_final_state_2.png', bbox_inches='tight')
        plt.close(fig)

        plt.close('all')

        log_filename = save_dir + '/' + landscape.__class__.__name__ + '/optimization_log.csv'
        if not os.path.exists(log_filename):
            with open(log_filename, 'a') as f:
                f.write('# Main timecode\tInit timecode\tFitness\n')

        with open(log_filename, 'a') as f:
            f.write('\t'.join([timecode2, timecode1, str(P.landscape_list[0].fitness)]) + '\n')
