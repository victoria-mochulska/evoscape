import random
import os

from evoscape.population_class import Population
from evoscape.landscapes.landscape_dataset_fitness import CellDiff_Dataset_Landscape
from evoscape.modules.module_class import Node
from evoscape.morphogen_regimes import *
from evoscape.landscape_visuals import *
from evoscape.helper_functions import plot_compare_cell_proportions, get_cell_data

from exp_description import *

# _____________________________________________________________________________

save_dir = 'saved_files_rev_2/'

# Constraining known response to signal (positive or negative)
# Minimum sigma is 0.2

#  Hyperparameters
noise = 0.2

#  Computation parameters
N = 200  # population size (e.g. 16 laptop, 200 computing server)
n_sim = 30
ndt = 200   # 200
ncells = 300  #
ngens0 = 101
ngens = 301 #

L = 5.

#  Priors
par_limits = {
    'x': (-L, L),
    'y': (-L, L),
    #  Wide prior for a
    'a': ((0., 20.), (-20., 20.), (-20, 20.)),
    's': ((0.2, 1.2), (-1., 1.), (-1., 1.)),
}

par_choice_values = {}

# _______________________________________________________________________________
# morphogen_times contain signal functions (CHIR and FGF as function of time)

morphogen_times_0 = ((chir_2_5, fgf_0_5),)
time_pars_0 = (0., delta * 3, 4)

#  All experiments
morphogen_times = ((no_chir, fgf_no_pd),  # No Chir
                   (chir_2_3, fgf_no_pd),   # Chir 2-3
                   (chir_2_4, fgf_no_pd),  # Chir 2-4
                   (chir_2_5, fgf_no_pd),  # Chir 2-5
                   (chir_2_5, fgf_0_3),    # Ch 2-5 FGF 0-3
                   (chir_2_5, fgf_0_4),    # Ch 2-5 FGF 0-4
                   (chir_2_5, fgf_0_5),    # Ch 2-5 FGF 0-5
                   )
time_pars = (0., delta * 7, 8)

tt = np.linspace(time_pars[0], time_pars[1], 101)
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




# ___________________________________________________________________________
landscape_pars_celldiff = {
    'A0': 0.005,
    'init_cond': (0., 0.),
    'regime': mr_linear_2signals,
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

if __name__ == '__main__':
    # for exp in morphogen_times:
    #     chir = exp[0](tt)
    #     fgf = exp[1](tt)
    #     plt.plot(tt/delta/2.+1.5, chir, lw=2, label='CHIR')
    #     plt.plot(tt/delta/2+1.5, fgf, lw=2, label='FGF')
    #     plt.ylim((-0.5, 2))
    #     plt.legend()
    #     plt.show()

    print('N = ', N)
    for sim in range(n_sim):

        # Set up for optimizing the first 4 timepoints
        fitness_pars_celldiff = {
            'ncells': ncells,  #
            'cell_data': cell_dataset_0,
            'init_state': 0,
            'attractor_states': (0,),
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

        mod = start_module_list[0]
        mod.par_limits = {'x': (-4., 0.), 'y': (-3., 3.)}  # Epi on the left
        mod.x = np.random.uniform(-4., 0.)
        mod.y = np.random.uniform(-2., 2.)

        # Constraining known response to signal (positive or negative):
        # lowering FGF - Epi bifurcates to AN -> correlated with FGF
        # Adding Chir - Epi bifurcates to main route -> anticorrelated with CHIR
        mod.par_limits['a'] = ((0., 20.), (-20., 0.), (0., 20.))
        # mod.par_limits['s'] = ((0.2, 1.2), (-1., 0.), (0., 1.))
        mod.a = np.array([np.random.uniform(low, high) for low, high in mod.par_limits['a']])
        # mod.s = np.array([np.random.uniform(low, high) for low, high in mod.par_limits['s']])

        start_module_list[1].par_limits = {'x': (-2., 0.), 'y': (-2., 2.)}  # Tr on the left
        start_module_list[1].x = np.random.uniform(-2., 0.)
        start_module_list[1].y = np.random.uniform(-2., 2.)

        start_module_list[2].par_limits = {'x': (-1., 1.), 'y': (-1., 1.)}  # CE around the center
        start_module_list[2].x = np.random.uniform(-1., 1.)
        start_module_list[2].y = np.random.uniform(-1., 1.)
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

            mod = module_list[3]
            mod.par_limits = {'x': (0., 4.), 'y': (0., 4.)}  # PN - upper quadrant
            mod.x = np.random.uniform(0., 4.)
            mod.y = np.random.uniform(0., 4.)
            #. Anticorrelated with both Chir and FGF
            mod.par_limits['a'] = ((0., 20.), (-20., 0.), (-20., 0.))
            # mod.par_limits['s'] = ((0.2, 1.2), (-1., 0.), (-1., 0.))
            mod.a = np.array([np.random.uniform(low, high) for low, high in mod.par_limits['a']])
            # mod.s = np.array([np.random.uniform(low, high) for low, high in mod.par_limits['s']])

            mod = module_list[4]
            mod.par_limits = {'x': (0., 4.), 'y': (-4., 0.)}  # M - lower quadrant
            mod.x = np.random.uniform(0., 4.)
            mod.y = np.random.uniform(-4., 0.)

            mod = module_list[5]
            mod.par_limits = {'x': (-5., -1.), 'y': (-4., 4.)}  # AN left upper
            mod.x = np.random.uniform(-5., -1.)
            mod.y = np.random.uniform(-4., 4.)
            # Anticorrelated with CHIR and FGF
            mod.par_limits['a'] = ((0., 20.), (-20., 0.), (-20., 0.))
            # mod.par_limits['s'] = ((0.2, 1.2), (-1., 0.), (-1., 0.))
            mod.a = np.array([np.random.uniform(low, high) for low, high in mod.par_limits['a']])
            # mod.s = np.array([np.random.uniform(low, high) for low, high in mod.par_limits['s']])

        # ____________________________________________________________________________________________________________

        # Reset fitness
        for landscape in P.landscape_list:
            landscape.fitness = -np.inf

        fitness_pars_celldiff = {
            'ncells': ncells,  #
            'cell_data': cell_dataset,  # full dataset
            'init_state': 0,
            'attractor_states': (3,),
            'non_attractor_states': (),
            'noise': noise,
            'penalty_weight': 0.1,
            'time_pars': time_pars,
            'morphogen_times': morphogen_times,
            'ndt': ndt,
            'weights': (1., 1., 0.1, 1., 1., 1., 1.),  # less weight for Ch2-4
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
        landscape.morphogen_times = (chir_2_5, fgf_0_3)
        times = (delta*0., delta*2., delta*4.)
        # for t in times:
        #     print('CHIR:', chir_2_5(t), 'FGF:', fgf_0_3(t))
        # for t in times:
        #     print('CHIR:', chir_2_5(t), 'FGF:', fgf_0_3(t))

        npoints = 201
        q = np.linspace(-L, L, npoints)
        xx, yy = np.meshgrid(q, q, indexing='xy')
        figures = visualize_all(landscape, xx, yy, times, density=0.45, plot_traj=False, color_scheme='order')
        for i in range(len(figures)):
            figures[i].savefig(results_dir + '/result_landscape_' + str(i) + '.png')
            plt.close(figures[i])

        landscape.morphogen_times = (chir_2_4, fgf_no_pd)
        times = (delta*4., delta*6.)
        # for t in times:
        #     print('CHIR:', chir_2_4(t), 'FGF:', fgf_no_pd(t))
        # for t in times:
        #     print()
        #     print('CHIR:', chir_2_4(t), 'FGF:', fgf_no_pd(t))
        figures = visualize_all(landscape, xx, yy, times, density=0.45, plot_traj=False, color_scheme='order')
        for i in range(len(figures)):
            figures[i].savefig(results_dir + '/result_landscape_' + str(i+3) + '.png')
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
