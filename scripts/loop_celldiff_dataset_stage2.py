import random
import os
import warnings
import pandas as pd

from evoscape.population_class import Population
from evoscape.landscapes.landscape_dataset_fitness import CellDiff_Dataset_Landscape
from evoscape.modules.module_class import Node
from evoscape.morphogen_regimes import mr_piecewise
from evoscape.landscape_visuals import *
from evoscape.helper_functions import plot_compare_cell_proportions, get_cell_data
from evoscape.module_helper_functions import modules_from_txt

warnings.simplefilter('ignore')

# _____________________________________________________________________________
save_dir = 'saved_files_10/'    # where to save simulations
data_dir = 'saved_files_4/CellDiff_Dataset_Landscape/'   # to load Stage 1 landscapes

file_name = data_dir + 'optimization_log.csv'


#  Hyperparameters
delta = 2.
noise = 0.2

#  Computation parameters
N = 200  # population size
n_sim = 300    # max number
ndt = 200   # integration steps per traj. timestep
ncells = 500   # was 300
ngens = 301

L = 5.  # field size

#  Priors
par_limits = {
    'x': (-L, L),
    'y': (-L, L),
    'a': (0., 16.),
    's': (0.1, 1.5),
}

par_choice_values = {}

# _______________________________________________________________________________

# Regime 1 - FGF, no Chir               (red)
# t0
# Regime 2 - Chir + FGF                (purple+red)
# t1
# Regime 3 - Chir (+PD)                (purple+green)
# ___________ new: __________________________________
# t2
# Regime 4 - Chir + end. FGF           (purple)
# t3
# Regime 5 - endogenous FGF (90% FGF)   (empty)


# #   Training data # #

#  Initial training: regimes 1, 2, 3

#  Ch 2-5 FGF 0-3:     *
#  regime 1 until day 2, regime 2 day 2-3, regime 3 day 3-5
# t0 = day 2, t1 = day 3, t2, t3 = never
# (D2, D3, 100., 100.)

#  Ch 2-5 FGF 0-4:      *
#  regime 1 until day 2, regime 2 day 2- 4, regime 3 day 4-5
# t0 = day 2, t1 = day 4

#  Ch 2-5 FGF 0-5:      *
#  regime 1 until day 2, regime 2 day 2-5
# t0 = day 2, t1 = day 5; t2, t3 = never
# (D2, 100., 100., 100.)

#  ________ New :  ______________
# Training for regimes 1, 4, 5

# No Chir:
# regime 1 until day 3, regime 5 day 3-5
# t0 = t1 = t2 = t3 = day 3
# (D3, D3, D3, D3)

# Chir 2-3:
# regime 1 until day 2, regime 2 day 2-3, regime 5 day 3-5
# t0 = day 2, t1=t2=t3 day 3
# (D2, D3, D3, D3)

# Chir 2-4:
# regime 1 until day 2, regime 2 day 2-3, regime 4 day 3-4, regime 5 day 4-5
# t0 = day 2, t1=t2 = day 3, t3 = day 4
# (D2, D3, D3, D4)

# Chir 2-5:
# regime 1 until day 2, regime 2 day 2-3, regime 4 day 3-5
# t0 = day 2, t1=t2 = day 3; t3=never
# (D2, D3, D3, 100.)

# + keep one of the old data   (Ch 2-5 FGF 0-3)

# ______________________________________________
# From prev training:  keep regimes 2, 3 unchanged
# for modules 0-4: fixed a1-2, s1-2, x, y; mutable a0, s0, a3-4, s3-4
# Mutable: all pars of new module (purple, AN)
# ______________________________________________________________________

#
filenames = ('NoCh.txt', 'Ch2-3.txt', 'Ch2-4.txt', 'Ch2-5.txt',
             'Ch2-5_FGF0-3.txt', 'Ch2-5_FGF0-4.txt', 'Ch2-5_FGF0-5.txt')
# plot traj: 0, 1, 3, 4, 5

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
    # print(filename)
    # print(cell_data)


# timepoint 0 = D1.5, 1 = D2, 3 = D3, 5 = D4, 7 = D5
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
    'morphogen_times': morphogen_times[0],
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
    'cell_data': cell_dataset,  # full dataset
    'init_state': 0,
    'attractor_states': (),
    'non_attractor_states': (),
    'noise': noise,
    'penalty_weight': 0.,
    'time_pars': time_pars,
    'morphogen_times': morphogen_times,
    'ndt': ndt,
    'weights': (1., 1., 2., 1., 1., 1., 1.),  # more weight for Ch2-4
}

# ________________________________________________

# load list of timecodes of Stage 1 optimization
log = pd.read_csv(file_name, sep='\t', names=['Timecode', 'Init timecode', 'Fitness'], skiprows=1)
# fitness_threshold = -0.2
n_landscapes = 240
fitness_threshold = np.partition(log['Fitness'], -n_landscapes)[-n_landscapes]
init_timecodes = list(log['Timecode'][log['Fitness'] >= fitness_threshold])
n_sim = np.min((n_sim, len(init_timecodes)))
gen = 300


if __name__ == '__main__':
    print('N = ', N)
    for sim in range(n_sim):
        timecode1 = init_timecodes[sim]

        # Load modules from Stage 1 optimization
        modules_filename = data_dir + timecode1 + '/' + timecode1 + '_module_list_' + str(gen) + '.txt'
        with open(modules_filename, 'r') as f:
            module_list = modules_from_txt(modules_filename)

        # Add new signalling regimes
        # Regime 4 ~ Regime 2, Regime 5 ~ Regime 1 (FGF Vs End. FGF)
        # Add a3=a1, s3=s1; a4=a0, s4=s0
        # immutable pars: x, y; immutable regimes: 1, 2 (mutable: 0, 3, 4)
        for module_ind, module in enumerate(module_list):
            # initial condition for endogenous FGF based on FGF
            module.a = np.append(module.a, module.a[[1, 0]])
            module.s = np.append(module.s, module.s[[1, 0]])
            # random initial condition for the new regimes
            # module.a = np.append(module.a, np.random.uniform(*par_limits['a'], 2))
            # module.s = np.append(module.s, np.random.uniform(*par_limits['s'], 2))
            if module_ind != 3:  # for green module, keep everything mutable
                module.set_immutable_idx([1, 2])
                module.remove_mutable_parameter('x')
                module.remove_mutable_parameter('y')

        # Set up population (init fitness = -inf)
        P = Population(N, CellDiff_Dataset_Landscape, landscape_pars_celldiff, prob_pars_celldiff,
                       False, par_limits, par_choice_values, start_module_list=module_list)

        # Add random M5 (purple module) with all pars mutable:
        for landscape in P.landscape_list:
            landscape.module_list.append(Node.generate(par_limits, par_choice_values,
                                                       n_regimes=landscape_pars_celldiff['n_regimes']))

        print('# '+str(sim))

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

        with open(results_dir + '/result_fitness_traj.npy', 'wb') as f:
            np.save(f, fitness_traj)

        landscape = P.landscape_list[0]

        #  Plot result Vs target proportions
        for k in range(len(cell_dataset)):
            fig = plot_compare_cell_proportions(cell_dataset[k], landscape.result[k], col_labels, col_colors,
                                                row_labels=row_labels)
            fig.show()
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
        experiments = (0, 1, 2, 3, 4, 5, 6)    # all experiments
        n = 50

        for exp in experiments:
            landscape.morphogen_times = fitness_pars_celldiff['morphogen_times'][exp]
            landscape.init_cells(n, 0, noise)
            fig = get_and_plot_traj(landscape, 0, delta * 7, 51, L, noise, frozen=False)
            # fig.show()
            fig.savefig(results_dir + '/result_cell_trajectories_'+str(exp)+'.png', bbox_inches='tight')
            plt.close(fig)

            fig = plot_cells(landscape, L)
            # fig.show()
            fig.savefig(results_dir + '/result_final_state_'+str(exp)+'.png', bbox_inches='tight')
            plt.close(fig)

        plt.close('all')

        log_filename = save_dir + '/' + landscape.__class__.__name__ + '/optimization_log.csv'
        if not os.path.exists(log_filename):
            with open(log_filename, 'a') as f:
                f.write('# Main timecode\tInit timecode\tFitness\n')

        with open(log_filename, 'a') as f:
            f.write('\t'.join([timecode2, timecode1, str(P.landscape_list[0].fitness)]) + '\n')
