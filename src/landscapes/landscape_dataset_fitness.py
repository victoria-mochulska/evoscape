import numpy as np

from helper_functions import kl_distance, d1_distance
from landscape_class import Landscape

result_distance = kl_distance
# result_distance = d1_distance


class CellDiff_Dataset_Landscape(Landscape):

    def get_fitness(self, fitness_pars):
        time_pars = fitness_pars['time_pars']
        times = np.linspace(time_pars[0], time_pars[1], time_pars[2])

        cell_data_matrices = fitness_pars['cell_data']  # 1st dimension is experimental conditions
        ncond = len(cell_data_matrices)
        morphogen_times = fitness_pars['morphogen_times']
        assert len(morphogen_times) == ncond, "The numbers of conditions do not match"

        self.result = []
        fitness = 0.
        penalty = 0.
        for k in range(ncond):
            self.morphogen_times = morphogen_times[k]
            cell_data = cell_data_matrices[k]
            cell_states = np.zeros(cell_data_matrices[0].shape)  # shape = (timepoints, states)
            self.init_cells(fitness_pars['ncells'], fitness_pars['init_state'], noise=fitness_pars['noise'])

            traj, states = self.run_cells(*time_pars, fitness_pars['noise'], ndt=fitness_pars['ndt'])
            for timepoint in range(len(times)):
                for cell_state in range(cell_states.shape[1]):
                    cell_states[timepoint, cell_state] = np.sum(states[:, timepoint] == cell_state)
            cell_states = cell_states / fitness_pars['ncells']  # normalized to give probability
            self.result.append(cell_states)

            fitness -= result_distance(cell_states, cell_data, None) / len(times)

            if fitness_pars['penalty_weight'] != 0:
                # for timepoint in range(len(times)):
                for timepoint in (0, len(times) - 1):
                    for state in fitness_pars['attractor_states']:
                        # for state in range(cell_states.shape[1]):
                        if cell_data[timepoint, state] != 0:
                            # attractor: a deterministic trajectory starting from a state has to end up in the same state
                            self.init_cells(1, state, noise=0.)
                            traj, states = self.run_cells(0., 10, 11, 0., ndt=50, frozen=True,
                                                          t_freeze=times[timepoint])
                            penalty += float(states[0, -1] != states[0, 0])

                    for state in fitness_pars['non_attractor_states']:
                        if cell_data[timepoint, state] != 0:
                            # non-attractor: a deterministic trajectory starting from a state has to exit it
                            self.init_cells(1, state, noise=0.)
                            traj, states = self.run_cells(0., 10, 11, 0., ndt=50, frozen=True,
                                                          t_freeze=times[timepoint])
                            penalty += float(states[0, -1] == states[0, 0])

        fitness -= penalty * fitness_pars['penalty_weight']
        return fitness

# ____________________________________________________________________________________________________________________
    # Calculate weights for each cell state to account for the total number of cells observed in given state:
    # column_weights = 1 / np.sum(cell_data, axis=0)
    # column_weights[np.isinf(column_weights)] = 0.
    # column_weights *= cell_data.shape[1]/sum(column_weights)  # normalize: sum of weights over states/columns
    # print('Weights', column_weights)
    # column_weights = np.tile(column_weights, (cell_data.shape[0], 1))  # repeat for all t


    # def get_cell_state(self, coordinate):
    #     dist = np.zeros(len(self.module_list))
    #     for i, module in enumerate(self.module_list):
    #         dist[i] = np.linalg.norm(coordinate - np.array((module.x, module.y)))
    #     return np.argmin(dist)

    # def get_avg_coordinates(self, ncells, times, noise, init_state=0, avg_over=1):
    #     avg_coordinates = np.zeros((ncells, 2))
    #     for icell in range(ncells):
    #         traj = self.get_trajectory_noisy(times,
    #                                          init_cond=(self.module_list[init_state].x, self.module_list[init_state].y),
    #                                          noise=noise)
    #         avg_coordinates[icell] = np.mean(traj[:, -avg_over:], axis=1)
    #     return avg_coordinates
