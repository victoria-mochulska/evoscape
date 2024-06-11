from landscape_class import Landscape


class CellDiff_Landscape(Landscape):
    # fitness_pars: cell_data, ncells, noise, init_state

    # def get_cell_state(self, coordinate):
    #     dist = np.zeros(len(self.module_list))
    #     for i, module in enumerate(self.module_list):
    #         dist[i] = np.linalg.norm(coordinate - np.array((module.x, module.y)))
    #     return np.argmin(dist)

    def get_avg_coordinates(self, ncells, times, noise, init_state=0, avg_over=1):  # ndt for integration? default=10
        avg_coordinates = np.zeros((ncells, 2))
        for icell in range(ncells):
            traj = self.get_trajectory_noisy(times,
                                             init_cond=(self.module_list[init_state].x, self.module_list[init_state].y),
                                             noise=noise)
            avg_coordinates[icell] = np.mean(traj[:, -avg_over:], axis=1)
        return avg_coordinates

    # def get_end_coordinates(self, ncells, times, noise, init_state=0): #ndt for integration? default=10
    #     end_coordinates = np.zeros((ncells,2))
    #     for icell in range(ncells):
    #         traj = self.get_trajectory_noisy(times,
    #                                     init_cond=(self.module_list[init_state].x, self.module_list[init_state].y),
    #                                          noise=noise)
    #         end_coordinates[icell] = np.mean(traj[:, -20:], axis = 1)
    #     return end_coordinates
    #
    # def get_start_coordinates(self, ncells, noise, init_state=0):
    #     start_coordinates = np.zeros((ncells,2))
    #     for icell in range(ncells):
    #         traj = self.get_trajectory_noisy(np.linspace(-100., -50., 51),
    #                                   init_cond=(self.module_list[init_state].x, self.module_list[init_state].y),
    #                                          noise=noise)
    #         start_coordinates[icell] = np.mean(traj[:, -10:], axis = 1)
    #         return start_coordinates

    def get_fitness(self, fitness_pars):
        time_pars = fitness_pars['time_pars']
        cell_states = np.zeros(fitness_pars['cell_data'].shape)  ## shape = (timepoints, states)
        self.init_cells(fitness_pars['ncells'], fitness_pars['init_state'], noise=fitness_pars['noise'])
        times = np.linspace(time_pars[0], time_pars[1], time_pars[2])
        traj, states = self.run_cells(*time_pars, fitness_pars['noise'], ndt=50)
        for timepoint in range(len(times)):
            for cell_state in range(cell_states.shape[1]):
                cell_states[timepoint, cell_state] = np.sum(states[:, timepoint] == cell_state)
        cell_states = cell_states / fitness_pars['ncells'] * 100.  # normalized to give percent
        self.result = cell_states

        penalty = 0.
        for timepoint in range(len(times)):
            for state in range(cell_states.shape[1]):
                if fitness_pars['cell_data'][timepoint, state] != 0:
                    # and
                    if state in fitness_pars['attractor_states']:
                        # attractor: a deterministic trajectory starting from a state has to end up in the same state
                        self.init_cells(1, state, noise=0.)
                        traj, states = self.run_cells(0., 20, 21, 0., ndt=50, frozen=True, t_freeze=times[timepoint])
                        penalty += float(states[0, -1] != states[0, 0])

                    if state in fitness_pars['non_attractor_states']:
                        # non-attractor: a deterministic trajectory starting from a state has to exit it
                        self.init_cells(1, state, noise=0.)
                        traj, states = self.run_cells(0., 15, 16, 0., ndt=50, frozen=True, t_freeze=times[timepoint])
                        penalty += float(states[0, -1] == states[0, 0])

        penalty *= fitness_pars['penalty_weight']  # penalty weight was 10 by default

        # Calculate weights for each cell state to account for the total number of cells observed in given state:
        column_weights = 1 / np.sum(fitness_pars['cell_data'], axis=0)
        # print('N cells', column_weights)
        column_weights[np.isinf(column_weights)] = 0.
        column_weights /= np.sum(column_weights)  # normalize: sum of weights over states/columns = 0
        # print('Weights', column_weights)
        column_weights = np.tile(column_weights, (fitness_pars['cell_data'].shape[0], 1))  # repeat for all t
        fitness = -np.sum(column_weights * np.abs(cell_states - fitness_pars['cell_data'])) / 2. / len(times) * \
                  fitness_pars['cell_data'].shape[1] - penalty
        return fitness

