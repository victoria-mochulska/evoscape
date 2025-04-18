import numpy as np

from . import Landscape


class Somitogenesis_Landscape(Landscape):
    # fitness_pars: time_pars, init_state, ncells, t0_shift, noise, high_value, low_value, t_stable, penalty_weight

    def get_kymo(self, ncells, time_pars, init_state, t0_shift=1., noise=0., ndt=100):
        kymo = np.zeros((ncells, time_pars[2]))
        for cell_ind in range(ncells):
            self.morphogen_times = (t0_shift * cell_ind,)
            self.init_cells(1, init_state, noise=noise)
            traj, states = self.run_cells(*time_pars, noise, ndt=ndt)
            kymo[cell_ind] = traj[0, 0, :]  # x-coordinate of the first (and only) cell in time
        self.morphogen_times = (0.,)
        self.result = kymo
        return kymo

    def get_fitness(self, fitness_pars):
        time_pars = fitness_pars['time_pars']
        ncells = fitness_pars['ncells']
        noise = fitness_pars['noise']
        ndt = fitness_pars['ndt']
        kymo = self.get_kymo(ncells, time_pars, fitness_pars['init_state'], fitness_pars['t0_shift'], noise, ndt)
        high = (kymo[:, -1] - fitness_pars['high_value'] > 0.).astype(int)
        low = (kymo[:, -1] - fitness_pars['low_value'] < 0.).astype(int)
        cross_high = np.where(high[1:-1] * (high[2:] - high[:-2]) != 0)[0] + 1
        cross_low = np.where(low[1:-1] * (low[2:] - low[:-2]) != 0)[0] + 1
        all_ind = np.concatenate((cross_high, cross_low))
        all_cross = np.concatenate((np.ones(len(cross_high), dtype=int), np.zeros(len(cross_low), dtype=int)))[
            np.argsort(all_ind)]
        n_boundaries = np.sum(np.diff(all_cross) != 0)
        penalty = fitness_pars['penalty_weight'] * np.sum((kymo[:, -fitness_pars['t_stable']:] -
                                                           np.tile(np.array([kymo[:, -1]]).T,
                                                                   (1, fitness_pars['t_stable']))) ** 2)
        fitness = n_boundaries - penalty
        return fitness
