import random
from copy import copy
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm, CenteredNorm
from skimage.measure import label

from .morphogen_regimes import mr_sigmoid
from .auxiliary_functions import integrate_EM
from .landscape_visuals import fp_type_colors, order_colors
from .module_class import Node  # , UnstableSpiral

# import slow_regimes
# import proliferation_regimes


class Landscape:
    def __init__(self, module_list=(), A0=0., init_cond=(0., 1.), mode=None, regime=mr_sigmoid,
                 n_regimes=2, morphogen_times=(0.,), used_fp_types=(Node,), immutable_pars_list=()):
        self.module_list = []
        for ind in range(len(module_list)):
            self.module_list.append(deepcopy(module_list[ind]))
            for par_name in immutable_pars_list:
                if par_name in self.module_list[ind].mutable_parameters_list:
                    self.module_list[ind].remove_mutable_parameter(par_name)
        self.A0 = A0
        self.mode = mode
        self.regime = regime
        self.n_regimes = n_regimes
        self.morphogen_times = morphogen_times
        self.used_fp_types = used_fp_types
        self.init_cond = init_cond
        self.max_n_modules = 15
        self.fitness = None
        self.result = None
        # ___ can contain cells ____
        self.cell_coordinates = None
        self.cell_states = None
        self.trajectories = None

    def __repr__(self):
        if not self.module_list:
            return 'Empty landscape'
        repr_str = 'Landscape with modules:'
        for module in self.module_list:
            module_str = module.__str__()
            repr_str += '\n' + module_str + ','
        repr_str = repr_str[:-1]
        return repr_str

    # ____________________________________________________________________________________

    @staticmethod
    def local_weight(r, sig):
        weight = np.exp(-0.5 * (r / sig) ** 2)
        return weight

    @staticmethod
    def fixed_point(module, x, y, R=1000, r=None):
        dx = np.zeros(x.shape)
        dy = np.zeros(x.shape)
        if r is None:
            r = np.sqrt(x ** 2 + y ** 2)
        J = module.J
        dx[r < R] = J[0][0] * x[r < R] + J[0][1] * y[r < R]
        dy[r < R] = J[1][0] * x[r < R] + J[1][1] * y[r < R]
        return dx, dy

    def __call__(self, t, q, return_weights=False):  # evaluate flow at (q, t)
        # q.shape = (2, m, n)  - grid of cell coordinates
        x = q[0]
        y = q[1]
        w = np.zeros((len(self.module_list), *x.shape))
        sig = np.zeros((len(self.module_list)))
        sign = np.zeros((len(self.module_list)), dtype='int')
        curl = np.zeros((len(self.module_list)), dtype='bool')
        dx, dy = np.zeros((len(self.module_list), *x.shape)), np.zeros((len(self.module_list), *x.shape))
        for i, module in enumerate(self.module_list):
            V, sig[i], A = module.get_current_pars(t, self.regime, *self.morphogen_times)
            if module.__class__.__name__ == 'Node' or module.__class__.__name__ == 'NegCenter':
                sign[i] = -1
            else:
                sign[i] = +1
            if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
                curl[i] = 1
            # mu[i] = 0. if module.mu is None else module.mu

            xr = x - module.x
            yr = y - module.y
            r = np.sqrt(xr ** 2 + yr ** 2)
            w[i, :] = A * self.local_weight(r, sig[i])
            # TODO: remove or change self.mode
            dx[i, :], dy[i, :] = self.fixed_point(module, xr, yr, R=1000., r=r)
        if return_weights:
            # potential = -np.sum(w*(mu*sig**2)[:, np.newaxis, np.newaxis], axis=0) + self.A0/4*(x**4+y**4)
            # rot_potential = np.sum(w*(omega*sig**2)[:, np.newaxis, np.newaxis], axis=0)
            potential = np.sum(w * (~curl * sign * sig ** 2)[:, np.newaxis, np.newaxis], axis=0) + self.A0 / 4 * (
                        x ** 4 + y ** 4)
            rot_potential = np.sum(w * (curl * sign * sig ** 2)[:, np.newaxis, np.newaxis], axis=0)
        # w0 = np.tile(w0, (2, 1))
        # derivs = self.A0 * w0 * np.array((-x, -y)) + (np.sum(w * dx, axis=0), np.sum(w * dy, axis=0))
        derivs = self.A0 * np.array((-x ** 3, -y ** 3)) + (np.sum(w * dx, axis=0), np.sum(w * dy, axis=0))
        if return_weights:
            return derivs, potential, rot_potential
        return derivs

    def flow(self, t, q, slow):
        derivs = self(t, q)
        if slow is None:
            slowing_rate = 0.
        else:
            slowing_rate = slow(t, q, self)
        return derivs * (1. - slowing_rate)

    # def frozen_landscape(self, t, q, t_freeze):
    #     return self(t_freeze, q)

    # ____________________________________________________________________________________________________________
    def get_cell_states_static(self, coordinate=None):
        if coordinate is None:
            coordinate = self.cell_coordinates

        dist = np.empty((coordinate.shape[1], len(self.module_list)))
        for i, module in enumerate(self.module_list):
            dist[:, i] = np.linalg.norm(coordinate.T - np.array((module.x, module.y)), axis=1)
        states = np.argmin(dist, axis=1)

        return states

    def get_cell_states(self, t, coordinate=None, measure='gaussian'):
        if coordinate is None:
            coordinate = self.cell_coordinates

        if measure == 'dist':
            dist = np.empty((coordinate.shape[1], len(self.module_list)))
            for i, module in enumerate(self.module_list):
                dist[:, i] = np.linalg.norm(coordinate.T - np.array((module.x, module.y)), axis=1)
            states = np.argmin(dist, axis=1)
        elif measure == 'gaussian':
            prob = np.zeros((coordinate.shape[1], len(self.module_list) + 1))
            for i, module in enumerate(self.module_list):
                V, st, at = module.get_current_pars(t, self.regime, *self.morphogen_times)
                prob[:, i] = np.exp(
                    -np.sum((coordinate.T - np.array((module.x, module.y))) ** 2, axis=1) / 2. / st ** 2) / st ** 2
            # print(prob/2/np.pi)
            prob = (prob.T / np.sum(prob, axis=1)).T
            prob[:, -1] = 0.3  # probability threshold
            # print(prob*100)
            states = np.argmax(prob, axis=1)
        return states

    @property
    def n(self):
        return np.sum(~np.isnan(np.sum(self.cell_coordinates, axis=0)))

    def init_cells(self, n, init_cond, noise=0.):
        if isinstance(init_cond, int):
            module0 = self.module_list[init_cond]
            init_cond = (module0.x, module0.y)
        elif init_cond is None:
            init_cond = self.init_cond
        init_cond = np.asarray(init_cond)
        self.cell_coordinates = np.ones((2, n)) * np.nan
        if init_cond.shape == (2, n):
            self.cell_coordinates = init_cond.astype('float')
        elif init_cond.shape == (2,):
            self.cell_coordinates = np.tile(init_cond.astype('float'), (n, 1)).T
        elif len(init_cond) == len(self.module_list) and np.sum(init_cond) == n:
            module_locs = np.array([(module.x, module.y) for module in self.module_list])
            self.cell_coordinates = np.repeat(module_locs, init_cond, axis=0).T
        else:
            print('Wrong shape of init_cond input')
        if noise != 0.:
            self.cell_coordinates += noise * np.random.randn(2, n)
        self.cell_states = self.get_cell_states_static()

    def reset_cells(self):
        self.cell_coordinates = None
        self.cell_states = None

    def add_cell(self, coord, noise=0.):
        if self.cell_coordinates is None:
            print('Initialize cells')
        else:
            coord = np.asarray(coord)
            if coord.shape == (2,):
                coord = np.reshape(coord, (2, 1))
            self.cell_coordinates = np.append(self.cell_coordinates, coord, axis=1)

    def plot_cells(self, L):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        coord = self.cell_coordinates
        states = self.get_cell_states_static()
        cmap_state = ListedColormap(['indianred', 'tab:orange', 'gold', 'tab:green', 'tab:blue', 'tab:purple', 'm'])
        norm_state = BoundaryNorm(np.arange(8) - 0.5, cmap_state.N)
        ax.scatter(coord[0], coord[1], s=8, alpha=0.3, c=states, cmap=cmap_state, norm=norm_state, edgecolors=None)
        ax.set_xlim([-L, L])
        ax.set_ylim([-L, L])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)
        plt.show()

    def run_cells(self, t0, tf, nt, noise=0., ndt=50, frozen=False, t_freeze=None):
        traj = np.empty((*self.cell_coordinates.shape, nt), dtype='float')  # shape = (2, n, nt)
        states = np.empty((self.cell_coordinates.shape[1], nt), dtype='int')  # shape = (n, nt)
        t = t0
        y = self.cell_coordinates
        traj[:, :, 0] = y
        states[:, 0] = self.get_cell_states(t)
        Delta_t = (tf - t0) / (nt - 1)
        dt = Delta_t / ndt
        sqrt_dt = np.sqrt(dt)
        if frozen:
            def f(t, q):
                return self(t_freeze, q)
        else:
            f = self
        for Delta_step in range(1, nt):
            for dt_step in range(ndt):
                y += f(t, y) * dt + noise * np.random.standard_normal(y.shape) * sqrt_dt
                t += dt
            traj[:, :, Delta_step] = y
            states[:, Delta_step] = self.get_cell_states(t)
        return traj, states

    def run_cells_prolif(self, t0, tf, nt, prolif_pars, remove_pars, noise=0., ndt=50, frozen=False, t_freeze=None):
        prolif_rate = prolif_pars['regime']
        remove_rate = remove_pars['regime']
        traj = np.ones((*self.cell_coordinates.shape, nt), dtype='float') * np.nan
        states = np.nan * np.ones((self.cell_coordinates.shape[1], nt), dtype='int')
        added_coords = np.array(((), ()))
        removed_coords = np.array(((), ()))
        t = t0
        traj[:, :, 0] = self.cell_coordinates
        states[:, 0] = self.get_cell_states(t)
        Delta_t = (tf - t0) / (nt - 1)
        dt = Delta_t / ndt
        sqrt_dt = np.sqrt(dt)
        if frozen:
            def f(t, q):
                return self(float(t_freeze), q)
        else:
            f = self
        for Delta_step in range(1, nt):
            for dt_step in range(ndt):
                y = self.cell_coordinates
                y += f(t, y) * dt + noise * np.random.standard_normal(y.shape) * sqrt_dt
                t += dt
                # TODO: calculate rates only for non-NaN cells
                prolif = np.random.random(y.shape[1]) < prolif_rate(t, y, self.get_cell_states(), prolif_pars) * dt
                prolif[np.isnan(y[0])] = False
                remove = np.random.random(y.shape[1]) < remove_rate(t, y, self.get_cell_states(), remove_pars) * dt
                remove[np.isnan(y[0])] = False
                if np.any(prolif):
                    # print(Delta_step, ':', dt_step, ' Adding', np.sum(prolif))
                    add = np.sum(prolif)
                    added_coords = np.append(added_coords, self.cell_coordinates[:, prolif], axis=1)
                    self.add_cell(self.cell_coordinates[:, prolif])
                    traj = np.append(traj, np.nan * np.ones((2, add, nt)), axis=1)
                    states = np.append(states, np.nan * np.ones((add, nt)), axis=0)
                    remove = np.append(remove, np.zeros(add, dtype='bool'))
                removed_coords = np.append(removed_coords, self.cell_coordinates[:, remove], axis=1)
                self.cell_coordinates[:, remove] = np.nan
            traj[:, :, Delta_step] = self.cell_coordinates
            states[:, Delta_step] = self.get_cell_states(t)
            # print(added_coords)
        return traj, states, added_coords, removed_coords

    def plot_traj(self, L, traj, states, added_coords=None, removed_coords=None):
        cmap_state = ListedColormap(['indianred', 'tab:orange', 'gold', 'tab:green', 'tab:blue', 'tab:purple', 'm'])
        norm_state = BoundaryNorm(np.arange(8) - 0.5, cmap_state.N)
        cmap_time = 'viridis'
        nt = traj.shape[2]

        # traj, states = self.run_cells(t0, tf, nt, noise=noise, ndt=ndt, frozen=frozen, t_freeze=t_freeze)

        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].scatter(traj[0, :, :], traj[1, :, :], s=8, alpha=0.2, c=np.tile(np.arange(nt), (states.shape[0], 1)),
                      cmap=cmap_time, edgecolors=None)
        ax[1].scatter(traj[0, :, :], traj[1, :, :], s=8, alpha=0.2, c=states, cmap=cmap_state, norm=norm_state,
                      edgecolors=None)
        ax[0].set_title('Trajectories by time')
        ax[1].set_title('Trajectories by cell state')

        ax[2].scatter(added_coords[0], added_coords[1], s=17, alpha=0.35, color='green', marker='+', edgecolors=None,
                      label='Proliferation event', zorder=2)
        ax[2].scatter(removed_coords[0], removed_coords[1], s=15, alpha=0.35, color='m', marker='x', edgecolors=None,
                      label='Cell death event')
        ax[2].legend()
        for axi in range(3):
            ax[axi].set_xlim([-L, L])
            ax[axi].set_ylim([-L, L])
            ax[axi].set_xticks([])
            ax[axi].set_yticks([])
        plt.show()

    def get_and_plot_traj(self, t0, tf, nt, L, noise, ndt=50, slow=None, frozen=False, t_freeze=None, colors=None):
        if colors is None:
            colors = ['indianred', 'tab:orange', 'gold', 'tab:green', 'tab:blue', 'tab:purple', 'm']
        cmap_state = ListedColormap(colors)
        norm_state = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap_state.N)
        cmap_time = 'viridis'
        traj, states = self.run_cells(t0, tf, nt, noise=noise, ndt=ndt, frozen=frozen, t_freeze=t_freeze)

        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].scatter(traj[0, :, :], traj[1, :, :], s=6, alpha=0.2, c=np.tile(np.arange(nt), (states.shape[0], 1)),
                      cmap=cmap_time, edgecolor=None)
        ax[1].scatter(traj[0, :, :], traj[1, :, :], s=6, alpha=0.2, c=states, cmap=cmap_state, norm=norm_state,
                      edgecolor=None)
        ax[0].set_xlim([-L, L])
        ax[0].set_ylim([-L, L])
        ax[1].set_xlim([-L, L])
        ax[1].set_ylim([-L, L])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()

    # ____________________________________________________________________________________________________________
    # def get_trajectory(self, times, t0_shift=0., init_cond=None):  # RK45 from scipy
    #     if init_cond is None:
    #         init_cond = self.init_cond
    #     for module_ind in range(len(self.module_list)):
    #         self.module_list[module_ind].t0 += t0_shift
    #     traj = integrate.solve_ivp(self, (times[0], times[-1]), init_cond, t_eval=times, method='RK45').y
    #     for module_ind in range(len(self.module_list)):
    #         self.module_list[module_ind].t0 -= t0_shift
    #     return traj
    #

    # ___________________________________________________________________________________________________________________

    def visualize(self, xx, yy, times, density=0.5, plot=True, plot_weights=False, plot_velocities=False,
                  plot_nullclines=True, plot_traj=True,
                  traj_times=(0., 100., 150), traj_start=50, traj_init_cond=(0., 1.), traj_noise=0.,
                  color_scheme='fp_types',
                  return_figure=False):
        dX, dY = np.zeros((len(times), *xx.shape)), np.zeros((len(times), *xx.shape))
        curl = np.zeros((len(self.module_list)), dtype='bool')
        for i, module in enumerate(self.module_list):
            if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
                curl[i] = 1
        for it in range(len(times)):
            # w = np.zeros((len(self.module_list), *xx.shape))
            # dx, dy = np.zeros((len(self.module_list), *xx.shape)), np.zeros((len(self.module_list), *xx.shape))
            circles = []
            #
            for i, module in enumerate(self.module_list):
                V, sig, A = module.get_current_pars(times[it], self.regime, *self.morphogen_times)
                if color_scheme == 'fp_types':
                    color = fp_type_colors[module.__class__.__name__]
                elif color_scheme == 'order':
                    color = order_colors[i]
                else:
                    color = 'grey'

                circles.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                          fill=True, alpha=0.3 * np.sqrt(A), clip_on=True, linewidth=0))  ##

            (dX[it], dY[it]), potential, rot_potential = self(times[it], (xx, yy), return_weights=True)

            if plot:
                if plot_traj:
                    # calculate a trajectory in frozen landscape
                    self.init_cells(1, traj_init_cond, noise=traj_noise)
                    traj, states = self.run_cells(traj_times[0], traj_times[1], traj_times[2], noise=traj_noise,
                                                  ndt=50, frozen=True, t_freeze=times[it])

                if plot_weights:
                    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
                    # for iax in range(4):
                    # ax[iax].axis('off')
                    ax[0].imshow(potential, cmap='Greens', origin='lower',
                                 extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
                    ax[0].contour(xx, yy, potential, origin='lower', colors='w')

                    ax[1].imshow(rot_potential, cmap='RdBu_r', origin='lower', norm=CenteredNorm(0),
                                 extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
                    ax[1].contour(xx, yy, rot_potential, colors='w', linestyles='solid', origin='lower', levels=12)

                    for iax in range(2):
                        ax[iax].set_xticks([])
                        ax[iax].set_yticks([])

                        # can add nullclines but they make the plot busy
                        for i, module in enumerate(self.module_list):
                            ax[iax].scatter(module.x, module.y, marker='x', c='k')
                    circles_ax = ax[2]
                    stream_ax = ax[3]
                else:
                    fig, stream_ax = plt.subplots(1, 1, figsize=(6, 6))
                    circles_ax = stream_ax

                if plot_velocities:
                    # fig, ax = plt.subplots(1, 2, figsize=(11, 5))
                    velocities_sq = dX[it] ** 2 + dY[it] ** 2
                    velocities = np.sqrt(velocities_sq)

                    # vel_plot = ax[0].imshow(velocities, cmap='BuPu', origin='lower', vmin=0, vmax=1.)
                    # plt.colorbar(vel_plot)
                    print('Min velocity:', round(np.min(velocities), 3), ', Max:', round(np.max(velocities), 3),
                          ', Mean:', round(np.mean(velocities), 3), ', Median:', round(np.median(velocities), 3))
                    # stream_ax = ax[1]

                for i in range(len(self.module_list)):
                    circles_ax.add_patch(copy(circles[i]))
                    circles_ax.set_xlim((np.min(xx), np.max(xx)))
                    circles_ax.set_ylim((np.min(yy), np.max(yy)))

                if plot_velocities:
                    stream_ax.imshow(velocities, alpha=0.5, cmap='Greys', origin='lower',
                                     extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
                    fp_labels, nlabels = label(velocities_sq < 1e-3, return_num=True)
                    for l in range(nlabels):
                        # if np.sum(fp_labels == l) <= 50:
                        fp = velocities_sq == np.min(velocities_sq[fp_labels == l])
                        # if np.sum(fp_labels == l) > 20:
                        #     fp = (velocities_sq < 5e-4) * fp_labels == l
                        # else:
                        #     fp = fp_labels == l
                        stream_ax.scatter(xx[fp], yy[fp], marker='o', s=50, color='gold', edgecolor=None, zorder=10)
                    # stream_ax.(velocities_sq<1e-, cmap='viridis', origin='lower', extent = (np.min(xx), np.max(xx), np.min(yy), np.max(yy)), interpolation=None)

                circles_ax.streamplot(xx, yy, dX[it], dY[it], density=density, arrowsize=2., arrowstyle='->',
                                      linewidth=1,
                                      color='grey')
                stream_ax.streamplot(xx, yy, dX[it], dY[it], density=density, arrowsize=2., arrowstyle='->',
                                     linewidth=1,
                                     color='grey')
                ######
                # test_fig, ax = plt.subplots(1,1)
                # ax.streamplot(xx, yy, dX_test, dY_test, density=density, arrowsize=2., arrowstyle='->', linewidth=1,
                #                      color='grey')
                # test_fig.show()
                #########
                # ax[1].streamplot(xx, yy, dX[it], dY[it], density=density, arrowsize=2., linewidth=4, color=np.sqrt(dX[it]**2+dY[it]**2), cmap='Greys')
                if plot_nullclines:
                    circles_ax.contour(xx, yy, dX[it], (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
                    circles_ax.contour(xx, yy, dY[it], (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)
                    stream_ax.contour(xx, yy, dX[it], (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
                    stream_ax.contour(xx, yy, dY[it], (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)
                if plot_traj:
                    stream_ax.plot(traj[0, 0, traj_start:], traj[1, 0, traj_start:], lw=2.5, color='forestgreen')
                    # print('plotting trajectory')
                    # print(traj[0, 0])
                    # stream_ax.plot(traj[0], traj[1], color='darkorange', linewidth=2.5, alpha=1.)
                stream_ax.set_xlim([np.min(xx), np.max(xx)])
                stream_ax.set_ylim([np.min(yy), np.max(yy)])
                stream_ax.set_xticks([])
                stream_ax.set_yticks([])
                circles_ax.set_xticks([])
                circles_ax.set_yticks([])
                plt.show()

        if return_figure:
            return fig
        return dX, dY

    # ____________________________________________________________________________________

    def get_fitness(self, fitness_pars):
        raise NotImplementedError

    def add_module(self, M):
        self.module_list.append(deepcopy(M))

    def del_module(self, del_ind):
        del self.module_list[del_ind]

    def mutate(self, par_limits, par_choice_values, prob_pars, fitness_pars):
        r = np.random.uniform()
        if r < prob_pars['prob_add'] or len(self.module_list) == 0:
            # print('Adding,', 'len =', len(self.module_list), ', r =', r)  ################################
            fp_type = random.choice(self.used_fp_types)
            self.add_module(fp_type.generate(par_limits, par_choice_values, n_regimes=self.n_regimes))  ## add regime
        elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] and len(self.module_list) > 1:
            # print('Deleting,', 'len =', len(self.module_list), ', r =', r)     ##############################
            del_ind = np.random.choice(len(self.module_list))
            self.del_module(del_ind)
        elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] + prob_pars['prob_shuffle'] and len(
                self.module_list) > 1:
            # print('Shuffling,', 'len =', len(self.module_list), ', r =', r) ##################################
            random.shuffle(self.module_list)
        else:
            # print('Modifying,', ', r =', r)
            mod_ind = np.random.choice(len(self.module_list))
            self.module_list[mod_ind].mutate(par_limits, par_choice_values)
        self.fitness = self.get_fitness(fitness_pars)

    def mutate_and_return(self, par_limits, par_choice_values, prob_pars, fitness_pars):
        r = np.random.uniform()
        if r < prob_pars['prob_add'] or len(self.module_list) == 0:
            fp_type = random.choice(self.used_fp_types)
            self.add_module(fp_type.generate(par_limits, par_choice_values, n_regimes=self.n_regimes))  ## add regime
        elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] and len(self.module_list) > 1 \
                or len(self.module_list) > self.max_n_modules:
            del_ind = np.random.choice(len(self.module_list))
            self.del_module(del_ind)
        elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] + prob_pars['prob_shuffle'] and len(
                self.module_list) > 1:
            random.shuffle(self.module_list)
        else:
            mod_ind = np.random.choice(len(self.module_list))
            self.module_list[mod_ind].mutate(par_limits, par_choice_values)
        self.fitness = self.get_fitness(fitness_pars)
        return self


# ___________________________________________________________________________

class Noisy_Somitogenesis_Landscape(Landscape):
    # do deterministic + noisy, take min n of stripes
    # fitness_pars: ncells, t0_shift, high_value, low_value, t_stable, eps,  noise, nsym, ndt
    def get_kymo_noisy(self, times, ncells=50, t0_shift=1., noise=0.05, nsym=5, ndt=10):
        kymos = np.zeros((nsym, ncells, len(times)))
        for cell_ind in range(ncells):
            for module_ind in range(len(self.module_list)):
                self.module_list[module_ind].t0 += t0_shift
            for i_sym in range(nsym):
                sol = integrate_EM(self, times[0], times[-1], len(times), self.init_cond, noise, ndt)
                kymos[i_sym, cell_ind, :] = sol[0]
        for module_ind in range(len(self.module_list)):
            self.module_list[module_ind].t0 -= t0_shift * ncells
        return kymos

    def get_fitness(self, times, fitness_pars):
        if 'ndt' not in fitness_pars:
            ndt = 10
        else:
            ndt = fitness_pars['ndt']
        if 'nsym' not in fitness_pars:
            nsym = 5
        else:
            nsym = fitness_pars['nsym']

        kymo = np.mean(self.get_kymo_noisy(times, fitness_pars['ncells'], fitness_pars['t0_shift'],
                                           fitness_pars['noise'], nsym, ndt), axis=0)
        high = (kymo[:, -1] - fitness_pars['high_value'] > 0.).astype(int)
        low = (kymo[:, -1] - fitness_pars['low_value'] < 0.).astype(int)
        cross_high = np.where(high[1:-1] * (high[2:] - high[:-2]) != 0)[0] + 1
        cross_low = np.where(low[1:-1] * (low[2:] - low[:-2]) != 0)[0] + 1
        all_ind = np.concatenate((cross_high, cross_low))
        all_cross = np.concatenate((np.ones(len(cross_high), dtype=int), np.zeros(len(cross_low), dtype=int)))[
            np.argsort(all_ind)]
        n_boundaries = np.sum(np.diff(all_cross) != 0)
        penalty = fitness_pars['eps'] * np.sum((kymo[:, -fitness_pars['t_stable']:] -
                                                np.tile(np.array([kymo[:, -1]]).T, (1, fitness_pars['t_stable']))) ** 2)
        fitness = n_boundaries - penalty
        return fitness


# ______________________________________________________________________________________________________
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


# old function - plotting trajectories with colored segments
#         def plot_trajectories(self, n, times, L, noise, init_cond=None, ndt=10, color_scheme='state', slow=None):
#             if isinstance(init_cond, int):
#                 # print('int')
#                 module0 = self.module_list[init_cond]
#                 init_cond = np.array((module0.x, module0.y))
#             elif init_cond is None:
#                 init_cond = self.init_cond
#             if color_scheme == 'state':
#                 cmap = ListedColormap(['indianred', 'tab:orange', 'gold', 'tab:green', 'tab:blue', 'tab:purple', 'm'])
#                 norm = BoundaryNorm(np.arange(8) - 0.5, cmap.N)
#             elif color_scheme == 'time':
#                 cmap = 'viridis'
#                 norm = Normalize()
#             plt.figure(figsize=(6, 6))
#             for i in range(n):
#                 traj = self.get_trajectory_noisy(times, noise=noise, ndt=ndt, init_cond=init_cond, slow=slow)
#                 if color_scheme == 'state':
#                     states = np.zeros(len(times)).astype('int')
#                     for it in range(len(times)):
#                         states[it] = self.get_cell_state(traj[:, it])
#                 points = traj.T.reshape(-1, 1, 2)
#                 segments = np.concatenate([points[:-1], points[1:]], axis=1)
#
#                 lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.4)
#                 if color_scheme == 'state':
#                     lc.set_array(states)
#                 elif color_scheme == 'time':
#                     lc.set_array(np.arange(len(times)))
#                 lc.set_linewidth(1.5)
#                 plt.gca().add_collection(lc)
#                 # plt.plot(traj[0], traj[1], lw=1, c='w')
#             plt.xlim([-L, L])
#             plt.ylim([-L, L])
#             plt.show()

# _____________________________________________________________________________________________________________________


class CellDifProlifLandscape(Landscape):

    def get_fitness(self, t0, tf, nt, fitness_pars):
        times = np.linspace(t0, tf, nt)
        cell_states = np.zeros(fitness_pars['cell_data'].shape)  ## shape = (timepoints, states)
        self.init_cells(fitness_pars['ncells'], fitness_pars['init_state'], 0.1)  # _________________ fixed init noise
        prolif_pars = fitness_pars['prolif_pars']
        remove_pars = fitness_pars['remove_pars']
        traj, states, added_coord, removed_coord = self.run_cells_prolif(t0, tf, nt, prolif_pars, remove_pars,
                                                                         fitness_pars['noise'], ndt=100)
        # calculate cell_states matrix using states of each cell
        for state in range(cell_states.shape[1]):
            cell_states[:, state] = np.nansum(states == state, axis=0)
        # ??
        # cell_states = cell_states / np.nansum(states, axis=0)[:, np.newaxis] * 100.  # normalized to give percent at a given timepoint
        self.result = cell_states

        # implement penalty

        # penalty = 0.
        # for timepoint in range(len(times)):
        #     for state in range(cell_states.shape[1]):
        #         if fitness_pars['cell_data'][timepoint, state] != 0:
        #             # and
        #             if state in fitness_pars['attractor_states']:
        #                 # attractor: a deterministic trajectory starting from a state has to end up in the same state
        #                 self.init_cells(1, (self.module_list[state].x, self.module_list[state].y))
        #                 traj, state = self.run_cells(0, 50, 51, t_freeze=times[timepoint])
        #                 state_traj_final_state = ...
        #                 penalty += float(state_traj_final_state != state)
        #             if state in fitness_pars['non_attractor_states']:
        #                 # non-attractor: a deterministic trajectory starting from a state has to exit it
        #                 state_traj = self.get_trajectory_frozen(times[timepoint], np.linspace(0., 15., 16),
        #                                                         init_cond=(self.module_list[state].x,
        #                                                                    self.module_list[state].y))
        #                 state_traj_final_state = self.get_cell_state((state_traj[:, -1]))
        #                 penalty += float(state_traj_final_state == state)

        # penalty *= fitness_pars['penalty_weight']  # penalty weight was 10 by default

        # Calculate weights for each cell state to account for the total number of cells observed in given state:
        ncells_state = np.sum(fitness_pars['cell_data'], axis=0)
        column_weights = 1 / ncells_state
        column_weights[np.isnan(column_weights)] = 0.
        column_weights /= np.sum(column_weights)  # normalize: sum of weights over states/columns = nstates
        column_weights = np.tile(column_weights, (fitness_pars['cell_data'].shape[0], 1))  # repeat for all t

        fitness = -np.sum(column_weights * np.abs(cell_states - fitness_pars['cell_data'])) / 2. / len(times) * \
                  fitness_pars['cell_data'].shape[1]  # - penalty
        return fitness


# _____________________________________________________________________________________________________________________

class DecisionGraph_Landscape(Landscape):
    # fitness_pars: cell_data (final timepoint only), ncells, noise, init_state
    # times: (0, t_final)

    def get_cell_state(self, coordinate):
        dist = np.zeros(len(self.module_list))
        for i, module in enumerate(self.module_list):
            dist[i] = np.linalg.norm(coordinate - np.array((module.x, module.y)))
        return np.argmin(dist)

    def get_avg_coordinates(self, ncells, times, noise, init_state=0, avg_over=1):  # ndt for integration? default=10
        avg_coordinates = np.zeros((ncells, 2))
        for icell in range(ncells):
            traj = self.get_trajectory_noisy(times,
                                             init_cond=(self.module_list[init_state].x, self.module_list[init_state].y),
                                             noise=noise)
            avg_coordinates[icell] = np.mean(traj[:, -avg_over:], axis=1)
        return avg_coordinates

    def get_fitness(self, times, fitness_pars):
        cell_states = np.zeros(fitness_pars['cell_data'].shape)  ## shape = (timepoints, states)
        # ndt = (times[1]-times[0])/0.01

        # initial condition ! uniform across the domain
        init_cond = fitness_pars['init_cond']  ##################################

        for icell in range(fitness_pars['ncells']):
            cell_traj = self.get_trajectory_noisy(times, init_cond=init_cond[icell],
                                                  noise=fitness_pars['noise'], ndt=50)  ################## ndt !

            final_state = self.get_cell_state(cell_traj[:, -1])
            cell_states[final_state] += 1
        cell_states = cell_states / fitness_pars['ncells'] * 100.  # normalized to give percent
        self.result = cell_states
        fitness = -np.sum(np.abs(cell_states - fitness_pars['cell_data'])) / 2. / len(times)
        return fitness
# ______________________________________________________________________________________________________________________


# ____________________________________________________________________________________________________________________
# def get_landscape_fitness(landscape, times, ncells, t0_shift, fitness_pars):
#     fitness = landscape.get_fitness(times, ncells, t0_shift, fitness_pars)
#     #     landscape.fitness = fitness
#     #     return landscape
#     return fitness
# ______________________________
# def mutate_and_return_noisy(self, par_limits, prob_pars, fitness_pars, times, noise, n_sym, ndt):
#     r = np.random.uniform()
#     if r < prob_pars['prob_add'] or len(self.module_list) == 0:
#         self.add_module(Module.generate(self.used_fp_types, par_limits))
#     elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] and len(self.module_list) > 1 \
#             or len(self.module_list) > 10:
#         del_ind = np.random.choice(len(self.module_list))
#         self.del_module(del_ind)
#     elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] + prob_pars['prob_shuffle'] and len(self.module_list) > 1:
#         random.shuffle(self.module_list)
#     else:
#         mod_ind = np.random.choice(len(self.module_list))
#         self.module_list[mod_ind].mutate(par_limits)
#     self.fitness = self.get_fitness_noisy(times, fitness_pars, noise, n_sym, ndt)
#     return self

# _____________________________________

# if module.fp_type == 'node':
#     J = ((-1., 0.), (0., -1.))
# if module.fp_type == 'unstable node':
#     J = ((+1., 0.), (0., +1.))
# if module.fp_type == 'unstable spiral':
#     J = ((+1., -module.omega), (+module.omega, +1.))
# if module.fp_type == 'spiral':
#     J = ((-1., -module.omega), (+module.omega, -1.))
# if module.fp_type == 'center':
#     J = ((0., -module.omega), (+module.omega, 0.))
# if module.fp_type == 'saddle':
#     P = np.array(((np.cos(module.rotation), -np.sin(module.rotation)),
#                   (np.sin(module.rotation), np.cos(module.rotation))))
#     J0 = np.array(((+1., 0.), (0., -1.)))
#     J = P @ J0 @ P.T
