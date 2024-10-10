import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from cmcrameri import cm as scm
from matplotlib.colors import ListedColormap, BoundaryNorm, CenteredNorm
from skimage.measure import label

plt.rcParams.update({'figure.dpi': 100})  # Change to 200 for high res figures

# to use for modules if colored by type
fp_type_colors = {
    'Node': 'tab:green',
    'UnstableNode': 'tab:blue',
    'Center': 'tab:purple',
    'NegCenter': 'hotpink',
}

# to use for modules if colored by order in the module_list
order_colors = (
    'indianred',
    'tab:orange',
    'gold',
    'tab:green',
    'tab:blue',
    # 'grey',
    'tab:purple',
    'm',
    # 'grey'
)


def visualize_landscape(landscape, xx, yy, regime, color_scheme='fp_types'):
    """ Simple visualization of landscape flow and modules in one regime. """
    density = 0.5
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    circles = []
    for i, module in enumerate(landscape.module_list):
        if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
            curl[i] = 1

    for i, module in enumerate(landscape.module_list):
        if module.a.size == 1 and module.s.size == 1 and regime == 0:
            sig = module.s
            A = module.a
        else:
            sig = module.s[regime]
            A = module.a[regime]
        if color_scheme == 'fp_types':
            color = fp_type_colors[module.__class__.__name__]
        elif color_scheme == 'order':
            color = order_colors[i]
        else:
            color = 'grey'

        circles.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                  fill=True, alpha=0.25 * np.sqrt(A), clip_on=True, linewidth=0))
    morphogen_times = landscape.morphogen_times
    landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
    (dX, dY), potential, rot_potential = landscape(float(regime), (xx, yy), return_potentials=True)

    fig, stream_ax = plt.subplots(1, 1, figsize=(5, 5))
    circles_ax = stream_ax

    for i in range(len(landscape.module_list)):
        circles_ax.add_patch(copy(circles[i]))
        circles_ax.set_xlim((np.min(xx), np.max(xx)))
        circles_ax.set_ylim((np.min(yy), np.max(yy)))

    stream_ax.streamplot(xx, yy, dX, dY, density=density, arrowsize=2., arrowstyle='->', linewidth=1,
                         color='grey')
    stream_ax.contour(xx, yy, dX, (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
    stream_ax.contour(xx, yy, dY, (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

    stream_ax.set_xlim([np.min(xx), np.max(xx)])
    stream_ax.set_ylim([np.min(yy), np.max(yy)])
    stream_ax.set_xticks([])
    stream_ax.set_yticks([])
    landscape.morphogen_times = morphogen_times
    # plt.show()
    return fig


# _______________________________________________________________________________________________________


def visualize_landscape_t(landscape, xx, yy, t, color_scheme='fp_types', traj_times=None, traj_init_cond=(0., 0.),
                          traj_start=0):
    """ Visualize the flow and modules at time t, with optional integrated trajectory in the frozen landscape. """
    density = 0.5
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    circles = []
    for i, module in enumerate(landscape.module_list):
        if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
            curl[i] = 1

    for i, module in enumerate(landscape.module_list):
        V, sig, A = module.get_current_pars(t, landscape.regime, *landscape.morphogen_times)
        if color_scheme == 'fp_types':
            color = fp_type_colors[module.__class__.__name__]
        elif color_scheme == 'order':
            color = order_colors[i]
        else:
            color = 'grey'

        circles.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                  fill=True, alpha=0.25 * np.sqrt(A), clip_on=True, linewidth=0))
    (dX, dY), potential, rot_potential = landscape(t, (xx, yy), return_potentials=True)

    fig, stream_ax = plt.subplots(1, 1, figsize=(4, 4))
    circles_ax = stream_ax

    for i in range(len(landscape.module_list)):
        circles_ax.add_patch(copy(circles[i]))
        circles_ax.set_xlim((np.min(xx), np.max(xx)))
        circles_ax.set_ylim((np.min(yy), np.max(yy)))

    stream_ax.streamplot(xx, yy, dX, dY, density=density, arrowsize=2., arrowstyle='->', linewidth=1,
                         color='grey')
    stream_ax.contour(xx, yy, dX, (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
    stream_ax.contour(xx, yy, dY, (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

    if traj_times is not None:
        landscape.init_cells(1, traj_init_cond, noise=0.)
        traj, states = landscape.run_cells(traj_times[0], traj_times[1], traj_times[2], noise=0., ndt=50, frozen=True,
                                           t_freeze=t)
        stream_ax.plot(traj[0, 0, traj_start:], traj[1, 0, traj_start:], lw=2.5, color='forestgreen')

    stream_ax.set_xlim([np.min(xx), np.max(xx)])
    stream_ax.set_ylim([np.min(yy), np.max(yy)])
    stream_ax.set_xticks([])
    stream_ax.set_yticks([])
    # landscape.morphogen_times = morphogen_times
    # plt.show()
    return fig, stream_ax

# _____________________________________________________________________________________________________________________


def visualize_potential(landscape, xx, yy, regime, color_scheme='fp_types', elev=None, azim=None, offset=2,
                        cmap_center=None, rot=False, scatter=False, zlim=None):
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    # circles = []
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(6, 6))
    ax.view_init(elev=elev, azim=azim)

    morphogen_times = landscape.morphogen_times
    landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
    (dX, dY), potential, rot_potential = landscape(float(regime), (xx, yy), return_potentials=True)
    if cmap_center is None:
        cmap_center = potential[0, 0]
    if rot:
        potential = rot_potential
        cmap = 'RdBu'
    else:
        cmap = scm.cork.reversed()

    if zlim is None:
        ax.set_zlim([np.min(potential) - offset, np.max(potential) + 2])
        zlow = np.min(potential) - offset
    else:
        ax.set_zlim(zlim)
        zlow = zlim[0]
    ax.contour(xx, yy, potential, zdir='z', offset=zlow, cmap=cmap, norm=CenteredNorm(cmap_center))
    ax.plot_surface(xx, yy, potential, cmap=cmap, linewidth=0, antialiased=False, norm=CenteredNorm(cmap_center))
    # if wind:
    #     right = rot_potential.copy()
    #     left = rot_potential.copy()
    #     right[rot_potential < 0] = 0
    #     left[rot_potential > 0] = 0
    #     ax.contour(xx, yy, right, zdir='z', offset=np.max(potential), cmap='RdBu', norm=CenteredNorm(0), zorder=10)
    #     ax.contour(xx, yy, np.abs(left), zdir='z', offset=np.max(potential), cmap='RdBu_r', norm=CenteredNorm(0),
    #                zorder=10)

    if scatter:
        for i, module in enumerate(landscape.module_list):
            if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
                curl[i] = 1
            if color_scheme == 'fp_types':
                color = fp_type_colors[module.__class__.__name__]
            elif color_scheme == 'order':
                color = order_colors[i]
            else:
                color = 'grey'
            ax.scatter(module.x, module.y, zlow, s=25, color=color, marker='D', zorder=20)

    landscape.morphogen_times = morphogen_times

    ax.set_xticks([])
    ax.set_yticks([])
    ax.zaxis.set_tick_params(color='white')
    ax.set_zticklabels([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # plt.tight_layout()
    # plt.show()
    return fig


# TODO: main visualizing function with 4 panels
def visualize_all(landscape, xx, yy, times, density=0.5, color_scheme='fp_types',
                  plot_velocities=True, plot_nullclines=True,
                  plot_traj=True, traj_times=(0., 100., 150), traj_start=50, traj_init_cond=(0., 1.), traj_noise=0., ):

    """
    Plot 4 panels: potential contour plot, rotational potential contour plot, flow plot with module circles,
    and flow plot with velocity magnitude
    :param landscape:
    :param xx:
    :param yy:
    :param times:
    :param density:
    :param color_scheme:
    :param plot_velocities:
    :param plot_nullclines:
    :param plot_traj:
    :param traj_times:
    :param traj_start:
    :param traj_init_cond:
    :param traj_noise:
    :return:
    """
    dX, dY = np.zeros((len(times), *xx.shape)), np.zeros((len(times), *xx.shape))
    
    figures = []

    for it in range(len(times)):

        (dX[it], dY[it]), potential, rot_potential = landscape(times[it], (xx, yy), return_potentials=True)

        circles = []
        for i, module in enumerate(landscape.module_list):
            V, sig, A = module.get_current_pars(times[it], landscape.regime, *landscape.morphogen_times)
            if color_scheme == 'fp_types':
                color = fp_type_colors[module.__class__.__name__]
            elif color_scheme == 'order':
                color = order_colors[i]
            else:
                color = 'grey'
            circles.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                      fill=True, alpha=0.25 * np.sqrt(A), clip_on=True, linewidth=0))

        vrange = (np.max(rot_potential) - np.min(rot_potential))/2.
        if vrange == 0.:
            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            circles_ax = ax[1]
            stream_ax = ax[2]
            # vrange = 1.
        else:
            fig, ax = plt.subplots(1, 4, figsize=(18, 4))
            circles_ax = ax[2]
            stream_ax = ax[3]
            ax[1].imshow(rot_potential, cmap='RdBu_r', origin='lower', norm=CenteredNorm(0, vrange),
                         extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
            ax[1].contour(xx, yy, rot_potential, colors='w', linestyles='solid', origin='lower')

        ax[0].imshow(potential, cmap=scm.cork.reversed(), origin='lower', norm=CenteredNorm(0),
                     extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
        ax[0].contour(xx, yy, -potential, origin='lower', colors='w')

        for iax in range(len(ax)):
            ax[iax].set_xticks([])
            ax[iax].set_yticks([])
            ax[iax].set_xlim((np.min(xx), np.max(xx)))
            ax[iax].set_ylim((np.min(yy), np.max(yy)))

        for i in range(len(landscape.module_list)):
            circles_ax.add_patch(copy(circles[i]))

        if plot_velocities:
            velocities_sq = dX[it] ** 2 + dY[it] ** 2
            velocities = np.sqrt(velocities_sq)
            # print('Min velocity:', round(np.min(velocities), 3), ', Max:', round(np.max(velocities), 3),
            #       ', Mean:', round(np.mean(velocities), 3), ', Median:', round(np.median(velocities), 3))

            stream_ax.imshow(velocities, alpha=0.3, cmap='Greys', origin='lower',
                             extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))

            # An attempt to plot fixed points - often ends up missing some points
            # fp_labels, nlabels = label(velocities_sq < 0.5, return_num=True)
            # for l in range(nlabels):
            #     if np.sum(fp_labels == l) <= 1000:
            #         fp = fp_labels == l
            #         stream_ax.imshow(fp, alpha=0.5, cmap='Blues', origin='lower',
            #                          extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
            # if np.sum(fp_labels == l) <= 50:
            # fp = velocities_sq == np.min(velocities_sq[fp_labels == l])
            # if np.sum(fp_labels == l) > 20:
            #     fp = (velocities_sq < 5e-4) * fp_labels == l
            # else:
            #     fp = fp_labels == l
            # stream_ax.scatter(xx[fp], yy[fp], marker='o', s=50, color='gold', edgecolor=None, zorder=10)

        circles_ax.streamplot(xx, yy, dX[it], dY[it], density=density, arrowsize=2., arrowstyle='->',
                              linewidth=1,
                              color='grey')
        stream_ax.streamplot(xx, yy, dX[it], dY[it], density=density, arrowsize=2., arrowstyle='->',
                             linewidth=1,
                             color='grey')

        if plot_nullclines:
            circles_ax.contour(xx, yy, dX[it], (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
            circles_ax.contour(xx, yy, dY[it], (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)
            stream_ax.contour(xx, yy, dX[it], (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
            stream_ax.contour(xx, yy, dY[it], (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

        if plot_traj:
            # calculate a trajectory in frozen landscape
            landscape.init_cells(1, traj_init_cond, noise=traj_noise)
            traj, states = landscape.run_cells(traj_times[0], traj_times[1], traj_times[2], noise=traj_noise,
                                               ndt=50, frozen=True, t_freeze=times[it])
            stream_ax.plot(traj[0, 0, traj_start:], traj[1, 0, traj_start:], lw=2.5, color='forestgreen')

        figures.append(fig)
        # plt.show()

    return figures


def plot_cells(landscape, L, colors=None):
    """ Plot the current cell locations and states """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    coord = landscape.cell_coordinates
    states = landscape.cell_states
    if colors is None:
        colors = order_colors
    cmap_state = ListedColormap(colors)
    norm_state = BoundaryNorm(np.arange(len(colors)+1) - 0.5, cmap_state.N)
    ax.scatter(coord[0], coord[1], s=8, alpha=0.3, c=states, cmap=cmap_state, norm=norm_state, edgecolors=None)
    ax.set_xlim([-L, L])
    ax.set_ylim([-L, L])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    return fig


def get_and_plot_traj(landscape, t0, tf, nt, L, noise, ndt=50, frozen=False, t_freeze=None, colors=None):
    """ Integrate trajectories for cells and visualize them in 2 panels:
    colored by timepoint and colored be cell state """
    if colors is None:
        colors = order_colors
    cmap_state = ListedColormap(colors)
    norm_state = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap_state.N)
    cmap_time = 'viridis'
    traj, states = landscape.run_cells(t0, tf, nt, noise=noise, ndt=ndt, frozen=frozen, t_freeze=t_freeze)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].scatter(traj[0, :, :], traj[1, :, :], s=6, alpha=0.2, c=np.tile(np.arange(nt), (states.shape[0], 1)),
                  cmap=cmap_time, edgecolor=None)
    ax[1].scatter(traj[0, :, :], traj[1, :, :], s=6, alpha=0.2, c=states, cmap=cmap_state, norm=norm_state,
                  edgecolors=None)
    ax[0].set_xlim([-L, L])
    ax[0].set_ylim([-L, L])
    ax[1].set_xlim([-L, L])
    ax[1].set_ylim([-L, L])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    return fig


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

# ______________________________________________________________________________________________________________________
    # if plot_weights:
    #     fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    #     # for iax in range(4):
    #         # ax[iax].axis('off')
    #     ax[0].imshow(potential, cmap='Greens', origin='lower', extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
    #     ax[0].contour(xx, yy, potential, origin='lower', colors='w')
    #
    #     ax[1].imshow(rot_potential, cmap='RdBu_r', origin='lower', norm=CenteredNorm(0), extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
    #     ax[1].contour(xx, yy, rot_potential, colors='w', linestyles='solid', origin='lower', levels=12)
    #
    #     for iax in range(2):
    #         ax[iax].set_xticks([])
    #         ax[iax].set_yticks([])
    #
    #     # can add nullclines but they make the plot busy
    #         for i, module in enumerate(self.module_list):
    #             ax[iax].scatter(module.x, module.y, marker='x', c='k')
    #     circles_ax = ax[2]
    #     stream_ax = ax[3]

    # if plot_velocities:
    #     # fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    #     velocities_sq = dX[it] ** 2 + dY[it] ** 2
    #     velocities = np.sqrt(velocities_sq)
    #
    #     # vel_plot = ax[0].imshow(velocities, cmap='BuPu', origin='lower', vmin=0, vmax=1.)
    #     # plt.colorbar(vel_plot)
    #     print('Min velocity:', round(np.min(velocities), 3), ', Max:', round(np.max(velocities), 3),
    #           ', Mean:', round(np.mean(velocities), 3), ', Median:', round(np.median(velocities), 3))
    #     # stream_ax = ax[1]

    # if plot_velocities:
    #     stream_ax.imshow(velocities, alpha=0.5, cmap='Greys', origin='lower', extent = (np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
    #     fp_labels, nlabels = label(velocities_sq < 1e-3, return_num=True)
    #     for l in range(nlabels):
    #         # if np.sum(fp_labels == l) <= 50:
    #         fp = velocities_sq == np.min(velocities_sq[fp_labels == l])
    #         # if np.sum(fp_labels == l) > 20:
    #         #     fp = (velocities_sq < 5e-4) * fp_labels == l
    #         # else:
    #         #     fp = fp_labels == l
    #         stream_ax.scatter(xx[fp], yy[fp], marker='o', s=50, color='gold', edgecolor=None, zorder=10)
    #     # stream_ax.(velocities_sq<1e-, cmap='viridis', origin='lower', extent = (np.min(xx), np.max(xx), np.min(yy), np.max(yy)), interpolation=None)

