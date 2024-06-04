from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as scm
from matplotlib.colors import CenteredNorm

plt.rcParams.update({'figure.dpi': 200})

fp_type_colors = {
    'Node': 'tab:green',
    'Spiral': 'gold',
    'UnstableSpiral': 'tab:orange',
    'UnstableNode': 'tab:blue',
    'DegenerateNode': 'm',
    'Center': 'tab:purple',
    'NegCenter': 'hotpink',
    'Saddle': 'indianred',
    'StableLine': 'grey',
}

order_colors = (
    'indianred',
    'tab:orange',
    'gold',
    'tab:green',
    'tab:blue',
    # 'k',
    'tab:purple',
    'm',
)


def visualize_landscape(landscape, xx, yy, regime, color_scheme='fp_types'):
    density = 0.5
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    circles = []
    for i, module in enumerate(landscape.module_list):
        if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
            curl[i] = 1

    for i, module in enumerate(landscape.module_list):
        # V, sig, A = module.get_current_pars(t, l.regime, *self.morphogen_times)
        sig = module.s[regime]
        A = module.a[regime]
        if color_scheme == 'fp_types':
            color = fp_type_colors[module.__class__.__name__]
        elif color_scheme == 'order':
            color = order_colors[i]
        else:
            color = 'grey'

        circles.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                  fill=True, alpha=0.3 * np.sqrt(A), clip_on=True, linewidth=0))
    morphogen_times = landscape.morphogen_times
    landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
    (dX, dY), potential, rot_potential = landscape(float(regime), (xx, yy), return_weights=True)

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
    # if plot_traj:
    #     stream_ax.plot(traj[0, 0, traj_start:], traj[1, 0, traj_start:], lw=2.5, color='forestgreen')

    stream_ax.set_xlim([np.min(xx), np.max(xx)])
    stream_ax.set_ylim([np.min(yy), np.max(yy)])
    stream_ax.set_xticks([])
    stream_ax.set_yticks([])
    landscape.morphogen_times = morphogen_times
    plt.show()
    return fig


# _______________________________________________________________________________________________________


def visualize_landscape_t(landscape, xx, yy, t, color_scheme='fp_types', traj_times=None, traj_init_cond=(0., 0.),
                          traj_start=0):
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
                                  fill=True, alpha=0.3 * np.sqrt(A), clip_on=True, linewidth=0))
    # morphogen_times = landscape.morphogen_times
    # landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
    (dX, dY), potential, rot_potential = landscape(t, (xx, yy), return_weights=True)

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
    # if plot_traj:
    #     stream_ax.plot(traj[0, 0, traj_start:], traj[1, 0, traj_start:], lw=2.5, color='forestgreen')

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


# _______________________________________________________________________________________________________


def visualize_potential(landscape, xx, yy, regime, color_scheme='fp_types', elev=None, azim=None, offset=2,
                        cmap_center=None, rot=False, scatter=False, zlim=None, show_axis=False, wind=False):
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    # circles = []
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(6, 6))
    ax.view_init(elev=elev, azim=azim)

    morphogen_times = landscape.morphogen_times
    landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
    (dX, dY), potential, rot_potential = landscape(float(regime), (xx, yy), return_weights=True)
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
    if wind:
        right = rot_potential.copy()
        left = rot_potential.copy()
        right[rot_potential < 0] = 0
        left[rot_potential > 0] = 0
        ax.contour(xx, yy, right, zdir='z', offset=np.max(potential), cmap='RdBu', norm=CenteredNorm(0), zorder=10)
        ax.contour(xx, yy, np.abs(left), zdir='z', offset=np.max(potential), cmap='RdBu_r', norm=CenteredNorm(0),
                   zorder=10)

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

    # scm.cork.reversed()

    landscape.morphogen_times = morphogen_times
    # if not show_axis:
    #     plt.axis('off')

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
