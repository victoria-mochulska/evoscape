import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from cmcrameri import cm as scm
from matplotlib.colors import ListedColormap, BoundaryNorm, CenteredNorm, Normalize
# import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import imageio.v2 as imageio

from .morphogen_regimes import mr_current_regime

plt.rcParams.update({'figure.dpi': 200})  # Change to 200 for high res figures, 100 for normal

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
    'tab:purple',
    # 'm',
)

cmap_state = ListedColormap(order_colors)
norm_state = BoundaryNorm(np.arange(len(order_colors) + 1) - 0.5, cmap_state.N)
cmap_time = 'viridis'


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
            sig = float(module.s)
            A = float(module.a)
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


def visualize_landscape_t(landscape, xx, yy, t, color_scheme='fp_types', circles=True, nullclines=True, density=0.5,
                          traj_times=None, traj_init_cond=(0., 0.), traj_start=0, traj_color='forestgreen', circle_opacity=0.25):
    """ Visualize the flow and modules at time t, with optional integrated trajectory in the frozen landscape. """

    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    circle_patches = []
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

        if circles:
            circle_patches.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                  fill=True, alpha=circle_opacity * np.sqrt(A), clip_on=True, linewidth=0))
    (dX, dY), potential, rot_potential = landscape(t, (xx, yy), return_potentials=True)

    fig, stream_ax = plt.subplots(1, 1, figsize=(5, 5))
    circles_ax = stream_ax

    if circles:
        for i in range(len(landscape.module_list)):
            circles_ax.add_patch(copy(circle_patches[i]))
            circles_ax.set_xlim((np.min(xx), np.max(xx)))
            circles_ax.set_ylim((np.min(yy), np.max(yy)))

    stream_ax.streamplot(xx, yy, dX, dY, density=density, arrowsize=2., arrowstyle='->', linewidth=1,
                         color='grey')
    if nullclines:
        stream_ax.contour(xx, yy, dX, (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
        stream_ax.contour(xx, yy, dY, (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

    if traj_times is not None:
        landscape.init_cells(1, traj_init_cond, noise=0.)
        traj, states = landscape.run_cells(traj_times[0], traj_times[1], traj_times[2], noise=0., ndt=50, frozen=True,
                                           t_freeze=t)
        stream_ax.plot(traj[0, 0, traj_start:], traj[1, 0, traj_start:], lw=3, color=traj_color)

        mid = traj_times[2] // 3 * 2
        x_coords = traj[0, 0, :]
        y_coords = traj[1, 0, :]
        arrow_size=0.3
        base = np.array([x_coords[mid], y_coords[mid]])
        direction = np.array([x_coords[mid] - x_coords[mid - 1], y_coords[mid] - y_coords[mid - 1], 0.])
        direction /= np.linalg.norm(direction)
        perp_vector = np.cross(direction, np.array([0, 0, 1]))[0:2]
        perp_vector /= np.linalg.norm(perp_vector)  # Normalize
        direction = direction[0:2]
        left = base + arrow_size * (perp_vector * 0.4 - direction)
        right = base + arrow_size * (-perp_vector * 0.4 - direction)
        stream_ax.plot(*zip(left, base, right), color=traj_color, linewidth=3, zorder=100)


    stream_ax.set_xlim([np.min(xx), np.max(xx)])
    stream_ax.set_ylim([np.min(yy), np.max(yy)])
    stream_ax.set_xticks([])
    stream_ax.set_yticks([])
    # landscape.morphogen_times = morphogen_times
    # plt.show()
    return fig, stream_ax


def visualize_cell_states(landscape, xx, yy, t, abs_threshold=0.):
    cell_states = landscape.get_cell_states(t, np.array((xx.flatten(), yy.flatten())), abs_threshold=abs_threshold)
    cmap_state = ListedColormap(['grey', ] + list(order_colors))
    norm_state = BoundaryNorm(np.arange(len(order_colors) + 2) - 1.5, cmap_state.N)
    fig, ax = visualize_landscape_t(landscape, xx, yy, t=t, color_scheme='order')
    # plt.figure()
    plt.imshow(np.reshape(cell_states, xx.shape), cmap=cmap_state, norm=norm_state, origin='lower',
               extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)), alpha=0.3, interpolation='nearest')
    return fig, ax

# _____________________________________________________________________________________________________________________


def visualize_potential(landscape, xx, yy, regime=None, t=None, color_scheme='fp_types', elev=None, azim=None, offset=2,
                        cmap_center=None, rot=False, rot_contour=False, min_contour_segment=80, scatter=False, zlim=None, axes=True):
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    # circles = []
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(6, 6))
    ax.view_init(elev=elev, azim=azim)

    if t is None and regime is not None:
        morphogen_times = landscape.morphogen_times
        landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
        t = float(regime)
    (dX, dY), potential, rot_potential = landscape(t, (xx, yy), return_potentials=True)
    if cmap_center is None:
        cmap_center = potential[0, 0]
    if rot:
        potential = rot_potential
        cmap = 'RdBu_r'
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
    if rot_contour:
        contour = plt.contour(xx, yy, rot_potential, levels=7, alpha=0)
        cmap_contour = plt.get_cmap('RdBu_r')
        norm = CenteredNorm(0., halfrange=np.max(np.abs(rot_potential)))
        for i, level_segments in enumerate(contour.allsegs[::-1]):
            level_value = contour.levels[-i]
            line_color = cmap_contour(norm(level_value))
            for segment in level_segments:
                if len(segment) < min_contour_segment:
                    continue  # Skip small segments
                x_coords = segment[:, 0]
                y_coords = segment[:, 1]
                derivs, z_coords, rot_z = landscape(t, (x_coords, y_coords), return_potentials=True)
                ax.plot(x_coords, y_coords, z_coords, color=line_color, linestyle='-', linewidth=2, zorder=100)

                arrow_size = 0.3

                if len(segment) > 80:
                    for mid in (len(x_coords) // 3, len(x_coords)//3*2):
                        base = np.array([x_coords[mid], y_coords[mid], z_coords[mid]])
                        direction = np.array([x_coords[mid] - x_coords[mid-1], y_coords[mid] - y_coords[mid-1],
                                              z_coords[mid] - z_coords[mid-1]])
                        direction /= np.linalg.norm(direction)
                        # if level_value < 0:
                        #     direction = -direction
                        perp_vector = np.cross(direction, np.array([0, 0, 1]))
                        perp_vector /= np.linalg.norm(perp_vector)  # Normalize
                        left = base + arrow_size * (perp_vector * 0.4 - direction)
                        right = base + arrow_size * (-perp_vector * 0.4 - direction)
                        ax.plot(*zip(left, base, right), color=line_color, linewidth=1.5, zorder=100)

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

    if regime is not None:
        landscape.morphogen_times = morphogen_times

    ax.set_xticks([])
    ax.set_yticks([])
    ax.zaxis.set_tick_params(color='white')
    ax.set_zticklabels([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # plt.tight_layout()
    if not axes:
        ax.set_axis_off()
    # plt.show()
    return fig


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

            stream_ax.imshow(velocities, alpha=0.5, cmap='Greys', origin='lower',
                             extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))

            # An attempt to plot fixed points - often ends up missing some points - high resolution needed
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
            stream_ax.plot(traj[0, 0, traj_start:], traj[1, 0, traj_start:], lw=3, color='forestgreen')



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


def get_and_plot_traj(landscape, t0, tf, nt, L, noise, ndt=50, s=6, frozen=False, t_freeze=None,
                      state_names=None, t_ticks=None, t_names=None):
    """ Integrate trajectories for cells and visualize in 2 panels: colored by time and by cell state """
    traj, states = landscape.run_cells(t0, tf, nt, noise=noise, ndt=ndt, frozen=frozen, t_freeze=t_freeze)

    fig = plt.figure(figsize=(9, 5))
    gs = GridSpec(2, 2, height_ratios=[20, 1], hspace=0.1, wspace=0.05)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_cbar, ax_state_cbar = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    time_values = np.tile(np.arange(nt), (states.shape[0], 1))
    ax0.scatter(traj[0, :, :], traj[1, :, :], s=s, alpha=0.2, c=time_values, cmap=cmap_time, edgecolor=None)
    sc0 = ax0.scatter(2*L*np.ones(nt), 2*L*np.ones(nt), c=np.linspace(t0, tf, nt), cmap=cmap_time, alpha=0.7, s=0.01, edgecolors='none')
    tbar = fig.colorbar(sc0, cax=ax_cbar, orientation='horizontal', label='Time')
    if t_ticks is not None:
        tbar.set_ticks(np.linspace(*t_ticks))
    else:
        tbar.set_ticks(np.linspace(t0, tf, nt))
    tbar.ax.tick_params(which='minor', length=0)
    if t_names is not None:
        tbar.set_ticklabels(t_names, fontsize=10)

    ax1.scatter(traj[0, :, :], traj[1, :, :], s=s, alpha=0.2, c=states, cmap=cmap_state, norm=norm_state, edgecolor=None)
    state_ticks = np.arange(len(order_colors))
    sc1 = ax1.scatter([2 * L] * len(order_colors), [2 * L] * len(order_colors), c=state_ticks, cmap=cmap_state, norm=norm_state, alpha=0.5, s=0.01)
    cbar = fig.colorbar(sc1, cax=ax_state_cbar, orientation='horizontal', label='Cell state')

    cbar.set_ticks(state_ticks)
    cbar.ax.tick_params(which='both', length=0)
    if state_names is not None:
        cbar.set_ticklabels(state_names)

    for ax in [ax0, ax1]:
        ax.set(xlim=[-L, L], ylim=[-L, L], aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def circle_plot(landscape, regime=None, L=6, color_scheme='order', lw=4):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    if regime is not None:
        regimes = (regime,)
    else:
        regimes = range(len(landscape.module_list[0].a))
    for i in range(len(landscape.module_list)):
        m = landscape.module_list[i]
        if color_scheme == 'fp_types':
            color = fp_type_colors[m.__class__.__name__]
        elif color_scheme == 'order':
            color = order_colors[i]
        else:
            color = 'grey'
        for j in regimes:
            circle = plt.Circle((m.x, m.y), 1.18 * m.s[j], color=color, fill=True,
                                alpha=0.25 * np.sqrt(m.a[j]), clip_on=False, linewidth=0,
                                linestyle='solid', zorder=i * 10)
            ax.add_patch(circle)
    ax.set_xlim((-L, L))
    ax.set_ylim((-L, L))
    ax.axis('off')
    ax.set_aspect('equal')
    return fig


def make_movie_discrete(landscape, xx, yy, labels, time_pars, n_cells, noise, init_cond=0,
                        circles=True, circle_opacity=0.1, density=0.65, nullclines=False,
                        fps=10, save_dir='', filename='movie.gif'):
    """
    Generate a trajectory movie with background streamplots that change between regimes
    """
    n_frames = time_pars[2]
    streamplots = [] # generate streamplots only once per condition
    for i in range(len(labels)):
        t_stream = landscape.morphogen_times[i-1] if i > 0 else time_pars[0]
        fig, ax = visualize_landscape_t(landscape, xx, yy, t_stream, color_scheme='order', circles=circles,
                                            nullclines=nullclines, circle_opacity=circle_opacity, density=density)
        ax.text(0.02, 0.95, labels[i], transform=ax.transAxes, fontsize=15, fontweight='bold')
        streamplots.append((fig, ax))

    landscape.init_cells(n_cells, init_cond, noise)
    times = np.linspace(*time_pars)
    traj, states = landscape.run_cells(*time_pars, noise, ndt=10, frozen=False)
    for i in range(n_frames):
        regime = mr_current_regime(times[i], *landscape.morphogen_times)
        fig, ax = streamplots[regime]
        sc = ax.scatter(traj[0, :, i], traj[1, :, i], s=25, alpha=1., c=states[:, i], cmap=cmap_state, norm=norm_state, zorder=10)
        fig.savefig(save_dir+f"frame_{i:03d}.png", dpi=150, bbox_inches='tight')
        sc.remove()

    for fig, ax in streamplots:
        plt.close(fig)

    frames = [imageio.imread(save_dir+f"frame_{i:03d}.png") for i in range(n_frames)]
    imageio.mimsave(save_dir+filename, frames, fps=fps)
    del frames
    print(f"Movie saved to {save_dir+filename}")

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

