import numpy as np
import random
from copy import deepcopy

from .. import mr_sigmoid
from .landscape_phase_analysis import LandscapePhaseAnalysisBase
from evoscape.modules.module_class import Node, UnstableNode, Center, NegCenter

def _flow(q_flat, xs, ys, sign, curl, sig, a, Js, A0, x0, return_potentials):
    x, y = q_flat
    xr = x[None, :] - xs
    yr = y[None, :] - ys
    r = np.sqrt(xr ** 2 + yr ** 2)

    # w = np.zeros_like(r)
    # mask = ~((a == 0) & (sig == 0))
    nonzero_sig = np.where(sig == 0, 1, sig)
    w = a * np.exp(-0.5 * (r / nonzero_sig) ** 2)

    dx = Js[:, :, 0, 0] * xr + Js[:, :, 0, 1] * yr
    dy = Js[:, :, 1, 0] * xr + Js[:, :, 1, 1] * yr

    dX = A0 * (-(x - x0[0]) ** 3) + np.sum(w * dx, axis=0)
    dY = A0 * (-(y - x0[1]) ** 3) + np.sum(w * dy, axis=0)
    derivs = np.stack((dX, dY), axis=0)

    if return_potentials:
        coefs = sign * (1-curl) * sig ** 2
        coefs_rot = (sign * curl) * sig ** 2
        pot = np.sum(w * coefs, axis=0) + A0 / 4 * ((x - x0[0]) ** 4 + (y - x0[1]) ** 4)
        pot_rot = np.sum(w * coefs_rot, axis=0)
        return derivs, pot, pot_rot
    return derivs

class Landscape(LandscapePhaseAnalysisBase):
    def __init__(self, module_list=(), A0=0., init_cond=(0., 1.), regime=mr_sigmoid, n_regimes=2,
                 morphogen_times=(0.,), used_fp_types=(Node,), immutable_pars_list=(), x0=(0., 0.)):
        """
        :param module_list: list of module objects
        :param A0: float - strength of global attraction (boundary condition of the potential)
        :param init_cond: default initial condition for cells in this landscape
        :param regime: function from morphogen_regimes - dynamics of module amplitudes and sizes
        :param n_regimes: number of morphogen conditions
        :param morphogen_times: times of changes in morphogens (signals)
        :param used_fp_types: list of module types to add in random mutation (Node, UnstableNode, Center, NegCenter)
        :param immutable_pars_list: list of parameter names to fix for all initial modules of the landscape
        """
        self.module_list = []
        # modules are stored in module_list
        for ind in range(len(module_list)):
            module_copy = deepcopy(module_list[ind])
            setattr(module_copy, "at_init", True)
            self.module_list.append(module_copy)
            for par_name in immutable_pars_list:
                if par_name in self.module_list[ind].mutable_parameters_list:
                    self.module_list[ind].remove_mutable_parameter(par_name)
        self._normalize_module_list_order()
        self.A0 = A0
        self.x0 = x0
        self.regime = regime
        self.n_regimes = n_regimes
        self.morphogen_times = morphogen_times
        self.used_fp_types = used_fp_types
        self.init_cond = init_cond
        self.min_n_modules = len(module_list)
        self.max_n_modules = 15

        self.fitness = None  # stores calculated fitness
        self.result = None  # stores some ouput besides fitness

        self.cell_coordinates = None  # current coordinates of cells
        self.cell_states = None  # current cell state assignments
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

    @property
    def n_nodes(self):
        return sum(isinstance(module, Node) for module in self.module_list)

    @property
    def node_modules(self):
        return self.module_list[:self.n_nodes]

    def _normalize_module_list_order(self):
        node_modules = [module for module in self.module_list if isinstance(module, Node)]
        other_modules = [module for module in self.module_list if not isinstance(module, Node)]
        self.module_list = node_modules + other_modules

    def _shuffle_node_modules(self):
        node_modules = list(self.node_modules)
        random.shuffle(node_modules)
        self.module_list = node_modules + self.module_list[self.n_nodes:]

    def _deletable_module_indices(self):
        return [
            index
            for index, module in enumerate(self.module_list)
            if not getattr(module, "at_init", False)
        ]

    @staticmethod
    def _coerce_coordinate_array(coordinate):
        coordinate = np.asarray(coordinate, dtype=float)
        if coordinate.ndim == 1:
            if coordinate.shape != (2,):
                raise ValueError("coordinate must have shape (2,) or (2, n)")
            coordinate = coordinate.reshape(2, 1)
        if coordinate.ndim != 2 or coordinate.shape[0] != 2:
            raise ValueError("coordinate must have shape (2, n)")
        return coordinate

    def _node_state_distance_matrix(self, coordinate):
        coordinate = self._coerce_coordinate_array(coordinate)
        if self.n_nodes == 0:
            return np.empty((coordinate.shape[1], 0), dtype=float)
        node_coords = np.array([(module.x, module.y) for module in self.node_modules], dtype=float)
        deltas = coordinate.T[:, None, :] - node_coords[None, :, :]
        return np.linalg.norm(deltas, axis=2)

    def _map_attractors_to_nodes(self, attractors, basin_labels=None, x_coords=None, y_coords=None):
        if self.n_nodes == 0:
            return {}

        node_coords = np.array([(module.x, module.y) for module in self.node_modules], dtype=float)
        attractor_order = []
        attractor_anchors = {}
        for attractor in attractors:
            attractor_type = attractor.get("type")
            if attractor_type == "fixed_point":
                anchor = np.asarray(attractor["point"], dtype=float)
            elif attractor_type == "cycle":
                anchor = np.asarray(attractor["center"], dtype=float)
            else:
                continue
            attractor_id = int(attractor["id"])
            attractor_order.append(attractor_id)
            attractor_anchors[attractor_id] = anchor

        attractor_node_states = {}

        if basin_labels is not None and x_coords is not None and y_coords is not None and attractor_order:
            x_coords = np.asarray(x_coords, dtype=float)
            y_coords = np.asarray(y_coords, dtype=float)
            basin_labels = np.asarray(basin_labels, dtype=int)
            sampled_labels = np.full(self.n_nodes, -1, dtype=int)

            for node_state, (x_node, y_node) in enumerate(node_coords):
                if (
                    x_node < x_coords[0]
                    or x_node > x_coords[-1]
                    or y_node < y_coords[0]
                    or y_node > y_coords[-1]
                ):
                    continue
                if x_coords.size > 1:
                    ix = int(np.rint((x_node - x_coords[0]) / (x_coords[1] - x_coords[0])))
                else:
                    ix = 0
                if y_coords.size > 1:
                    iy = int(np.rint((y_node - y_coords[0]) / (y_coords[1] - y_coords[0])))
                else:
                    iy = 0
                if 0 <= ix < basin_labels.shape[1] and 0 <= iy < basin_labels.shape[0]:
                    sampled_labels[node_state] = int(basin_labels[iy, ix])

            for attractor_id in attractor_order:
                owner_nodes = np.flatnonzero(sampled_labels == attractor_id)
                if owner_nodes.size == 0:
                    continue
                if owner_nodes.size == 1:
                    node_state = int(owner_nodes[0])
                else:
                    anchor = attractor_anchors[attractor_id]
                    distances = np.linalg.norm(node_coords[owner_nodes] - anchor[None, :], axis=1)
                    node_state = int(owner_nodes[int(np.argmin(distances))])
                attractor_node_states[attractor_id] = node_state
            return attractor_node_states

        for attractor_id in attractor_order:
            anchor = attractor_anchors[attractor_id]
            distances = np.linalg.norm(node_coords - anchor[None, :], axis=1)
            attractor_node_states[attractor_id] = int(np.argmin(distances))

        return attractor_node_states

    def _auto_basin_ranges(self, coordinate):
        coordinate = self._coerce_coordinate_array(coordinate)
        if self.module_list:
            module_coords = np.array([(module.x, module.y) for module in self.module_list], dtype=float).T
            all_coords = np.concatenate((coordinate, module_coords), axis=1)
        else:
            all_coords = coordinate

        x_min = float(np.min(all_coords[0]))
        x_max = float(np.max(all_coords[0]))
        y_min = float(np.min(all_coords[1]))
        y_max = float(np.max(all_coords[1]))
        span = max(x_max - x_min, y_max - y_min, 1.0)
        pad = max(0.5, 0.15 * span)
        return (x_min - pad, x_max + pad), (y_min - pad, y_max + pad)

    def _build_basin_state_grid(self, t, coordinate, grid_points=201):
        coordinate = self._coerce_coordinate_array(coordinate)
        x_range, y_range = self._auto_basin_ranges(coordinate)
        x_coords = np.linspace(*x_range, int(grid_points))
        y_coords = np.linspace(*y_range, int(grid_points))
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='xy')

        fixed_points = self.find_fixed_points(t, x_range, y_range)
        saddle_manifolds = self.find_saddle_manifolds(
            t,
            fixed_points=fixed_points,
            x_range=x_range,
            y_range=y_range,
        )
        basin_result = self.find_attractor_basins_manifold(
            t,
            xx,
            yy,
            fixed_points=fixed_points,
            saddle_manifolds=saddle_manifolds,
        )

        labels = np.asarray(basin_result["labels"], dtype=int)
        attractor_node_states = self._map_attractors_to_nodes(
            basin_result["attractors"],
            basin_labels=labels,
            x_coords=x_coords,
            y_coords=y_coords,
        )
        basin_states = np.full(xx.shape, -1, dtype=int)
        for attractor_id, node_state in attractor_node_states.items():
            basin_states[labels == attractor_id] = node_state

        return x_coords, y_coords, basin_states

    @staticmethod
    def _sample_regular_grid(x_grid, y_grid, grid_values, coordinate):
        coordinate = Landscape._coerce_coordinate_array(coordinate)
        if grid_values.size == 0:
            return np.full(coordinate.shape[1], -1, dtype=int)

        if x_grid.size > 1:
            dx = float(x_grid[1] - x_grid[0])
            ix = np.rint((coordinate[0] - x_grid[0]) / dx).astype(int)
        else:
            ix = np.zeros(coordinate.shape[1], dtype=int)

        if y_grid.size > 1:
            dy = float(y_grid[1] - y_grid[0])
            iy = np.rint((coordinate[1] - y_grid[0]) / dy).astype(int)
        else:
            iy = np.zeros(coordinate.shape[1], dtype=int)

        ix = np.clip(ix, 0, len(x_grid) - 1)
        iy = np.clip(iy, 0, len(y_grid) - 1)
        return np.asarray(grid_values[iy, ix], dtype=int)

# ______________________________________________________________________________________________________________________
# ______________________________ Landscape dynamics calculation ________________________________________________________

    @staticmethod
    def local_weight(r, sig):
        """ Potential kernel (gaussian) """
        weight = np.exp(-0.5 * (r / sig) ** 2)
        return weight

    @staticmethod
    def fixed_point(module, x, y):
        J = module.J  # Jacobian of the module
        dx = J[0][0] * x + J[0][1] * y
        dy = J[1][0] * x + J[1][1] * y
        return dx, dy

    def __call__(self, t, q, return_potentials=False):
        """
        Evaluate the flow at coordinates q and time t
        :param t: float
        :param q: array of shape (2, m, n); q[0] are x-coordinates, q[1] are y-coordinates
        :param return_potentials: bool
        :return: tuple of arrays with x and y derivatives, potentials (optional)
        """
        q = np.asarray(q)
        grid_shape = q.shape[1:]
        n_pts = q[1:].size

        if not self.module_list:
            x, y = q
            zeros = np.zeros(grid_shape)
            derivs = np.stack([zeros, zeros], axis=0)
            pot = self.A0 / 4 * ((x - self.x0[0]) ** 4 + (y - self.x0[1]) ** 4)
            if return_potentials:
                return derivs, pot, zeros
            return derivs

        q_flat = q.reshape(2, -1)
        xs = np.array([m.x for m in self.module_list])[:, None]
        ys = np.array([m.y for m in self.module_list])[:, None]
        sign = np.array([-1 if isinstance(m, (Node, NegCenter)) else +1 for m in self.module_list])[:, None]
        curl = np.array([1 if isinstance(m, (Center, NegCenter)) else 0 for m in self.module_list])[:, None]

        Js = np.stack([m.J for m in self.module_list], axis=0)[:, None, :, :]

        pars = [m.get_current_pars(t, self.regime, *self.morphogen_times)[1:] for m in self.module_list]
        sig_list, a_list = zip(*pars)
        sig = np.stack([np.broadcast_to(np.asarray(s), (n_pts,)) for s in sig_list], axis=0)
        a = np.stack([np.broadcast_to(np.asarray(amp), (n_pts,)) for amp in a_list], axis=0)

        res = _flow(q_flat, xs, ys, sign, curl, sig, a, Js, self.A0, self.x0, return_potentials)

        if return_potentials:
            derivs, pot, pot_rot = res
            derivs = derivs.reshape((2,) + grid_shape)
            pot = pot.reshape(grid_shape)
            pot_rot = pot_rot.reshape(grid_shape)
            return derivs, pot, pot_rot
        derivs = res.reshape((2,) + grid_shape)
        return derivs

    # def __call__(self, t, q, return_potentials=False):
    #     """
    #     Evaluate the flow at coordinates q and time t
    #     :param t: float
    #     :param q: array of shape (2, m, n); q[0] are x-coordinates, q[1] are y-coordinates
    #     :param return_potentials: bool
    #     :return: tuple of arrays with x and y derivatives, potentials (optional)
    #     """
    #     x = q[0]
    #     y = q[1]
    #     w = np.zeros((len(self.module_list), *x.shape))
    #     sig = np.zeros((len(self.module_list)))
    #     sign = np.zeros((len(self.module_list)), dtype='int')   # Sign of modules (+1 or -1)
    #     curl = np.zeros((len(self.module_list)), dtype='bool')  # Is the module rotational (0 or 1)
    #     dx, dy = np.zeros((len(self.module_list), *x.shape)), np.zeros((len(self.module_list), *x.shape))
    #     for i, module in enumerate(self.module_list):
    #         V, sig[i], A = module.get_current_pars(t, self.regime, *self.morphogen_times)
    #         if module.__class__.__name__ == 'Node' or module.__class__.__name__ == 'NegCenter':
    #             sign[i] = -1
    #         else:
    #             sign[i] = +1
    #         if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
    #             curl[i] = 1
    #
    #         xr = x - module.x
    #         yr = y - module.y
    #         r = np.sqrt(xr ** 2 + yr ** 2)
    #         w[i, :] = A * self.local_weight(r, sig[i])
    #         dx[i, :], dy[i, :] = self.fixed_point(module, xr, yr)
    #     derivs = self.A0 * np.array((-(x-self.x0[0]) ** 3, -(y-self.x0[1]) ** 3)) + (np.sum(w * dx, axis=0), np.sum(w * dy, axis=0))
    #     if return_potentials:
    #         broadcast_shape = (len(self.module_list),) + (1,) * len(x.shape)
    #         coefs = (~curl * sign * sig ** 2).reshape(broadcast_shape)
    #         potential = np.sum(w * coefs, axis=0) + self.A0 / 4 * ((x-self.x0[0])**4 + (y - self.x0[1])**4)
    #         coefs_rot = (curl * sign * sig ** 2).reshape(broadcast_shape)
    #         rot_potential = np.sum(w * coefs_rot, axis=0)
    #         # potential = np.sum(w * (~curl * sign * sig ** 2)[:, np.newaxis, np.newaxis], axis=0) + self.A0 / 4 * (x ** 4 + y ** 4)
    #         # rot_potential = np.sum(w * (curl * sign * sig ** 2)[:, np.newaxis, np.newaxis], axis=0)
    #
    #         return derivs, potential, rot_potential
    #     return derivs

    # __________________________________________________________________________________________________________________
    # __________________________________ For evolutionary optimization _________________________________________________

    def get_fitness(self, fitness_pars):
        """ Compute the fitness function, using a dict of fitness parameters. Override in child classes. """
        raise NotImplementedError

    def add_module(self, M):
        """ Add a module to the landscape """
        module_copy = deepcopy(M)
        setattr(module_copy, "at_init", False)
        self.module_list.append(module_copy)
        self._normalize_module_list_order()

    def del_module(self, del_ind):
        """ Remove the module at index del_ind from the landscape """
        del self.module_list[del_ind]
        self._normalize_module_list_order()

    def mutate(self, par_limits, par_choice_values, prob_pars, fitness_pars):
        """
        In-place modification of the landscape.
        Randonly mutate the landscape according to prob_pars. If a parameter is mutated, new values are sampled
        according to par_limits and par_choice_values. Recalculate the fitness using fitness_pars.
        :param par_limits:
        :param par_choice_values:
        :param prob_pars:
        :param fitness_pars:
        """
        r = np.random.uniform()
        if r < prob_pars['prob_add'] or len(self.module_list) == 0:
            # print('Adding,', 'len =', len(self.module_list), ', r =', r)
            fp_type = random.choice(self.used_fp_types)
            self.add_module(fp_type.generate(par_limits, par_choice_values, n_regimes=self.n_regimes))
        elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] and len(self.module_list) > 1:
            # print('Deleting,', 'len =', len(self.module_list), ', r =', r)
            del_ind = np.random.choice(len(self.module_list))
            self.del_module(del_ind)
        # elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] + prob_pars['prob_shuffle'] and len(
        #         self.module_list) > 1:
        #     # print('Shuffling,', 'len =', len(self.module_list), ', r =', r)
        #     random.shuffle(self.module_list)
        else:
            # print('Modifying,', ', r =', r)
            mod_ind = np.random.choice(len(self.module_list))
            self.module_list[mod_ind].mutate(par_limits, par_choice_values)
        self.fitness = self.get_fitness(fitness_pars)

    def mutate_and_return(self, par_limits, par_choice_values, prob_pars, fitness_pars):
        """
        Mutates and also returns the landscape - required for parallel computation.
        Randonly mutate the landscape according to prob_pars. If a parameter is mutated, new values are sampled
        according to par_limits and par_choice_values. Recalculate the fitness using fitness_pars.
        :param par_limits:
        :param par_choice_values:
        :param prob_pars:
        :param fitness_pars:
        """
        r = np.random.uniform()
        if r < prob_pars['prob_add'] or len(self.module_list) == 0:
            fp_type = random.choice(self.used_fp_types)
            self.add_module(fp_type.generate(par_limits, par_choice_values, n_regimes=self.n_regimes))
        elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] and len(self.module_list) > self.min_n_modules \
                or len(self.module_list) > self.max_n_modules:
            deletable_indices = self._deletable_module_indices()
            if deletable_indices:
                del_ind = int(np.random.choice(deletable_indices))
                self.del_module(del_ind)
        elif r < prob_pars['prob_add'] + prob_pars['prob_drop'] + prob_pars['prob_shuffle'] and len(
                self.module_list) > 1:
            self._shuffle_node_modules()
        else:
            mod_ind = np.random.choice(len(self.module_list))
            self.module_list[mod_ind].mutate(par_limits, par_choice_values)
        self.fitness = self.get_fitness(fitness_pars)
        return self

# ______________________________________________________________________________________________________________________
# _____________________________________ Everything to do with cells ____________________________________________________

    def init_cells(self, n, init_cond, noise=0.):
        """
        Initialize cells in the landscape with a given initial condition.
        :param n: int, number of cells
        :param init_cond: int or array/tuple of length 2 or array of shape (2, n) or array of length self.n_nodes.
            Int: Node state number, all cells are initialized at the Node location.
            Tuple: (x,y) - same coordinate for all cells.
            Array (2, n) - x and y coordinates for n cells.
            Array (self.n_nodes) - number of cells starting at each Node, numbers must sum to n.
        :param noise: amplitude of gaussian noise added to each cell's initial coordinate.
        """
        if isinstance(init_cond, (int, np.integer)):
            if self.n_nodes == 0:
                raise ValueError("init_cells(int) requires at least one Node module.")
            if int(init_cond) < 0 or int(init_cond) >= self.n_nodes:
                raise ValueError(f"Node init_cond must be between 0 and {self.n_nodes - 1}.")
            module0 = self.module_list[int(init_cond)]
            init_cond = (module0.x, module0.y)
        elif init_cond is None:
            init_cond = self.init_cond

        init_cond = np.asarray(init_cond)
        if init_cond.shape == (2, n):
            self.cell_coordinates = init_cond.astype('float')
        elif init_cond.shape == (2,):
            self.cell_coordinates = np.tile(init_cond.astype('float'), (n, 1)).T
        elif init_cond.ndim == 1 and len(init_cond) == self.n_nodes and np.sum(init_cond) == n:
            if self.n_nodes == 0:
                raise ValueError("init_cells(counts) requires at least one Node module.")
            module_locs = np.array([(module.x, module.y) for module in self.node_modules])
            self.cell_coordinates = np.repeat(module_locs, init_cond, axis=0).T
        else:
            raise ValueError('Wrong shape of init_cond input')

        if noise != 0.:
            self.cell_coordinates += noise * np.random.randn(2, n)
        self.cell_states = self.get_cell_states_static()

    def reset_cells(self):
        """ Remove all stored cell coordinates and states """
        self.cell_coordinates = None
        self.cell_states = None

    @property
    def n(self):
        """ Number of cells currently in the landscape """
        return np.sum(~np.isnan(np.sum(self.cell_coordinates, axis=0)))

    def get_cell_states_static(self, coordinate=None):
        """
        Return cell states given cell coordinates.
        Assignment is based on proximity to modules and does not depend on time or signals.
        :param coordinate: array of shape (2, n) where n is the number of cells
            (optional, can use the current coordinates stored in landscape)
        :return: states - array of length n of ints
        """
        if coordinate is None:
            coordinate = self.cell_coordinates
        coordinate = self._coerce_coordinate_array(coordinate)
        if self.n_nodes == 0:
            return np.full(coordinate.shape[1], -1, dtype=int)
        dist = self._node_state_distance_matrix(coordinate)
        return np.argmin(dist, axis=1).astype(int)

    def get_cell_states(
        self,
        t,
        coordinate=None,
        measure='gaussian',
        prob_threshold=0.,
        abs_threshold=0.,
        t_freeze=None,
        basin_grid=None,
    ):
        """
        Return cell states given cell coordinates. Assignent based on a chosen distance measure, can depend on time or signals.
        :param t: float, timepoint
        :param coordinate: array of shape (2, n) where n is the number of cells
            (optional, can use the current coordinates stored in landscape)
        :param measure: 'dist' - based on Euclidean distance to Node modules.
            'gaussian' - based on a gaussian mixture model over Node modules, taking into account time-dependent module size.
            'basin' - based on the current basin of attraction, looked up on an auto-generated rasterized grid.
            'basin static' - based on a basin grid computed at t_freeze and sampled for the current coordinates.
        :return: states - array of length n of ints
        """
        if coordinate is None:
            coordinate = self.cell_coordinates
        coordinate = self._coerce_coordinate_array(coordinate)
        if self.n_nodes == 0:
            return np.full(coordinate.shape[1], -1, dtype=int)
        if measure not in ('dist', 'gaussian', 'basin', 'basin static', 'basin_static'):
            raise ValueError("measure must be one of 'dist', 'gaussian', 'basin', 'basin static', or 'basin_static'")
        states = None

        if measure == 'dist':
            dist = self._node_state_distance_matrix(coordinate)
            states = np.argmin(dist, axis=1)
        elif measure == 'gaussian':
            prob = np.zeros((coordinate.shape[1], self.n_nodes + 1), dtype=float)
            for i, module in enumerate(self.node_modules):
                V, st, at = module.get_current_pars(t, self.regime, *self.morphogen_times)
                del V
                if st == 0 or at == 0:
                    prob[:, i] = 0.
                else:
                    prob[:, i] = np.exp(
                        -np.sum((coordinate.T - np.array((module.x, module.y))) ** 2, axis=1) / 2. / st ** 2) / st ** 2
            if abs_threshold != 0:
                prob[:, -1] = abs_threshold
                row_sums = np.sum(prob, axis=1, keepdims=True)
                zero_rows = row_sums[:, 0] <= 0
                prob[zero_rows, -1] = 1.0
                row_sums[zero_rows] = 1.0
                prob = prob / row_sums
            else:
                row_sums = np.sum(prob[:, :-1], axis=1, keepdims=True)
                zero_rows = row_sums[:, 0] <= 0
                row_sums[zero_rows] = 1.0
                prob[:, :-1] = prob[:, :-1] / row_sums
                prob[:, -1] = prob_threshold
            states = np.argmax(prob, axis=1)
            states[states == self.n_nodes] = -1
            if abs_threshold == 0:
                states[zero_rows] = -1
        elif measure == 'basin':
            if basin_grid is None:
                basin_grid = self._build_basin_state_grid(t, coordinate)
            x_coords, y_coords, basin_states = basin_grid
            states = self._sample_regular_grid(
                np.asarray(x_coords, dtype=float),
                np.asarray(y_coords, dtype=float),
                np.asarray(basin_states, dtype=int),
                coordinate,
            )
        elif measure == 'basin static' or measure == 'basin_static':
            if basin_grid is None:
                if t_freeze is None:
                    raise ValueError("measure 'basin static' requires t_freeze or a precomputed basin_grid.")
                basin_grid = self._build_basin_state_grid(t_freeze, coordinate)
            x_coords, y_coords, basin_states = basin_grid
            states = self._sample_regular_grid(
                np.asarray(x_coords, dtype=float),
                np.asarray(y_coords, dtype=float),
                np.asarray(basin_states, dtype=int),
                coordinate,
            )
        else:
            raise ValueError("measure must be one of 'dist', 'gaussian', 'basin', 'basin static', or 'basin_static'")
        return states

    def run_cells(self, t0, tf, nt, noise=0., ndt=50, frozen=False, t_freeze=None, get_states=True):
        """
        Run trajectories for cells in the landscape.
        :param t0: float, start time
        :param tf: float, end time
        :param nt: int, number of timepoints
        :param noise: float, amplitude of gaussian noise
        :param ndt: int, number of integration steps per timepoint
        :param frozen: bool, whether to fix the landscape paremeters
        :param t_freeze: if frozen, provide the time at which to calculate the landscape, to be kept constant
        :param get_states: bool or str, whether to compute cell states, or which state measure to use
        :return: traj (array of shape (2, n, nt)) and states (int array of shape (2, nt))
        """
        traj = np.empty((*self.cell_coordinates.shape, nt), dtype='float')
        y = self.cell_coordinates
        traj[:, :, 0] = y
        t = t0
        Delta_t = (tf - t0) / (nt - 1)
        dt = Delta_t / ndt
        sqrt_dt = np.sqrt(dt)

        if get_states:
            state_measure = 'gaussian' if get_states is True else get_states
            if state_measure not in ('dist', 'gaussian', 'basin', 'basin static', 'basin_static'):
                raise ValueError(
                    "get_states must be True, False, or one of 'dist', 'gaussian', 'basin', 'basin static', 'basin_static'"
                )
            if (state_measure == 'basin static' or state_measure == 'basin_static') and t_freeze is None:
                raise ValueError("get_states='basin static' requires t_freeze to be provided.")
            state_basin_grid = (
                self._build_basin_state_grid(t_freeze, y)
                if state_measure == 'basin static' or state_measure == 'basin_static'
                else None
            )
            state_time = (
                t_freeze
                if state_measure == 'basin static' or state_measure == 'basin_static'
                else (t_freeze if frozen and t_freeze is not None else t)
            )
            states = np.empty((self.cell_coordinates.shape[1], nt), dtype='int')
            states[:, 0] = self.get_cell_states(
                state_time,
                coordinate=y,
                measure=state_measure,
                t_freeze=t_freeze,
                basin_grid=state_basin_grid,
            )
        else:
            state_measure = None
            state_basin_grid = None
            states = None

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
            if get_states:
                state_time = (
                    t_freeze
                    if state_measure == 'basin static' or state_measure == 'basin_static'
                    else (t_freeze if frozen and t_freeze is not None else t)
                )
                states[:, Delta_step] = self.get_cell_states(
                    state_time,
                    coordinate=y,
                    measure=state_measure,
                    t_freeze=t_freeze,
                    basin_grid=state_basin_grid,
                )
        if get_states:
            self.cell_states = states[:, -1]
        return traj, states

# ______________________________________________________________________________________________________________________
