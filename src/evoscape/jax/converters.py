import jax.numpy as jnp
import numpy as np

from evoscape.landscapes import Landscape
from evoscape.modules import Node, UnstableNode, NegCenter, Center
from evoscape.morphogen_regimes import mr_const, mr_linear_2signals, mr_piecewise, mr_sigmoid

from .types import ModuleDynamic, ModuleStatic, LandscapeDynamic, LandscapeStatic


def from_regime_to_number(regime):
    dic = {
        "mr_const" : 0,
        "mr_sigmoid" : 1,
        "mr_piecewise" : 2,
        "mr_linear_2signals": 3,
    }
    return dic[regime.__name__]


def landscape_to_pytree(landscape):
    """
    Convert Landscape object into JAX-friendly PyTrees:
    returns (dynamic, static)
    """

    if landscape.module_list:
        x, y, a, s = [], [], [], []
        tau_list = []
        use_tau_list = []
        J_list = []

        for module in landscape.module_list:
            x.append(module.x)
            y.append(module.y)
            a.append(module.a)
            s.append(module.s)
            J_list.append(module.J)

            if module.tau is None:
                tau_list.append(0.0)
                use_tau_list.append(False)
            else:
                tau_list.append(module.tau)
                use_tau_list.append(True)

        module_static = ModuleStatic(
            J=jnp.array(J_list),
            use_tau=jnp.array(use_tau_list),
            tau=jnp.array(tau_list),
        )

        module_dynamic = ModuleDynamic(
            x=jnp.array(x),
            y=jnp.array(y),
            a=jnp.array(a),
            s=jnp.array(s),
        )

    else:
        n_regimes = max(int(landscape.n_regimes), 1)

        module_static = ModuleStatic(
            J=jnp.zeros((0, 2, 2)),
            use_tau=jnp.zeros((0,), dtype=bool),
            tau=jnp.zeros((0,)),
        )

        module_dynamic = ModuleDynamic(
            x=jnp.zeros((0,)),
            y=jnp.zeros((0,)),
            a=jnp.zeros((0, n_regimes)),
            s=jnp.zeros((0, n_regimes)),
        )

    static = LandscapeStatic(
        A0=landscape.A0,
        x0=jnp.array(landscape.x0),
        n_regimes=landscape.n_regimes,
        regime_id=from_regime_to_number(landscape.regime),
        module=module_static,
        morphogen_times=jnp.array(landscape.morphogen_times),
        init_cond=jnp.array(landscape.init_cond),
    )

    dynamic = LandscapeDynamic(
        module=module_dynamic,
    )

    return dynamic, static

# !  When converting back to Landscape, using numpy regimes - Victoria
def from_number_to_regime(number):
    l = [mr_const, mr_sigmoid, mr_piecewise, mr_linear_2signals]
    return l[number]


def pytree_to_landscape(
    dynamic: LandscapeDynamic,
    static: LandscapeStatic,
    trajectories=None,
    cell_coordinates=None,
    cell_states=None,
    fitness=None,
    result=None,
):
    """
    Reconstruct original Landscape object from pytree representation
    """

    module_list = []

    n_modules = dynamic.module.x.shape[0]

    used_fp_types = {Node: False, UnstableNode: False, Center: False, NegCenter: False}
    for i in range(n_modules):
        tau = None
        if static.module.use_tau[i]:
            tau = float(static.module.tau[i])

        J = np.asarray(static.module.J[i])

        if np.array_equal(J, np.array(((-1, 0.0), (0.0, -1.0)))):
            module = Node(
                x=float(dynamic.module.x[i]),
                y=float(dynamic.module.y[i]),
                a=np.array(dynamic.module.a[i]),
                s=np.array(dynamic.module.s[i]),
                tau=tau,
            )
            used_fp_types[Node] = True
        elif np.array_equal(J, np.array(((+1, 0.0), (0.0, +1.0)))):
            module = UnstableNode(
                x=float(dynamic.module.x[i]),
                y=float(dynamic.module.y[i]),
                a=np.array(dynamic.module.a[i]),
                s=np.array(dynamic.module.s[i]),
                tau=tau,
            )
            used_fp_types[UnstableNode] = True
        elif np.array_equal(J, np.array(((0.0, -1.0), (+1.0, 0.0)))):
            module = Center(
                x=float(dynamic.module.x[i]),
                y=float(dynamic.module.y[i]),
                a=np.array(dynamic.module.a[i]),
                s=np.array(dynamic.module.s[i]),
                tau=tau,
            )
            used_fp_types[Center] = True
        else:
            module = NegCenter(
                x=float(dynamic.module.x[i]),
                y=float(dynamic.module.y[i]),
                a=np.array(dynamic.module.a[i]),
                s=np.array(dynamic.module.s[i]),
                tau=tau,
            )
            used_fp_types[NegCenter] = True
        module_list.append(module)

    used_fp_types = tuple(
        moduletype for moduletype, use in used_fp_types.items()
        if use
    )

    morphogen_times_array = np.asarray(static.morphogen_times)
    if morphogen_times_array.size == 0:
        morphogen_times = ()
    else:
        morphogen_times = tuple(np.atleast_1d(morphogen_times_array).tolist())

    landscape = Landscape(
        module_list=module_list,
        A0=float(static.A0),
        init_cond=tuple(np.asarray(static.init_cond).tolist()),
        regime=from_number_to_regime(static.regime_id),
        n_regimes=int(static.n_regimes),
        morphogen_times=morphogen_times,
        used_fp_types=used_fp_types,
        x0=tuple(np.asarray(static.x0).tolist()),
    )

    landscape.fitness = fitness
    landscape.result = result

    landscape.cell_coordinates = cell_coordinates
    landscape.cell_states = cell_states
    landscape.trajectories = trajectories

    return landscape
