from .landscape_pytree import landscape_to_pytree
from .optim import make_step, clip_dynamic, run_optimization
from .utils import init_cell
from .dynamics import _integrate
from .regimes import wrapped_regime
# Main function running the landscape optimization 
def run_experiment(
    landscape,
    key,
    n,
    cell_noise,
    t0,
    tf,
    nt,
    ndt,
    noise,
    iterations,
    η,
    get_states,
    loss_function, 
    loss_param= None,
    signal_param = None
):
    dynamic, static = landscape_to_pytree(landscape)

    regime = wrapped_regime(static, signal_param)
    if loss_param is None:
        def loss_fn(dyn, key):
            return loss_function(dyn, static,  n, cell_noise, t0, tf, nt, ndt, noise, key, regime, get_states)
    else:
        def loss_fn(dyn, key):
            return loss_function(dyn, static, n, cell_noise, t0, tf, nt, ndt, noise, key, regime, get_states, loss_param)

    step_fn = make_step(loss_fn, η, clip_dynamic)

    (dynamic, key), losses = run_optimization(
        dynamic,
        key,
        step_fn,
        iterations
    )

    # Final states
    key, y0 = init_cell(key,n, dynamic.init_cond,noise=noise)
    _, traj, states = _integrate(key, y0, t0, tf, nt, ndt, noise,dynamic,static, regime, get_states)

    return dynamic, static, losses, traj, states