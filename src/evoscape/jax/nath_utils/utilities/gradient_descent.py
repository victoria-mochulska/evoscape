import jax
import jax.numpy as jnp
from .helmholtz_decomp import *
from .module_gaussian import *
from .visualizations import *
from src.trajectory_loss_functions import compare_trajectories_all_points
from src.trajectory_loss_functions import compare_mini_traj
from src.trajectory_loss_functions import give_mini_traj
from src.trajectory_loss_functions import compare_end_part
from src.trajectory_loss_functions import compare_end_part_time
from src.trajectory_loss_functions import choose_point
from functools import partial


import tqdm
import optax
import time




@jax.jit
def cost_func2(params, Z) -> float:
    """Loss function from a trajectory. 
    
    We take the difference between the the data and the predicted trajectory, calculate the norm of that and then take the mean.

    Parameters
    ----------
    params : Parameters of the gaussians in shape (N,6)
    X, Y : Coordinates 
    Z : Data
    """
    _, prediction = make_trajectories(params, Z)

    return jnp.mean(JAXnorm(prediction - Z)**2)

@jax.jit
def loss_magnitude_func(fit, data):
    """Loss function that calculate the error on the vectors norm. 

    Parameters
    ----------
    fit : Fitted vectors
    data : Vectors from the data
    
    Returns
    -------
        The mean squared error on the vectors' norm.
    """

    magnitude_vec_data = JAXnorm(data)
    magnitude_vec_fit = JAXnorm(fit)
    loss_magnitude = (magnitude_vec_fit - magnitude_vec_data)**2
    return jnp.mean(loss_magnitude)

@jax.jit
def loss_angle_func(fit, data):
    """Loss function that calculate the error of the angles between 2 vectors.

    Parameters
    ----------
    fit : Fitted vectors
    data : Vectors from the data

    Returns
    -------
        The mean squared error on the vectors' angle
    """
    eps = 1e-12
    dot_product = jnp.sum(fit * data, axis=0)
    magnitude_vec_data = JAXnorm(data)
    magnitude_vec_fit = JAXnorm(fit)
    loss_angle = -1 * (dot_product / (magnitude_vec_fit * magnitude_vec_data + eps)) + 1
    return jnp.mean(loss_angle)

@jax.jit
def loss_sigma_func2(params, amplitude, sharpness, sigma_max):
    """Loss function that penalize very large sigmas (the standard deviation of the gaussian).

    We use an exponential function that penalize very large sigmas after a certain value. Usually
    it is defined as the diameter of the space we're working in. 

    Parameters
    ---------
    params : Parameters of the gaussians in shape (N,6)
    amplitude : Amplitude of the penalization potential
    sharpness : Sharpness of the penalization potential
    sigma_max : The biggest sigma we want to "allow"

    Returns
    -------
        The penalization on the sigmas with the "exponential barrier"
    """
    sigmas = jnp.concatenate([params[0][:, 3:5].reshape(-1), params[1][:, 3:5].reshape(-1)])
    sigmas_squared = sigmas**2
    loss_sigmas = JAXexponential_barrier(sigmas_squared, amplitude, sharpness, sigma_max)
    return jnp.sum(loss_sigmas)


@jax.jit
def loss_sigma_func(params, threshold, frac_height):
    """Loss function that penalize very large sigmas (the standard deviation of the gaussian).

    We use the softplus function that penalize very large sigmas after a certain value. Usually
    it is defined as the diameter of the space we're working in. 

    Parameters
    ---------
    params : Parameters of the gaussians in shape (N,6)
    threshold : The threshold at which we usually diameter of the space we work in
    frac_height: Fraction of the height of the gaussian we consider the width

    Returns
    -------
        The penalization on the sigmas with the "softplus barrier"
    """
    sigmas = jnp.concatenate([params[0][:, 3:5].reshape(-1), params[1][:, 3:5].reshape(-1)])
    sigmas_squared = sigmas**2
    value = 2*jnp.sqrt(-2*sigmas_squared * jnp.log(1/frac_height))
    loss_sigmas = JAXsoftplus_barrier(value, threshold)
    return jnp.sum(loss_sigmas)




@jax.jit 
def cost_func3(params, X, Y, Z, amplitude, sharpness, sigma_max):
    """Loss function for fitting a landscape with vector field.

    We do a linear combination of: 1. the error on the angle and magnitude of the vectors and 2. the penalization of the sigmas.

    Parameters
    ----------
    params : Parameters of the gaussians in shape (N, 6)
    X, Y : The coordinates of the data
    Z : The data
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas

    Returns
    -------
        A linear combination of both type of loss

    """
    prediction_vec = JAXflow_grad_mixture(X, Y, params[0]) + JAXflow_rot_mixture(X, Y, params[1])


    magnitude_angle_loss = loss_magnitude_func(prediction_vec, Z) + loss_angle_func(prediction_vec, Z)
    sigma_loss = loss_sigma_func2(params, amplitude, sharpness, sigma_max)

    return 0.2*(magnitude_angle_loss) + 0.8*(sigma_loss)   #0.3*(abc + deg) + 0.7*(mean_loss_sigmas)


@jax.jit 
def cost_func(params, X, Y, Z, threshold, frac_height, angle_importance):
    """Loss function for fitting a landscape with vector field.

    We do a linear combination of: 1. the error on the angle and magnitude of the vectors and 2. the penalization of the sigmas.

    Parameters
    ----------
    params : Parameters of the gaussians in shape (N, 6)
    X, Y : The coordinates of the data
    Z : The data
    threshold : The threshold at which we penalize usually diameter of the space we work in
    frac_height: Fraction of the height of the gaussian we consider the width

    Returns
    -------
        A linear combination of both type of loss

    """
    prediction_vec = JAXflow_grad_mixture(X, Y, params[0]) + JAXflow_rot_mixture(X, Y, params[1])


    magnitude_angle_loss = loss_magnitude_func(prediction_vec, Z) + angle_importance * loss_angle_func(prediction_vec, Z)
    sigma_loss = loss_sigma_func(params, threshold, frac_height)

    return 0.4*(magnitude_angle_loss) + 0.6*(sigma_loss)


# gradient of the loss function
grad_cost = jax.grad(cost_func, argnums=(0))






@jax.jit
def gradient_descent(X, Y, Z, lr, init_params_grad, init_params_rot, threshold, frac_height, iter):
    """Basic gradient descent using JAX automatic differentiation.

    We define a gradient step that calculate the gradient for a set of parameters. We normalize the gradient for both the gradient gaussians 
    and rotational gaussians. We make a step forward in this direction with the learning step. Since the parameters for the 
    gaussians can have a different number of gaussians for gradient and rotational we use a tuple that we pack at each step.

    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    lr : Learning step
    init_params : Initial parameters for both the rotational and gradient gaussians
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take

    Returns
    -------
        The loss at the and of the steps for the parameters, gradient parameters and rotational parameters
    """
    params = (init_params_grad, init_params_rot)

    def gradient_step(t, params):
        grad_params = grad_cost(params, X, Y, Z, threshold, frac_height)

        grad_grad, grad_rot = grad_params  # dépaquetage tuple

        grad_grad_norm = jnp.max(jnp.abs(grad_grad), axis=-1, keepdims=True)
        grad_rot_norm  = jnp.max(jnp.abs(grad_rot),  axis=-1, keepdims=True)

        new_params_grad = params[0] - lr * grad_grad / (grad_grad_norm + 1e-12)
        new_params_rot  = params[1] - lr * grad_rot  / (grad_rot_norm  + 1e-12)

        params = (new_params_grad, new_params_rot)

        return params

    params = jax.lax.fori_loop(0, iter, gradient_step, params)

    loss = cost_func(params, X, Y, Z, threshold, frac_height)

    return loss, params[0], params[1]





@jax.jit
def batch_gradient_descent(X, Y, Z, lr, params_grad_batch, params_rot_batch, threshold, frac_height, iter):
    """Batched version of the gradient descent.
     
    Take 1 set of parameters (gradient and rotational) and vmap it for multiple set of parameters. Makes it possible 
    to gradient descent multiple initial conditions at the same time.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    lr : Learning step
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    def single_run(params_grad_batch, params_rot_batch):
        return gradient_descent(X, Y, Z, lr, params_grad_batch, params_rot_batch, threshold, frac_height, iter)
    return jax.vmap(single_run)(params_grad_batch, params_rot_batch)


def train_verbose(X, Y, Z, lr, params_grad_batch, params_rot_batch, threshold, frac_height, maxiter, verbose_dt=500):
    """Verbose version of the batched gradient descent.
     
    Use the tqdm library to print where we are in the gradient descent iteration. Use the verbose_dt argument to choose
    which each time we print.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    lr : Learning step
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take
    verbose_dt : Number of iteration between eac

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    pbar = tqdm.tqdm(total=maxiter)
    evol = []

    for i in range(maxiter//verbose_dt):
        losses, params_grad_batch, params_rot_batch = batch_gradient_descent(X, Y, Z, lr, params_grad_batch, params_rot_batch, threshold, frac_height, verbose_dt)
        loss, _, _ = select_best(losses, params_grad_batch, params_rot_batch)
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(losses)

    return losses, params_grad_batch, params_rot_batch, evol









# Gradient descent function with optax optimizer
@partial(jax.jit, static_argnames=("optimizer", "threshold", "frac_height", "n_iter", "angle_importance"))
def gradient_descent_optax(X, Y, Z, optimizer, init_params_grad, init_params_rot, threshold, frac_height, angle_importance, n_iter):
    def step_optax(t, carry):
        """
        Step function for gradient descent with optax optimizer
        """
        params, opt_state = carry
        # retrieving loss and gradients from our cost function
        grads = grad_cost(params, X, Y, Z, threshold, frac_height, angle_importance)

        """# noise on parameters and offset based on the mean of the gradients and the offset
        noise_params = jnp.mean(grads[0]) * jax.random.normal(rng_key, shape=grads[0].shape)
        noise_offset = grads[1] * jax.random.normal(rng_key, shape=grads[1].shape)"""

        # adding the noise to the gradients
        #noisy_grads = (grads[0] + noise_params, grads[1] + noise_offset)
 
        # updating the optimizer
        updates, opt_state = optimizer.update(grads, opt_state)

        # new parameters and offset
        new_params = optax.apply_updates(params, updates)

        return (new_params, opt_state)
    
    # initial parameters
    params = (init_params_grad, init_params_rot)

    opt_state = optimizer.init(params)
    carry = (params, opt_state)

    carry = jax.lax.fori_loop(0, n_iter, step_optax, carry)

    loss = cost_func(carry[0], X, Y, Z, threshold, frac_height, angle_importance)


    return loss, carry[0]



@partial(jax.jit, static_argnames=("optimizer", "threshold", "frac_height", "iter", "angle_importance"))
def batch_optax(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, threshold, frac_height, angle_importance, iter):
    """Batched version of the optax gradient descent.
     
    Take 1 set of parameters (gradient and rotational) and vmap it for multiple set of parameters. Makes it possible 
    to gradient descent multiple initial conditions at the same time.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    def single_run(params_grad_batch, params_rot_batch):
        return gradient_descent_optax(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, threshold, frac_height, angle_importance, iter)
    return jax.vmap(single_run)(params_grad_batch, params_rot_batch)


def train_verbose_optax(X, Y, Z, lr, params_grad_batch, params_rot_batch, threshold, frac_height, angle_importance, maxiter, verbose_dt=500):
    """Verbose version of the optax batched gradient descent.
     
    Use the tqdm library to print where we are in the gradient descent iteration. Use the verbose_dt argument to choose
    which each time we print.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    lr : Learning step
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take
    verbose_dt : Number of iteration between eac

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    pbar = tqdm.tqdm(total=maxiter)
    evol = []

    params_grad_history = []
    params_rot_history = []
    loss_history = []
    # Optimizer for optax fitting
    # initializing the optimizer
    optimizer = optax.adam(lr)


    for i in range(maxiter//verbose_dt):
        losses, (params_grad_batch, params_rot_batch) = batch_optax(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, threshold, frac_height, angle_importance, verbose_dt)
        loss, _, _ = select_best(losses, params_grad_batch, params_rot_batch)
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(losses)

        params_grad_history.append(np.array(params_grad_batch))
        params_rot_history.append(np.array(params_rot_batch))
        loss_history.append(np.array(losses))

    return losses, params_grad_batch, params_rot_batch, evol, params_grad_history, params_rot_history, loss_history








def select_best(losses, params_grad, params_rot):
    """Function that select the best set of parameters for the modules based on the loss.

    We keep only the losses that are finite and then we take the minimum in the `losses` array.

    Parameters
    ----------
    loseses : Array of losses for all the conditions we tried in parallel
    params : Set of parameters in the same order of the losses

    Returns
    -------
        The best set of parameters and its loss
    """
    is_finite = jnp.isfinite(losses)
    if jnp.any(is_finite):
        filtered_losses = jnp.where(is_finite, losses, jnp.inf)
        idx = jnp.argmin(filtered_losses)

        if params_grad == None:
            return filtered_losses[idx], None, params_rot[idx]
        elif params_rot == None:
            return filtered_losses[idx], params_grad[idx], None
        else: 
            return filtered_losses[idx], params_grad[idx], params_rot[idx]
    else:
        return None, jnp.inf, jnp.inf
    

def select_best_first_run(losses):
    """Function that select the best set losses to get an idea of how many gaussian we need.

    In the algorithm, we try a lot of combination and calculate a trial loss for those. We keed the 
    best loss out of them. The index of this loss tells you how many gaussian we should need.

    Parameters
    ----------
    loseses : Array of losses for all the conditions we tried in parallel

    Returns
    -------
        The index of this best loss
    """
    losses = jnp.array(losses)
    is_finite = jnp.isfinite(losses)
    if jnp.any(is_finite):
        filtered_losses = jnp.where(is_finite, losses, jnp.inf)
        
        idx = jnp.argmin(filtered_losses)

        return filtered_losses[idx], idx


def vector_difference(vec1, vec2):
    """Compute the mean squared error on two vectors based on the difference of magnitude
    and angle.

    Parameters
    ----------
    vec1, vec2 : The two vector field we compare

    Returns
    -------
        The addition of both mean squared error
    """
    eps = 1e-12
    
    dot_product = jnp.sum(vec1 * vec2, axis=0)

    magnitude_vec1 = JAXnorm(vec1)
    magnitude_vec2 = JAXnorm(vec2)

    cos_theta = -1 * (dot_product / (magnitude_vec1 * magnitude_vec2 + eps)) + 1

    loss_magnitude = jnp.mean((magnitude_vec1 - magnitude_vec2)**2)
    loss_angle = jnp.mean(cos_theta)

    return loss_angle + loss_magnitude




def RMS_points(data, simulated_data):
    """Root mean squared of 2 set of points. This tell you how globally the trajectory is not good.

    Parameters
    ----------
    data : True data
    simulated_data : Simulated data
    """
    return jnp.sqrt(jnp.mean((data - simulated_data)**2))

def RMS_vectors(data, simulated_data):
    """Root mean squared of 2 set of vectors. This tell you how locally the trajectory is not good.

    Parameters
    ----------
    data : True data
    simulated_data : Simulated data
    """
    return jnp.sqrt(jnp.mean(JAXnorm(data - simulated_data)**2))





























#### DESCENT WITH TRAJ


@jax.jit 
def cost_func_traj_all_pt(params, X, Y, Z, Z_pt, threshold, frac_height):
    """Loss function for fitting a landscape with vector field.

    We do a linear combination of: 1. the error on the angle and magnitude of the vectors and 2. the penalization of the sigmas.

    Parameters
    ----------
    params : Parameters of the gaussians in shape (N, 6)
    X, Y : The coordinates of the data
    Z : The data
    threshold : The threshold at which we penalize usually diameter of the space we work in
    frac_height: Fraction of the height of the gaussian we consider the width

    Returns
    -------
        A linear combination of both type of loss

    """
    prediction_vec = JAXflow_grad_mixture(X, Y, params[0]) + JAXflow_rot_mixture(X, Y, params[1])


    magnitude_angle_loss = loss_magnitude_func(prediction_vec, Z) + loss_angle_func(prediction_vec, Z)
    sigma_loss = loss_sigma_func(params, threshold, frac_height)
    C = jnp.swapaxes(Z_pt, 0, 2)
    loss_traj = compare_trajectories_all_points(params, C, C[:,0,:])


    return 0.9*(magnitude_angle_loss) + 0.1*(sigma_loss) + loss_traj


grad_cost_traj = jax.grad(cost_func_traj_all_pt, argnums=(0))




# Gradient descent function with optax optimizer
@jax.jit
def gradient_descent_optax_traj(X, Y, Z, Z_pt, optimizer, init_params_grad, init_params_rot, threshold, frac_height, n_iter):
    def step_optax(t, carry):
        """
        Step function for gradient descent with optax optimizer
        """
        params, opt_state = carry
        # retrieving loss and gradients from our cost function
        grads = grad_cost_traj(params, X, Y, Z, Z_pt, threshold, frac_height)

        """# noise on parameters and offset based on the mean of the gradients and the offset
        noise_params = jnp.mean(grads[0]) * jax.random.normal(rng_key, shape=grads[0].shape)
        noise_offset = grads[1] * jax.random.normal(rng_key, shape=grads[1].shape)"""

        # adding the noise to the gradients
        #noisy_grads = (grads[0] + noise_params, grads[1] + noise_offset)
 
        # updating the optimizer
        updates, opt_state = optimizer.update(grads, opt_state)

        # new parameters and offset
        new_params = optax.apply_updates(params, updates)

        return (new_params, opt_state)
    
    # initial parameters
    params = (init_params_grad, init_params_rot)

    opt_state = optimizer.init(params)
    carry = (params, opt_state)

    carry = jax.lax.fori_loop(0, n_iter, step_optax, carry)

    loss = cost_func_traj_all_pt(carry[0], X, Y, Z, Z_pt, threshold, frac_height)


    return loss, carry[0]



@jax.jit
def batch_optax_traj(X, Y, Z, Z_pt, optimizer, params_grad_batch, params_rot_batch, threshold, frac_height, iter):
    """Batched version of the optax gradient descent.
     
    Take 1 set of parameters (gradient and rotational) and vmap it for multiple set of parameters. Makes it possible 
    to gradient descent multiple initial conditions at the same time.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    def single_run(params_grad_batch, params_rot_batch):
        return gradient_descent_optax_traj(X, Y, Z, Z_pt, optimizer, params_grad_batch, params_rot_batch, threshold, frac_height, iter)
    return jax.vmap(single_run)(params_grad_batch, params_rot_batch)


def train_verbose_optax_traj(X, Y, Z, Z_pt, optimizer, lr, params_grad_batch, params_rot_batch, threshold, frac_height, maxiter, verbose_dt=500):
    """Verbose version of the optax batched gradient descent.
     
    Use the tqdm library to print where we are in the gradient descent iteration. Use the verbose_dt argument to choose
    which each time we print.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    lr : Learning step
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take
    verbose_dt : Number of iteration between eac

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    pbar = tqdm.tqdm(total=maxiter)
    evol = []
    optimizer = optax.adam(lr)

    # Optimizer for optax fitting
    # initializing the optimizer

    for i in range(maxiter//verbose_dt):
        losses, (params_grad_batch, params_rot_batch) = batch_optax_traj(X, Y, Z, optimizer, Z_pt, params_grad_batch, params_rot_batch, threshold, frac_height, verbose_dt)
        loss, _, _ = select_best(losses, params_grad_batch, params_rot_batch)
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(losses)

    return losses, params_grad_batch, params_rot_batch, evol
































### FRAC POINT TRAJ


@partial(jax.jit, static_argnames=("threshold", "frac_height", "length_of_mini_traj", "Delta_t"))
def cost_func_traj_frac_pt(params, X, Y, Z, threshold, frac_height, initial_points, mini_traj, length_of_mini_traj, Delta_t):
    """Loss function for fitting a landscape with vector field.

    We do a linear combination of: 1. the error on the angle and magnitude of the vectors and 2. the penalization of the sigmas.

    Parameters
    ----------
    params : Parameters of the gaussians in shape (N, 6)
    X, Y : The coordinates of the data
    Z : The data
    threshold : The threshold at which we penalize usually diameter of the space we work in
    frac_height: Fraction of the height of the gaussian we consider the width

    Returns
    -------
        A linear combination of both type of loss

    """
    prediction_vec = JAXflow_grad_mixture(X, Y, params[0]) + JAXflow_rot_mixture(X, Y, params[1])


    magnitude_angle_loss = loss_magnitude_func(prediction_vec, Z) + loss_angle_func(prediction_vec, Z)
    #sigma_loss = loss_sigma_func(params, threshold, frac_height)


    loss_traj = compare_mini_traj(params, initial_points, mini_traj, length_of_mini_traj, Delta_t)


    return (magnitude_angle_loss) + loss_traj


grad_cost_traj_frac = jax.grad(cost_func_traj_frac_pt, argnums=(0))




# Gradient descent function with optax optimizer
@partial(jax.jit, static_argnames=("optimizer", "n_iter", "threshold", "frac_height", "length_of_mini_traj", "Delta_t"))
def gradient_descent_optax_traj_frac(X, Y, Z, optimizer, init_params_grad, init_params_rot, threshold, frac_height, initial_points, mini_traj, length_of_mini_traj, Delta_t, n_iter):
    def step_optax(t, carry):
        """
        Step function for gradient descent with optax optimizer
        """
        params, opt_state = carry
        # retrieving loss and gradients from our cost function
        grads = grad_cost_traj_frac(params, X, Y, Z, threshold, frac_height, initial_points, mini_traj, length_of_mini_traj, Delta_t)

        """# noise on parameters and offset based on the mean of the gradients and the offset
        noise_params = jnp.mean(grads[0]) * jax.random.normal(rng_key, shape=grads[0].shape)
        noise_offset = grads[1] * jax.random.normal(rng_key, shape=grads[1].shape)"""

        # adding the noise to the gradients
        #noisy_grads = (grads[0] + noise_params, grads[1] + noise_offset)
 
        # updating the optimizer
        updates, opt_state = optimizer.update(grads, opt_state)

        # new parameters and offset
        new_params = optax.apply_updates(params, updates)

        return (new_params, opt_state)
    
    # initial parameters
    params = (init_params_grad, init_params_rot)

    opt_state = optimizer.init(params)
    carry = (params, opt_state)

    carry = jax.lax.fori_loop(0, n_iter, step_optax, carry)

    loss = cost_func_traj_frac_pt(carry[0], X, Y, Z, threshold, frac_height, initial_points, mini_traj, length_of_mini_traj, Delta_t)


    return loss, carry[0]



@partial(jax.jit, static_argnames=("optimizer","n_iter", "threshold", "frac_height", "length_of_mini_traj", "Delta_t"))
def batch_optax_traj_frac(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, threshold, frac_height, initial_points, mini_traj, length_of_mini_traj, Delta_t, n_iter):
    """Batched version of the optax gradient descent.
     
    Take 1 set of parameters (gradient and rotational) and vmap it for multiple set of parameters. Makes it possible 
    to gradient descent multiple initial conditions at the same time.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    def single_run(params_grad_batch, params_rot_batch):
        return gradient_descent_optax_traj_frac(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, threshold, frac_height, initial_points, mini_traj, length_of_mini_traj, Delta_t, n_iter)
    return jax.vmap(single_run)(params_grad_batch, params_rot_batch)


def train_verbose_optax_traj_frac(X, Y, Z, Z_pt, lr, params_grad_batch, params_rot_batch, threshold, frac_height, frac_time, frac_space, maxiter, verbose_dt=500):
    """Verbose version of the optax batched gradient descent.
     
    Use the tqdm library to print where we are in the gradient descent iteration. Use the verbose_dt argument to choose
    which each time we print.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    lr : Learning step
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take
    verbose_dt : Number of iteration between eac

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    pbar = tqdm.tqdm(total=maxiter)
    evol = []
    optimizer = optax.adam(lr)

    coord, shape_time, shape_space = Z_pt.shape
    Delta_t = (shape_time-1)/shape_time

    interval_of_trajectories = int((100/frac_space))
    length_of_mini_traj = int(shape_time * (frac_time/100))

    initial_points, mini_traj = give_mini_traj(jnp.swapaxes(Z_pt, 0, 2), interval_of_trajectories, length_of_mini_traj)
    print(type(length_of_mini_traj))
    print(type(Delta_t))
    print(type(initial_points))
    print(type(mini_traj))

    for i in range(maxiter//verbose_dt):
        losses, (params_grad_batch, params_rot_batch) = batch_optax_traj_frac(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, threshold, frac_height, initial_points, mini_traj, length_of_mini_traj, Delta_t, verbose_dt)
        loss, _, _ = select_best(losses, params_grad_batch, params_rot_batch)
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(losses)

    return losses, params_grad_batch, params_rot_batch, evol


























### RAND POINT TRAJ


@partial(jax.jit, static_argnames=("length_traj", "Delta_t"))
def cost_func_traj_rand_pt(params, X, Y, Z, traj, initial_points, length_traj, Delta_t):
    """Loss function for fitting a landscape with vector field.

    We do a linear combination of: 1. the error on the angle and magnitude of the vectors and 2. the penalization of the sigmas.

    Parameters
    ----------
    params : Parameters of the gaussians in shape (N, 6)
    X, Y : The coordinates of the data
    Z : The data
    threshold : The threshold at which we penalize usually diameter of the space we work in
    frac_height: Fraction of the height of the gaussian we consider the width

    Returns
    -------
        A linear combination of both type of loss

    """
    prediction_vec = JAXflow_grad_mixture(X, Y, params[0]) + JAXflow_rot_mixture(X, Y, params[1])


    magnitude_angle_loss = loss_magnitude_func(prediction_vec, Z) + loss_angle_func(prediction_vec, Z)
    #sigma_loss = loss_sigma_func(params, threshold, frac_height)

    loss_traj = compare_end_part(params, traj, initial_points, length_traj, Delta_t)


    return (magnitude_angle_loss) + loss_traj


grad_cost_traj_rand = jax.grad(cost_func_traj_rand_pt, argnums=(0))




# Gradient descent function with optax optimizer
@partial(jax.jit, static_argnames=("optimizer","n_iter", "num_pts_iter", "length_traj", "Delta_t"))
def gradient_descent_optax_traj_rand(X, Y, Z, optimizer, init_params_grad, init_params_rot, data, data_okay, keys, num_pts_iter, length_traj, Delta_t, n_iter):
    def step_optax(t, carry):
        """
        Step function for gradient descent with optax optimizer
        """
        key = keys[t]

        indices, initial_points, traj = choose_point(key, data, data_okay, num_pts_iter, length_traj)
        
        params, opt_state = carry
        # retrieving loss and gradients from our cost function
        grads = grad_cost_traj_rand(params, X, Y, Z, traj, initial_points, length_traj, Delta_t)

        """# noise on parameters and offset based on the mean of the gradients and the offset
        noise_params = jnp.mean(grads[0]) * jax.random.normal(rng_key, shape=grads[0].shape)
        noise_offset = grads[1] * jax.random.normal(rng_key, shape=grads[1].shape)"""

        # adding the noise to the gradients
        #noisy_grads = (grads[0] + noise_params, grads[1] + noise_offset)
 
        # updating the optimizer
        updates, opt_state = optimizer.update(grads, opt_state)

        # new parameters and offset
        new_params = optax.apply_updates(params, updates)

        return (new_params, opt_state)
    
    # initial parameters
    params = (init_params_grad, init_params_rot)

    opt_state = optimizer.init(params)
    carry = (params, opt_state)

    carry = jax.lax.fori_loop(0, n_iter, step_optax, carry)
    indices, initial_points, traj = choose_point(keys[-1], data, data_okay, num_pts_iter, length_traj)
    loss = cost_func_traj_rand_pt(carry[0], X, Y, Z, traj, initial_points, length_traj, Delta_t)


    return loss, carry[0]



@partial(jax.jit, static_argnames=("optimizer","n_iter", "num_pts_iter", "length_traj", "Delta_t"))
def batch_optax_traj_rand(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, data, data_okay, keys, num_pts_iter, length_traj, Delta_t, n_iter):
    """Batched version of the optax gradient descent.
     
    Take 1 set of parameters (gradient and rotational) and vmap it for multiple set of parameters. Makes it possible 
    to gradient descent multiple initial conditions at the same time.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    def single_run(params_grad_batch, params_rot_batch):
        return gradient_descent_optax_traj_rand(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, data, data_okay, keys, num_pts_iter, length_traj, Delta_t, n_iter)
    return jax.vmap(single_run)(params_grad_batch, params_rot_batch)


def train_verbose_optax_traj_rand(X, Y, Z, lr, percent_data, params_grad_batch, params_rot_batch, data, base_key, length_traj, maxiter, verbose_dt=500):
    """Verbose version of the optax batched gradient descent.
     
    Use the tqdm library to print where we are in the gradient descent iteration. Use the verbose_dt argument to choose
    which each time we print.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    lr : Learning step
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take
    verbose_dt : Number of iteration between eac

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    pbar = tqdm.tqdm(total=maxiter)
    evol = []
    params_grad_history = []
    params_rot_history = []
    loss_history = []
    Delta_t = (data.shape[1]-1)/data.shape[1]


    data_okay = data[:, :-length_traj ,:]
    shape_space, shape_time, coord = data_okay.shape
    optimizer = optax.adam(lr)
    print(lr)

    print("shapes", shape_space, shape_time)
    num_pts_iter =  int((shape_space * shape_time) * (percent_data/100))
    print("num pts", num_pts_iter)

    keys = jax.random.split(base_key, num=maxiter)


    for i in range(maxiter//verbose_dt):
        losses, (params_grad_batch, params_rot_batch) = batch_optax_traj_rand(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, data, data_okay, keys, num_pts_iter, length_traj, Delta_t, verbose_dt)
        loss, _, _ = select_best(losses, params_grad_batch, params_rot_batch)
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(losses)

        params_grad_history.append(np.array(params_grad_batch))
        params_rot_history.append(np.array(params_rot_batch))
        loss_history.append(np.array(losses))

    return losses, params_grad_batch, params_rot_batch, evol, params_grad_history, params_rot_history, loss_history


































































@partial(jax.jit, static_argnames=("derivative_importance"))
def cost_func_time(params, X, Y, Z, derivative_importance):
    """Loss function for fitting a landscape with vector field.

    We do a linear combination of: 1. the error on the angle and magnitude of the vectors and 2. the penalization of the sigmas.

    Parameters
    ----------
    params : Parameters of the gaussians in shape (N, 6)
    X, Y : The coordinates of the data
    Z : The data
    threshold : The threshold at which we penalize usually diameter of the space we work in
    frac_height: Fraction of the height of the gaussian we consider the width

    Returns
    -------
        A linear combination of both type of loss

    """
    prediction_vec = prediction_of_vectors(X, Y, params)


    magnitude_angle_loss = loss_magnitude_func(prediction_vec, Z) + loss_angle_func(prediction_vec, Z)

    loss_change_parameters_velocity = minimal_param_change_velocity(params)

    loss_change_parameters_acceleration = minimal_param_change_acceleration(params)

    return (magnitude_angle_loss) + derivative_importance * (loss_change_parameters_velocity + loss_change_parameters_acceleration)


grad_cost_time = jax.grad(cost_func_time, argnums=(0))






# Gradient descent function with optax optimizer
@partial(jax.jit, static_argnames=("optimizer", "derivative_importance","n_iter"))
def gradient_descent_optax_time(X, Y, Z, optimizer, init_params_grad, init_params_rot, derivative_importance, n_iter):
    def step_optax(t, carry):
        """
        Step function for gradient descent with optax optimizer
        """
        params, opt_state = carry
        # retrieving loss and gradients from our cost function
        grads = grad_cost_time(params, X, Y, Z, derivative_importance)

 
        # updating the optimizer
        updates, opt_state = optimizer.update(grads, opt_state)

        # new parameters and offset
        new_params = optax.apply_updates(params, updates)

        return (new_params, opt_state)
    
    # initial parameters
    params = (init_params_grad, init_params_rot)

    opt_state = optimizer.init(params)
    carry = (params, opt_state)

    carry = jax.lax.fori_loop(0, n_iter, step_optax, carry)

    loss = cost_func_time(carry[0], X, Y, Z, derivative_importance)


    return loss, carry[0]



@partial(jax.jit, static_argnames=("optimizer", "derivative_importance", "iter"))
def batch_optax_time(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, derivative_importance, iter):
    """Batched version of the optax gradient descent.
     
    Take 1 set of parameters (gradient and rotational) and vmap it for multiple set of parameters. Makes it possible 
    to gradient descent multiple initial conditions at the same time.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    def single_run(params_grad_batch, params_rot_batch):
        return gradient_descent_optax_time(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, derivative_importance, iter)
    return jax.vmap(single_run)(params_grad_batch, params_rot_batch)


def train_verbose_optax_time(X, Y, Z, lr, params_grad_batch, params_rot_batch, derivative_importance, maxiter, verbose_dt=500):
    """Verbose version of the optax batched gradient descent.
     
    Use the tqdm library to print where we are in the gradient descent iteration. Use the verbose_dt argument to choose
    which each time we print.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    lr : Learning step
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take
    verbose_dt : Number of iteration between eac

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    pbar = tqdm.tqdm(total=maxiter)
    evol = []

    params_grad_history = []
    params_rot_history = []
    loss_history = []
    # Optimizer for optax fitting
    # initializing the optimizer
    optimizer = optax.adam(lr)


    for i in range(maxiter//verbose_dt):
        losses, (params_grad_batch, params_rot_batch) = batch_optax_time(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, derivative_importance, verbose_dt)
        loss, _, _ = select_best(losses, params_grad_batch, params_rot_batch)
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(losses)

        params_grad_history.append(np.array(params_grad_batch))
        params_rot_history.append(np.array(params_rot_batch))
        loss_history.append(np.array(losses))

    return losses, params_grad_batch, params_rot_batch, evol, params_grad_history, params_rot_history, loss_history




































@jax.jit
def minimal_param_change_velocity(params):
    grad_params, rot_params = params

    grad_derivative = grad_params[:, 1:, :] - grad_params[:, :-1, :]
    rot_derivative = rot_params[:, 1:, :] - rot_params[:, :-1, :]

    return jnp.mean(jnp.sum(grad_derivative**2, axis=-1)) + jnp.mean(jnp.sum(rot_derivative**2, axis=-1))

@jax.jit
def minimal_param_change_acceleration(params):
    grad_params, rot_params = params

    grad_derivative2 = grad_params[:, 2:, :] - 2 * grad_params[:, 1:-1, :] + grad_params[:, :-2, :]
    rot_derivative2 = rot_params[:, 2:, :] - 2 * rot_params[:, 1:-1, :] + rot_params[:, :-2, :]

    return jnp.mean(jnp.sum(grad_derivative2**2, axis=-1)) + jnp.mean(jnp.sum(rot_derivative2**2, axis=-1))


### RAND POINT TRAJ


@partial(jax.jit, static_argnames=("length_traj", "Delta_t", "derivative_importance"))
def cost_func_traj_rand_pt_time(params, X, Y, Z, traj, initial_points, time_coords, length_traj, derivative_importance, Delta_t):
    """Loss function for fitting a landscape with vector field.

    We do a linear combination of: 1. the error on the angle and magnitude of the vectors and 2. the penalization of the sigmas.

    Parameters
    ----------
    params : Parameters of the gaussians in shape (N, 6)
    X, Y : The coordinates of the data
    Z : The data
    threshold : The threshold at which we penalize usually diameter of the space we work in
    frac_height: Fraction of the height of the gaussian we consider the width

    Returns
    -------
        A linear combination of both type of loss

    """
    prediction_vec = prediction_of_vectors(X, Y, params)


    magnitude_angle_loss = loss_magnitude_func(prediction_vec, Z) + loss_angle_func(prediction_vec, Z)

    loss_traj = compare_end_part_time(params, traj, initial_points, time_coords, length_traj, Delta_t)

    loss_change_parameters_velocity = minimal_param_change_velocity(params)

    loss_change_parameters_acceleration = minimal_param_change_acceleration(params)

    return (magnitude_angle_loss) + loss_traj + derivative_importance * (loss_change_parameters_velocity + loss_change_parameters_acceleration)


grad_cost_traj_rand_time = jax.grad(cost_func_traj_rand_pt_time, argnums=(0))




# Gradient descent function with optax optimizer
@partial(jax.jit, static_argnames=("optimizer","n_iter", "num_pts_iter", "length_traj", "Delta_t", "derivative_importance"))
def gradient_descent_optax_traj_rand_time(X, Y, Z, optimizer, init_params_grad, init_params_rot, data, data_okay, keys, num_pts_iter, length_traj, derivative_importance, Delta_t, n_iter):
    def step_optax(t, carry):
        """
        Step function for gradient descent with optax optimizer
        """
        key = keys[t]

        (space_coords, time_coords), initial_points, traj = choose_point(key, data, data_okay, num_pts_iter, length_traj)
        
        params, opt_state = carry
        # retrieving loss and gradients from our cost function
        grads = grad_cost_traj_rand_time(params, X, Y, Z, traj, initial_points, time_coords, length_traj, derivative_importance, Delta_t)

        """# noise on parameters and offset based on the mean of the gradients and the offset
        noise_params = jnp.mean(grads[0]) * jax.random.normal(rng_key, shape=grads[0].shape)
        noise_offset = grads[1] * jax.random.normal(rng_key, shape=grads[1].shape)"""

        # adding the noise to the gradients
        #noisy_grads = (grads[0] + noise_params, grads[1] + noise_offset)
 
        # updating the optimizer
        updates, opt_state = optimizer.update(grads, opt_state)

        # new parameters and offset
        new_params = optax.apply_updates(params, updates)

        return (new_params, opt_state)
    
    # initial parameters
    params = (init_params_grad, init_params_rot)

    opt_state = optimizer.init(params)
    carry = (params, opt_state)

    carry = jax.lax.fori_loop(0, n_iter, step_optax, carry)
    (space_coords, time_coords), initial_points, traj = choose_point(keys[-1], data, data_okay, num_pts_iter, length_traj)
    loss = cost_func_traj_rand_pt_time(carry[0], X, Y, Z, traj, initial_points, time_coords, length_traj, derivative_importance, Delta_t)


    return loss, carry[0]



@partial(jax.jit, static_argnames=("optimizer","n_iter", "num_pts_iter", "length_traj", "Delta_t", "derivative_importance"))
def batch_optax_traj_rand_time(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, data, data_okay, keys, num_pts_iter, length_traj, derivative_importance, Delta_t, n_iter):
    """Batched version of the optax gradient descent.
     
    Take 1 set of parameters (gradient and rotational) and vmap it for multiple set of parameters. Makes it possible 
    to gradient descent multiple initial conditions at the same time.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    def single_run(params_grad_batch, params_rot_batch):
        return gradient_descent_optax_traj_rand_time(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, data, data_okay, keys, num_pts_iter, length_traj, derivative_importance, Delta_t, n_iter)
    return jax.vmap(single_run)(params_grad_batch, params_rot_batch)


def train_verbose_optax_traj_rand_time(X, Y, Z, lr, percent_data, params_grad_batch, params_rot_batch, data, base_key, length_traj, derivative_importance, maxiter, verbose_dt=500):
    """Verbose version of the optax batched gradient descent.
     
    Use the tqdm library to print where we are in the gradient descent iteration. Use the verbose_dt argument to choose
    which each time we print.
    
    Parameters
    ----------
    X, Y : Coordinates of our space
    Z : Data
    lr : Learning step
    params_batch : Initial parameters for both the rotational and gradient gaussians but batched
    amplitude : Amplitude of the penalization potential for the sigmas
    sharpness : Sharpness of the penalization potential for the sigmas
    sigma_max : The biggest sigma we want to "allow" for the penalization of the sigmas
    iter : Number of step we want to take
    verbose_dt : Number of iteration between eac

    Returns
    -------
        The losses at the and of the steps for the parameters, batched gradient parameters and batched rotational parameters
    """
    pbar = tqdm.tqdm(total=maxiter)
    evol = []
    params_grad_history = []
    params_rot_history = []
    loss_history = []
    Delta_t = (data.shape[1]-1)/data.shape[1]


    data_okay = data[:, :-length_traj ,:]
    shape_space, shape_time, coord = data_okay.shape
    optimizer = optax.adam(lr)
    print(lr)

    print("shapes", shape_space, shape_time)
    num_pts_iter =  int((shape_space * shape_time) * (percent_data/100))
    print("num pts", num_pts_iter)

    keys = jax.random.split(base_key, num=maxiter)


    for i in range(maxiter//verbose_dt):
        losses, (params_grad_batch, params_rot_batch) = batch_optax_traj_rand_time(X, Y, Z, optimizer, params_grad_batch, params_rot_batch, data, data_okay, keys, num_pts_iter, length_traj, derivative_importance, Delta_t, verbose_dt)
        loss, _, _ = select_best(losses, params_grad_batch, params_rot_batch)
        pbar.update(verbose_dt)
        pbar.set_postfix({'loss': loss})
        evol.append(losses)

        params_grad_history.append(np.array(params_grad_batch))
        params_rot_history.append(np.array(params_rot_batch))
        loss_history.append(np.array(losses))

    return losses, params_grad_batch, params_rot_batch, evol, params_grad_history, params_rot_history, loss_history




