import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np




@jax.jit
def JAXexponential_barrier(x, amplitude, sharpness, shift):
    """Exponential barrier "potential" to restrict a parameter in a loss function.

    Grows exponentially when `x` is greater than 30% of the derivative.

    Parameters
    ----------
    x : The parameter or variable
    amplitude : Amplitude of the exponential growth
    sharpness : Sharpness of the growth
    shift : Shift of the growth
    """
    return amplitude * (jnp.exp(sharpness * (x - 2*shift)))


@jax.jit
def JAXsoftplus_barrier(x, threshold):
    """Softplus (conitnuous ReLU) barrier "potential" to restrict a parameter in a loss function.

    Penalize when `x` start to go beyond the treshold

    Parameters
    ----------
    x : The parameter or variable
    threshold : The threshold at which we start to penalize
    """
    return jnp.log(1 + jnp.exp(x - threshold))



@jax.jit
def JAXgaussian_function(X, Y, params):
    """General gaussian function. It is anisotropic in x and y with sigmas and also rotated.

    Parameters
    ----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the gaussian, in order -> position X, position Y, amplitude, sigma x and sigma 2, theta, shape (N,6)
    """
    x0, y0, amp, size1, size2, theta = params
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    size1_squared = size1**2
    size2_squared = size2**2

    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    gaussian = amp * size1_squared * size2_squared * jnp.exp(-( (x_prime**2)/(2*size1_squared) + (y_prime**2)/(2*size2_squared)  ))

    return gaussian

@jax.jit
def JAXflow_grad(X, Y, params):
    """Gradient part of the flow based off the analytical gradient of a gaussian.

    Parameters
    ----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the flow, in order -> position X, position Y, amplitude, sigma x and sigma 2, theta, shape (N,6)
    """
    x0, y0, amp, size1, size2, theta = params
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    size1_squared = size1**2
    size2_squared = size2**2
    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    front = amp * jnp.exp(-( (x_prime**2)/(2*size1_squared) + (y_prime**2)/(2*size2_squared)  ))

    return front * jnp.array([x_prime * size2_squared * cosine - y_prime * size1_squared * sine,
                               x_prime * size2_squared * sine + y_prime * size1_squared * cosine])



@jax.jit
def JAXflow_rot(X, Y, params):
    """Rotational part of the flow based off the analytical skew gradient of a gaussian.

    Parameters
    -----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the flow, in order -> position X, position Y, amplitude, sigma x and sigma 2, theta, shape (N,6)
    """
    x0, y0, amp, size1, size2, theta = params
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    size1_squared = size1**2
    size2_squared = size2**2
    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    front = amp * jnp.exp(-( (x_prime**2)/(2*size1_squared) + (y_prime**2)/(2*size2_squared)  ))

    return -front * jnp.array([-x_prime * size2_squared * sine - y_prime * size1_squared * cosine,
                              x_prime * size2_squared * cosine - y_prime * size1_squared * sine])


@jax.jit
def JAXflow_grad_time(X, Y, params, t):
    """Gradient part of the flow based off the analytical gradient of a gaussian.

    Parameters
    ----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the flow, in order -> position X, position Y, amplitude, sigma x and sigma 2, theta, shape (N,6)
    t : Time dependance for the parameters
    """
    return JAXflow_grad(X, Y, params[t]) # return the flow for a specific time




@jax.jit
def JAXflow_rot_time(X, Y, params, t):
    """Rotational part of the flow based off the analytical skew gradient of a gaussian.

    Parameters
    -----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the flow, in order -> position X, position Y, amplitude, sigma x and sigma 2, theta, shape (N,6)
    t : Time dependance for the parameters
    """
    return JAXflow_rot(X, Y, params[t]) # return the flow for a specific time

@jax.jit
def JAXgaussian_mixture(X, Y, params):
    """Sum of multiple gaussians.

    Parameters
    ----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the gaussians, in order -> position X, position Y, amplitude, sigma x and sigma 2, theta, shape (N,6)
    
    Returns
    -------
        The sum of the gaussians
    """

    gaussians = jax.vmap(JAXgaussian_function, in_axes = (None, None, 0))(X, Y, params)

    return jnp.sum(gaussians, axis=0)


@jax.jit
def JAXflow_grad_mixture(X, Y, params):
    """Sum of multiple gradient flows from gaussians.

    Parameters
    ----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the gaussians, in order -> position X, position Y, amplitude, sigma x and sigma 2, theta, shape (N,6)
    
    Returns
    -------
        The sum of the gradient flows
    """
    grad_flows = jnp.sum(jax.vmap(JAXflow_grad, in_axes = (None, None, 0))(X, Y, params), axis=0)

    return grad_flows

@jax.jit
def JAXflow_rot_mixture(X, Y, params):
    """Sum of multiple rotational flows from gaussians.

    Parameters
    ----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the gaussians, in order -> position X, position Y, amplitude, sigma x and sigma 2, shape (N,6)
    
    Returns
    -------
        The sum of the rotational flows
    """
    rot_flows = jnp.sum(jax.vmap(JAXflow_rot, in_axes = (None, None, 0))(X, Y, params), axis=0)

    return rot_flows 


@jax.jit
def JAXflow_grad_mixture_time(X, Y, params, t):
    """Sum of multiple gradient flows from gaussians.

    Parameters
    ----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the gaussians, in order -> position X, position Y, amplitude, sigma x and sigma 2, theta, shape (N,6)
    t : Time dependance for the parameters
    
    Returns
    -------
        The sum of the gradient flows
    """
    grad_flows = jnp.sum(jax.vmap(JAXflow_grad_time, in_axes = (None, None, 0))(X, Y, params, t), axis=0)

    return grad_flows

@jax.jit
def JAXflow_rot_mixture_time(X, Y, params, t):
    """Sum of multiple rotational flows from gaussians.

    Parameters
    ----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the gaussians, in order -> position X, position Y, amplitude, sigma x and sigma 2, shape (N,6)
    t : Time dependance for the parameters

    Returns
    -------
        The sum of the rotational flows
    """
    rot_flows = jnp.sum(jax.vmap(JAXflow_rot_time, in_axes = (None, None, 0))(X, Y, params, t), axis=0)

    return rot_flows 




@jax.jit
def JAXflow(X, Y, params):
    """Both gradient and rotational flow based on gaussian modules.

    Parameters
    -----------
    X : X coordinate
    Y : Y coordinate
    params : Parameters for the flow, in order -> position X, position Y, amplitude, sigma x and sigma 2, theta, shape (N,6)
    """
    # grad
    x0, y0, amp, size1, size2, theta = jnp.transpose(params[0])
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    size1_squared = size1**2
    size2_squared = size2**2
    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    front = amp * jnp.exp(-( (x_prime**2)/(2*size1_squared) + (y_prime**2)/(2*size2_squared)  ))

    grad = front * jnp.array([x_prime * size2_squared * cosine - y_prime * size1_squared * sine,
                               x_prime * size2_squared * sine + y_prime * size1_squared * cosine])

    # rot
    x0, y0, amp, size1, size2, theta = jnp.transpose(params[1])
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    size1_squared = size1**2
    size2_squared = size2**2
    x_prime = (X - x0) * cosine + (Y - y0) * sine
    y_prime = -(X - x0) * sine + (Y - y0) * cosine

    front = amp * jnp.exp(-( (x_prime**2)/(2*size1_squared) + (y_prime**2)/(2*size2_squared)  ))

    rot = -front * jnp.array([-x_prime * size2_squared * sine - y_prime * size1_squared * cosine,
                              x_prime * size2_squared * cosine - y_prime * size1_squared * sine])
    
    return jnp.sum(grad + rot)




# Function for initializing random initial parameters
def random_init2(key, N, X, Y, max, min):
    """Randomly initialized parameters for a number of gaussians.

    `x0s` and `y0s` are uniformly sampled between extremum of meshgrid.

    `amplitudes` are uniformly sampled between extremum of the potential from helmholtz decomposition.

    `sigmas x` and `sigmas y` are uniformly sampled between `0.001` and `5` (arbitrary for now, in context of drosophila).

    `thetas` are uniformly sampled between `0` and `2*pi`.

    Parameters
    ----------
    key : jax.random.PRNGKey
    N : number of gaussian
    X, Y : meshgrid (shape: HxW)
    max, min : extremum of the potential from numerical helmholtz decomposition

    Returns
    -------
        A jnp array of stacked parameters of shape (N, 6)
    """

    subkeys = jax.random.split(key, 6)

    # spatial domain
    x_min, x_max = jnp.min(X), jnp.max(X)
    y_min, y_max = jnp.min(Y), jnp.max(Y)

    # random initialization of gaussians (x0, y0)
    x0s = jax.random.uniform(subkeys[0], shape=(N,), minval=x_min, maxval=x_max)
    y0s = jax.random.uniform(subkeys[1], shape=(N,), minval=y_min, maxval=y_max)


    # initialization of amplitudes
    amps = jax.random.uniform(subkeys[2], shape=(N,), minval=min, maxval=max)

    # initialization of size
    size1 = jax.random.uniform(subkeys[3], shape=(N,), minval=0.001, maxval = 5)
    size2 = jax.random.uniform(subkeys[4], shape=(N,), minval=0.001, maxval = 5)

    # initialization of rotation
    theta = jax.random.uniform(subkeys[5], shape=(N,), minval=0, maxval = 2*jnp.pi)

    # stacking everything as (N, 4)
    params = jnp.stack([x0s, y0s, amps, size1, size2, theta], axis=1)

    return params


# Function for initializing random initial parameters
def random_init(key, N, X, Y, max, min):
    """
    key : jax.random.PRNGKey
    N   : number of gaussian
    X,Y : meshgrid (shape: HxW)
    Z   : potential to approximate(shape: HxW)
    """

    subkeys = jax.random.split(key, 6)

    # spatial domain
    x_min, x_max = jnp.min(X)*2, jnp.max(X)*2
    y_min, y_max = jnp.min(Y)*2, jnp.max(Y)*2

    # random initialization of gaussians (x0, y0)
    x0s = jax.random.uniform(subkeys[0], shape=(N,), minval=x_min, maxval=x_max)
    y0s = jax.random.uniform(subkeys[1], shape=(N,), minval=y_min, maxval=y_max)


    # initialization of amplitudes
    amps = jax.random.uniform(subkeys[2], shape=(N,), minval=min, maxval=max)


    # initialization of size
    size1 = jax.random.uniform(subkeys[3], shape=(N,), minval=0.001, maxval = 5)
    size2 = jax.random.uniform(subkeys[4], shape=(N,), minval=0.001, maxval = 5)

    # initialization of rotation
    theta = jax.random.uniform(subkeys[5], shape=(N,), minval=0, maxval = 2*jnp.pi)

    # stacking everything as (N, 4)
    params = jnp.stack([x0s, y0s, amps, size1, size2, theta], axis=1)

    return params


# Function for initializing random initial parameters
def random_init_time(key, N, X, Y, max, min, time):
    """
    key : jax.random.PRNGKey
    N   : number of gaussian
    X,Y : meshgrid (shape: HxW)
    Z   : potential to approximate(shape: HxW)
    """

    subkeys = jax.random.split(key, 6)

    # spatial domain
    x_min, x_max = jnp.min(X)*2, jnp.max(X)*2
    y_min, y_max = jnp.min(Y)*2, jnp.max(Y)*2

    # random initialization of gaussians (x0, y0)
    x0s = jax.random.uniform(subkeys[0], shape=(N,), minval=x_min, maxval=x_max)
    y0s = jax.random.uniform(subkeys[1], shape=(N,), minval=y_min, maxval=y_max)


    # initialization of amplitudes
    amps = jax.random.uniform(subkeys[2], shape=(N,), minval=min, maxval=max)


    # initialization of size
    size1 = jax.random.uniform(subkeys[3], shape=(N,), minval=0.001, maxval = 5)
    size2 = jax.random.uniform(subkeys[4], shape=(N,), minval=0.001, maxval = 5)

    # initialization of rotation
    theta = jax.random.uniform(subkeys[5], shape=(N,), minval=0, maxval = 2*jnp.pi)

    # stacking everything as (N, 6)
    params0 = jnp.stack([x0s, y0s, amps, size1, size2, theta], axis=1)

    params = jnp.broadcast_to(params0, (time, N, 6))
    params = jnp.swapaxes(params, 0, 1)

    return params



def make_trajectories(params, data, ndt=50):
    nt = len(data[0,:,0])
    Delta_t = (nt-1)/nt
    dt = Delta_t / ndt
    print(f"dt={dt}")

    input = data[:,0,:] # shape (2,N)

    # donne (2,N)
    def f(pos):
        pos_x = pos[0]
        pos_y = pos[1]
        return JAXflow_grad_mixture(pos_x, pos_y, params[0]) + JAXflow_rot_mixture(pos_x, pos_y, params[1])

    def sum(pos, _):    
        pos = pos + f(pos) * dt
        return pos, pos # ('carryover', 'accumulated')

    initial_carry = input

    final_pos, traj = lax.scan(sum, initial_carry, None, length=nt*ndt - 1)

    traj = jnp.concatenate(
    [input[None, :, :], traj],
    axis=0
    )

    return final_pos, jnp.swapaxes(traj[::ndt,:,:], axis1=0, axis2=1)



def make_trajectories_personnal(func, params, init, length, dt):
    input = init

    # donne (2,N)
    def f(pos):
        pos_x = pos[0]
        pos_y = pos[1]
        return func(pos_x, pos_y, *params)

    def sum(pos, _):    
        pos = pos + f(pos) * dt
        return pos, pos # ('carryover', 'accumulated')

    initial_carry = input

    final_pos, traj = lax.scan(sum, initial_carry, None, length=length-1)

    traj = jnp.concatenate(
        [initial_carry[None, :], traj],
        axis=0
    )

    return final_pos, traj



def prediction_of_vectors(X, Y, params):
    params_grad, params_rot = params
    grad = jax.vmap(
        JAXflow_grad_mixture,
        in_axes=(0, 0, 1)
    )(X, Y, params_grad)

    rot = jax.vmap(
        JAXflow_rot_mixture,
        in_axes=(0, 0, 1)
    )(X, Y, params_rot)
    prediction = grad + rot

    return jnp.swapaxes(prediction, 0, 1)

def make_trajectories_time_dependance2(params, data, ndt=50):
    nt = len(data[0,:,0])
    Delta_t = (nt-1)/nt
    dt = Delta_t / ndt
    print(f"dt={dt}")

    input = data[:,0,:] # shape (2,N)
    start_time = 0

    """# donne (2,N)
    def f(spacio_time):
        pos, t = spacio_time
        pos_x = pos[0]
        pos_y = pos[1]
        
        return JAXflow_grad_mixture_time(pos_x, pos_y, params[0], t//ndt) + JAXflow_rot_mixture_time(pos_x, pos_y, params[1], t//ndt)
    """
    def sum(carry, _):
        pos, step = carry

        frame = step // ndt
        current_params = params[:,frame,:]

        v = (
            JAXflow_grad_mixture(pos[0], pos[1], current_params[0])
            + JAXflow_rot_mixture(pos[0], pos[1], current_params[1])
        )

        pos = pos + dt * v

        return (pos, step + 1), pos # ('carryover', 'accumulated')

    initial_carry = (input, start_time)

    final_pos, traj = lax.scan(sum, initial_carry, None, length=nt*ndt - 1)

    traj = jnp.concatenate(
    [input[None, :, :], traj],
    axis=0
    )

    return final_pos, jnp.swapaxes(traj[::ndt,:,:], axis1=0, axis2=1)


def make_trajectories_time_dependance(params, data, ndt=50):

    nt = data.shape[1]

    Delta_t = (nt-1)/nt
    dt = Delta_t / ndt

    params_grad, params_rot = params

    input = data[:,0,:]  # (2, number_points)

    def sum(carry, _):

        pos, step = carry

        frame = step // ndt

        grad_t = jax.lax.dynamic_index_in_dim(
            params_grad,
            frame,
            axis=1,
            keepdims=False
        )

        rot_t = jax.lax.dynamic_index_in_dim(
            params_rot,
            frame,
            axis=1,
            keepdims=False
        )

        v = (
            JAXflow_grad_mixture(pos[0], pos[1], grad_t)
            + JAXflow_rot_mixture(pos[0], pos[1], rot_t)
        )

        pos = pos + dt*v

        return (pos, step+1), pos


    initial_carry = (input, 0)

    final_pos, traj = lax.scan(
        sum,
        initial_carry,
        None,
        length=nt*ndt-1
    )

    traj = jnp.concatenate(
        [input[None,:,:], traj],
        axis=0
    )

    return final_pos, jnp.swapaxes(traj[::ndt,:,:], axis1=0, axis2=1)





def save_info(params_grad_l, params_rot_l, losses_l, directory_and_filename):
    """
    Saves the info of the module and the losses associated as a .npz file.
    Usually it's the "history" of the optimization. Meaning the evolution of the parameters through 100 epochs let say.
    """
    np.savez(
    directory_and_filename,
    params_grad=params_grad_l,
    params_rot=params_rot_l,
    losses=losses_l,
    )

def save_best_params(params_grad, params_rot, directory_and_filename):
    """
    Saves the best parameters of the optimization as a .npz file.
    """
    np.savez(
    directory_and_filename,
    best_params_grad=params_grad,
    best_params_rot=params_rot,
    )

def write_info(params_grad_l, params_rot_l, losses_l, directory_and_filename):
    """
    Writes the info of the module and the losses associated as a .txt file.
    Usually it's the "history" of the optimization. Meaning the evolution of the parameters through 100 epochs let say.
    """
    with open(directory_and_filename, "w") as f:

        for checkpoint in range(len(losses_l)):

            f.write(f"\n=== Checkpoint {checkpoint} ===\n")

            for candidate in range(losses_l.shape[1]): # before was gpu*times

                f.write(
                    f"Candidate {candidate} | "
                    f"Loss = {losses_l[checkpoint, candidate]:.6f}\n"
                )

                f.write(
                    f"Params grad = "
                    f"{params_grad_l[checkpoint, candidate]}\n"
                )
                f.write(
                    f"Params rot = "
                    f"{params_rot_l[checkpoint, candidate]}\n"
                )

            f.write("\n")

def write_ordered_result(ordered_params_grad, ordered_params_rot, ordered_losses, directory_and_filename):
    """
    Writes the ordered result of the optimisationas a .txt file.
    """
    with open(directory_and_filename, "w") as f:
        for candidate in range(len(ordered_losses)):

            f.write(
                f"Candidate {candidate} | "
                f"Loss = {ordered_losses[candidate]:.6f}\n"
            )

            f.write(
                f"Params grad = "
                f"{ordered_params_grad[candidate].tolist()}\n"
            )
            f.write(
                f"Params rot = "
                f"{ordered_params_rot[candidate].tolist()}\n"
            )

        f.write("\n")