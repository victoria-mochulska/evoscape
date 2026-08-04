import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from .visualizations import *
import jax
import jax.numpy as jnp

dx = dy = 0.01          # increment

WIDTH = 4
npoints = 501
# 2D Grid
x = np.linspace(-WIDTH, WIDTH, npoints)
y = np.linspace(-WIDTH, WIDTH, npoints)
X, Y = np.meshgrid(x, y)


points = x.size


def F(x,y):
    #return [-y*x/(x**2 + y**2 + 1), x/(x**2 + y**2 + 1)]
    return [10*(2*x - 8*(x**3)/3 - 2*y)*np.exp(-x**2-y**2), 0.1*(2*x - 0.2*y+ 0.02)*np.exp(-x**2-y**2)]
    #return [x * np.exp(-x**2 - y**2), y * np.exp(-x**2 - y**2)]

vector_field = F(X,Y)



def gradient_gaussians(gaussian_mixture):
    @jax.jit
    def JAXgrad_FCN(X, Y, params):
        grad_x = jax.grad(gaussian_mixture, argnums=0)
        grad_y = jax.grad(gaussian_mixture, argnums=1)

        X_flat = X.reshape(-1)
        Y_flat = Y.reshape(-1)

        dF_dx = jax.vmap(grad_x, in_axes=(0, 0, None))(X_flat, Y_flat, params).reshape(X.shape)
        dF_dy = jax.vmap(grad_y, in_axes=(0, 0, None))(X_flat, Y_flat, params).reshape(Y.shape)

        return -1 * jnp.stack([dF_dx, dF_dy], axis=0)
    return JAXgrad_FCN

def skewgradient_gaussians(gaussian_mixture):
    @jax.jit
    def JAXskewgrad_FCN(X, Y, params):
        grad_x = jax.grad(gaussian_mixture, argnums=0)
        grad_y = jax.grad(gaussian_mixture, argnums=1)

        X_flat = X.reshape(-1)
        Y_flat = Y.reshape(-1)

        dF_dx = jax.vmap(grad_x, in_axes=(0, 0, None))(X_flat, Y_flat, params).reshape(X.shape)
        dF_dy = jax.vmap(grad_y, in_axes=(0, 0, None))(X_flat, Y_flat, params).reshape(Y.shape)

        return jnp.stack([-dF_dy, dF_dx], axis=0)
    return JAXskewgrad_FCN
    

@jax.jit
def JAXgradFinite(F, dx, dy):
    dF_dx = (jnp.roll(F, -1, axis=1) - jnp.roll(F, 1, axis=1)) / (2 * dx)
    dF_dy = (jnp.roll(F, -1, axis=0) - jnp.roll(F, 1, axis=0)) / (2 * dy)
    return jnp.array([dF_dx, dF_dy])

def grad(F, dx, dy):
    """Gradient operator"""
    dF_dx = np.gradient(F, dx, axis=1)
    dF_dy = np.gradient(F, dy, axis=0)

    return np.array([dF_dx, dF_dy])

@jax.jit
def JAXskew_gradFinite(F, dx, dy):
    dF_dx = (jnp.roll(F, -1, axis=1) - jnp.roll(F, 1, axis=1)) / (2 * dx)
    dF_dy = (jnp.roll(F, -1, axis=0) - jnp.roll(F, 1, axis=0)) / (2 * dy)
    return jnp.array([-dF_dy, dF_dx])

def skew_grad(F, dx, dy):
    """Skew gradient operator"""
    dF_dx = np.gradient(F, dx, axis=1)
    dF_dy = np.gradient(F, dy, axis=0)

    return np.array([-dF_dy, dF_dx])



def curl(F, dx ,dy):
    """2D Curl operator"""
    Fx, Fy = F

    dFx_dy = np.gradient(Fx, dy, axis=0)
    dFy_dx = np.gradient(Fy, dx, axis=1)
    curl = dFy_dx - dFx_dy

    return curl


def div(F, dx, dy):
    """Divergence operator"""
    Fx, Fy = F

    dFx_dx = np.gradient(Fx, dx, axis=1)
    dFy_dy = np.gradient(Fy, dy, axis=0)
    divergence = dFx_dx + dFy_dy

    return divergence


def norm(F):
    """Calculate the norm of vector"""
    Fx, Fy = F

    return np.sqrt(Fx**2 + Fy**2)

@jax.jit
def JAXnorm(F):
    """Calculate the norm of vector"""
    Fx, Fy = F

    return jnp.sqrt(Fx**2 + Fy**2)



def error(F1, F2):
    """Error between 2 vector fields"""

    # return the norm of the difference between the 2 vectors
    return norm(F1 - F2)


def JAXerror(F1, F2):
    """Error between 2 vector fields"""

    # return the norm of the difference between the 2 vectors
    return JAXnorm(F1 - F2)



def window_function(x,y):
    return  np.exp(-(x**2+y**2)/1) #np.exp(-(x**2+y**2)/0.04)


def helmholtz_decomp(X, Y, vector_field, fields=False, streamplots=False, error_display=False, potentials=False):
    """Does the helmholtz decomposition of a vector field

    Parameters
    ----------
    X :
        X coordinates
    Y :
        Y coordinates
    vector_field :
        The 2D vector field to decompose
    field : bool (default: "False")
        Show the original vector field and recomposed vector field
    streamplot : bool (default: "False")
        Show the original streamplot and recomposed streamplot
    error_display : bool (default: "False")
        Show the error on the recomposed vector field
    potentials : bool (default: "False")
        Show the curl free and divergence free potential found

    Returns
    -------
    phi
        Scalar curl free potential 
    psy
        Scalar divergence free potential
    vector_field_recomp
        The recomposed vector field from the the two potentials
    fig_error
        The error figure (-1 if none)
    fig_streamplots
        The streamplot figures (-1 if none)
    fig_fields
        The vector field figures (-1 if none)
    fig_potentials
        The potential figures (-1 if none)

    """

    dx = X[0, 1] - X[0, 0]  # change along x-axis (horizontal direction)
    dy = Y[1, 0] - Y[0, 0]  # change along y-axis (vertical direction)

    points = X.shape[1]

    def Green(x,y):
        """Green's function for 2D Laplace operator"""
        eps = 1e-10
        r = np.sqrt(x**2 + y**2) + eps

        return (1/(2*np.pi)) * np.log(r)
    

    def convolution(divF, rotF, dx, dy):
        """Convolution for the two functions  Div(F) * Green  &  Curl(F) * Green"""

        pad_x, pad_y = divF.shape 

        new_shape = (2 * pad_x, 2 * pad_y)      # new shape is 2 times as wide

        divF_padded = np.zeros(new_shape)       # we fill it up with zeroes
        rotF_padded = np.zeros(new_shape)

        divF_padded[pad_x//2:pad_x//2+pad_x, pad_y//2:pad_y//2+pad_y] = divF    # we center the function at the center
        rotF_padded[pad_x//2:pad_x//2+pad_x, pad_y//2:pad_y//2+pad_y] = rotF  

        # we create a big grid for Green's function
        ny, nx = new_shape
        x = np.arange(-nx//2, nx//2) * dx
        y = np.arange(-ny//2, ny//2) * dy
        big_x, big_y = np.meshgrid(x, y, indexing='xy')
        big_green = Green(big_x, big_y)

        G_shifted = np.fft.ifftshift(big_green) # we center the green function at x = 0, y = 0
        

        # FFT
        fft_divF = np.fft.fft2(divF_padded)
        fft_G = np.fft.fft2(G_shifted)
        fft_rotF = np.fft.fft2(rotF_padded)


        product_phi = fft_divF * fft_G
        product_psy = fft_rotF * fft_G


        # Inverse FFT
        result_phi = np.fft.ifft2(product_phi).real * dx * dy
        result_psy = np.fft.ifft2(product_psy).real * dx * dy


        # we retrieve the original grid
        result_phi = result_phi[pad_x//2:pad_x//2+pad_x, pad_y//2:pad_y//2+pad_y]
        result_psy = result_psy[pad_x//2:pad_x//2+pad_x, pad_y//2:pad_y//2+pad_y]

        return result_phi, result_psy






    # Filtering function if needed
    def filtering(F, eps):

        norm = norm(F)

        mask = norm < eps

        F[0][mask] = 0
        F[1][mask] = 0

            

    #print("Checkpoint 1 : Calculating Curl of Vector Field")

    rot_F = curl(vector_field, dx, dy)

    #print("Checkpoint 2 : Calculating Divergence of Vector Field")

    div_F = -1 * div(vector_field, dx, dy)

    #print("Checkpoint 3 : Convolution of Green and Divergence/Curl")

    phi, psy = convolution(div_F, rot_F, dx, dy)

    #print("Checkpoint 4 : Recomposing the Vector Field")

    gradient_potential = -1 * grad(phi, dx, dy)
    rotation_potential = skew_grad(psy, dx, dy)

    vector_field_recomp = rotation_potential + gradient_potential

    #filtering(champ_recomp)

    #print("Checkpoint 5 : Visualization !")


    Fx, Fy = vector_field
    Fx_recomp, Fy_recomp = vector_field_recomp
    fig_error = -1
    fig_streamplots = -1
    fig_fields = -1
    fig_potentials = -1

    if error_display:

        error_on_vect_field = error(vector_field, vector_field_recomp)
        print(round(np.max(error_on_vect_field), 2))

        fig_error = show_error_display(X, Y, error_on_vect_field, "Error on vector field")

    if streamplots:
        fig_streamplots =show_streamplots(X, Y, vector_field, vector_field_recomp, "Original streamplot", "Recomposed streamplot")

    if fields:
        fig_fields = show_fields(X, Y, vector_field, vector_field_recomp, "Original vector field", "Recomposed vector field")

    if potentials:
        fig_potentials = show_potentials(X, Y, phi, psy, "Phi recomposed", "Psy recomposed")

    #print("||F||:", np.linalg.norm(vector_field))
    #print("||F_recomp||:", np.linalg.norm(vector_field_recomp))
    #plt.show()

    return phi, psy, vector_field_recomp, fig_error, fig_streamplots, fig_fields, fig_potentials


