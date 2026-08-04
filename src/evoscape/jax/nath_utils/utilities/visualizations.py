import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation
from .module_gaussian import *
from matplotlib.patches import Ellipse
from pathlib import Path
from matplotlib.collections import LineCollection

def show_potentials(X, Y, phi, psy, title1, title2):
    """Show two windows of potentials. The potentials usually are the gradient and rotational potential.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    phi, psy : Gradient potential and rotational potential
    title1, title2 : Titles of the 2 plot

    Returns
    -------
    `fig` object of the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf_div = ax1.plot_surface(X, Y, phi, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax1.set_title(title1)
    ax1.contour(X, Y, phi, levels=7, alpha=0)
    
    
    surf_rot = ax2.plot_surface(X, Y, -psy, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax2.set_title(title2)
    ax1.contour(X, Y, phi, levels=7, alpha=0)

    return fig




def show_animated_potentials(X, Y, params1, params2, title1, title2, filename, interval=50):
    """Show one potential. The potential usually is the gradient or rotational potential.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    pot : Gradient potential or rotational potential
    title : Title of the plot

    Returns
    -------
    `fig` object of the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})

    def update(frame):
        fit_grad = JAXgaussian_mixture(X, Y, params1[frame])
        fit_rot = -JAXgaussian_mixture(X, Y, params2[frame])
        draw_potentials(ax1, ax2, X, Y, fit_grad, fit_rot, title1, title2)
        fig.suptitle("Time = " + str(frame))
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=params1.shape[0], interval=interval)

    writer = PillowWriter(fps=1000 // interval)
    ani.save(filename, writer=writer)

    plt.close(fig)

    return ani



def show_potential(X, Y, pot, title, save=False, filename=None):
    """Show one potential. The potential usually is the gradient or rotational potential.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    pot : Gradient potential or rotational potential
    title : Title of the plot

    Returns
    -------
    `fig` object of the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf_div = ax.plot_surface(X, Y, pot, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_title(title)
    ax.contour(X, Y, pot, levels=7, alpha=0)

    if save:
        plt.savefig(filename)

    return fig



def draw_potential(ax, X, Y, pot, title):
    ax.clear()
    ax.plot_surface(X, Y, pot, cmap=cm.coolwarm)
    ax.set_title(title)
    
def draw_potentials(ax1, ax2, X, Y, pot1, pot2, title1, title2):
    ax1.clear()
    ax1.plot_surface(X, Y, pot1, cmap=cm.coolwarm)
    ax1.set_title(title1)

    ax2.clear()
    ax2.plot_surface(X, Y, pot2, cmap=cm.coolwarm)
    ax2.set_title(title2)

def show_animated_potential(X, Y, params, title, filename, interval=50, neg=False):
    """Show one potential. The potential usually is the gradient or rotational potential.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    params : Parameters of the potentials
    title : Title of the plot

    Returns
    -------
    `ani` object of the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": "3d"})

    def update(frame):
        fit = JAXgaussian_mixture(X, Y, params[frame])
        if neg:
            draw_potential(ax, X, Y, -fit, title + " | Time = " + str(frame))
        else:
            draw_potential(ax, X, Y, fit, title + " | Time = " + str(frame))
        fig.suptitle("Time = " + str(frame))

    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=params.shape[0], interval=interval)

    writer = PillowWriter(fps=1000 // interval)
    ani.save(filename, writer=writer)

    plt.close(fig)

    return ani





def show_fields(X, Y, vector_field1, vector_field2, title1, title2, nbr_arrows=20):
    """Show two windows of vector field. The two fields usually are the gradient and rotational vector fields.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field1, vector_field2 : Gradient field and rotational field
    title1, title2 : Titles of the 2 plot
    nbr_arrows : Density of the vector fields

    Returns
    -------
    `fig` object of the plot
    """
    points = X.shape[1]

    Fx1, Fy1 = vector_field1


    step = points // nbr_arrows  # Number of arrows 

    X_s = X[::step, ::step]
    Y_s = Y[::step, ::step]



    Fx1_s = Fx1[::step, ::step] 
    Fy1_s = Fy1[::step, ::step]
    norm1 = np.sqrt(Fx1_s**2 + Fy1_s**2) + 1e-12
    """Fx1_s = Fx1_s / norm1
    Fy1_s = Fy1_s / norm1"""

    

    Fx2, Fy2 = vector_field2

    Fx2_s = Fx2[::step, ::step] 
    Fy2_s = Fy2[::step, ::step]
    norm2 = np.sqrt(Fx2_s**2 + Fy2_s**2) + 1e-12
    """Fx2_s = Fx2_s / norm2
    Fy2_s = Fy2_s / norm2"""
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.quiver(X_s, Y_s, Fx1_s, Fy1_s, norm1, cmap='coolwarm', lw=2, width=0.005, headwidth=3, alpha=1)
    #ax1.quiver(X_s, Y_s, Fx_s, Fy_s, color='blue', lw=2, width=0.005, headwidth=3, alpha=1)

    ax1.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax1.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(title1)
    ax1.set_aspect('equal')
    ax1.grid(True)


    ax2.quiver(X_s, Y_s, Fx2_s, Fy2_s, norm2, cmap='coolwarm', lw=2, width=0.005, headwidth=3, alpha=1)
    #ax2.quiver(X_s, Y_s, Fx_r, Fy_r, color='orange', lw=2, width=0.005, headwidth=3, alpha=1)

    ax2.contour(X, Y, Fx2, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax2.contour(X, Y, Fy2, levels=(0,), colors='black', linewidths=2)

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(title2)
    ax2.set_aspect('equal')
    ax2.grid(True)

    return fig


def show_fields_with_modules(X, Y, vector_field1, vector_field2, params1, params2, title1, title2, nbr_arrows=20):
    """Show two windows of vector field. The two fields usually are the gradient and rotational vector fields.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field1, vector_field2 : Gradient field and rotational field
    title1, title2 : Titles of the 2 plot
    nbr_arrows : Density of the vector fields

    Returns
    -------
    `fig` object of the plot
    """
    points = X.shape[1]

    Fx1, Fy1 = vector_field1


    step = points // nbr_arrows  # Number of arrows 

    X_s = X[::step, ::step]
    Y_s = Y[::step, ::step]



    Fx1_s = Fx1[::step, ::step] 
    Fy1_s = Fy1[::step, ::step]
    norm1 = np.sqrt(Fx1_s**2 + Fy1_s**2) + 1e-12
    """Fx1_s = Fx1_s / norm1
    Fy1_s = Fy1_s / norm1"""

    

    Fx2, Fy2 = vector_field2

    Fx2_s = Fx2[::step, ::step] 
    Fy2_s = Fy2[::step, ::step]
    norm2 = np.sqrt(Fx2_s**2 + Fy2_s**2) + 1e-12
    """Fx2_s = Fx2_s / norm2
    Fy2_s = Fy2_s / norm2"""
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.quiver(X_s, Y_s, Fx1_s, Fy1_s, norm1, cmap='coolwarm', lw=2, width=0.005, headwidth=3, alpha=1)
    #ax1.quiver(X_s, Y_s, Fx_s, Fy_s, color='blue', lw=2, width=0.005, headwidth=3, alpha=1)

    ax1.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax1.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)
    
    amp_max = np.max(np.abs(params1[:, 2]))*1.5
    for module in params1:
        ellipse = create_module_patch(module, amp_max)
        ax1.add_patch(ellipse)
    
    ax1.set_xlim(X.min(), X.max())
    ax1.set_ylim(Y.min(), Y.max())
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(title1)
    ax1.set_aspect('equal')
    ax1.grid(True)


    ax2.quiver(X_s, Y_s, Fx2_s, Fy2_s, norm2, cmap='coolwarm', lw=2, width=0.005, headwidth=3, alpha=1)
    #ax2.quiver(X_s, Y_s, Fx_r, Fy_r, color='orange', lw=2, width=0.005, headwidth=3, alpha=1)

    ax2.contour(X, Y, Fx2, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax2.contour(X, Y, Fy2, levels=(0,), colors='black', linewidths=2)

    amp_max = np.max(np.abs(params2[:, 2]))*1.5
    for module in params2:
        ellipse = create_module_patch(module, amp_max)
        ax2.add_patch(ellipse)
    
    ax2.set_xlim(X.min(), X.max())
    ax2.set_ylim(Y.min(), Y.max())
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(title2)
    ax2.set_aspect('equal')
    ax2.grid(True)

    return fig




def show_field(X, Y, vector_field, title, nbr_arrows=20, nullclines=True):
    """Show one vector field. The vector field usually is the gradient or rotational field.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field : Gradient field or rotational field
    title : Title of the plot
    nbr_arrows : Density of the vector field

    Returns
    -------
    `fig` object of the plot
    """
    points = X.shape[1]

    Fx1, Fy1 = vector_field


    step = points // nbr_arrows  # Number of arrows 

    X_s = X[::step, ::step]
    Y_s = Y[::step, ::step]


    Fx1_s = Fx1[::step, ::step] 
    Fy1_s = Fy1[::step, ::step]
    norm1 = np.sqrt(Fx1_s**2 + Fy1_s**2) + 1e-12
    """Fx1_s = Fx1_s / norm1
    Fy1_s = Fy1_s / norm1"""

    fig = plt.figure(figsize=(6, 6))

    plt.quiver(X_s, Y_s, Fx1_s, Fy1_s, norm1, cmap='coolwarm', lw=2, width=0.005, headwidth=3, alpha=1)
    #ax1.quiver(X_s, Y_s, Fx_s, Fy_s, color='blue', lw=2, width=0.005, headwidth=3, alpha=1)

    if nullclines:
        plt.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
        plt.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    
    return fig



def draw_field(ax, X, Y, vector_field, title, step, nullclines):

    Fx1, Fy1 = vector_field

    X_s = X[::step, ::step]
    Y_s = Y[::step, ::step]

    Fx1_s = Fx1[::step, ::step] 
    Fy1_s = Fy1[::step, ::step]
    norm1 = np.sqrt(Fx1_s**2 + Fy1_s**2) + 1e-12

    ax.clear()
    ax.quiver(X_s, Y_s, Fx1_s, Fy1_s, norm1, cmap='coolwarm', lw=2, width=0.005, headwidth=3, alpha=1)
    if nullclines:
        ax.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
        ax.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.grid(True)


def draw_fields(ax1, ax2, X, Y, vector_field1, vector_field2, title1, title2, step, nullclines):

    X_s = X[::step, ::step]
    Y_s = Y[::step, ::step]

    Fx1, Fy1 = vector_field1
    Fx1_s = Fx1[::step, ::step] 
    Fy1_s = Fy1[::step, ::step]
    norm1 = np.sqrt(Fx1_s**2 + Fy1_s**2) + 1e-12



    Fx2, Fy2 = vector_field2
    Fx2_s = Fx2[::step, ::step] 
    Fy2_s = Fy2[::step, ::step]
    norm2 = np.sqrt(Fx2_s**2 + Fy2_s**2) + 1e-12

    ax1.clear()
    ax1.quiver(X_s, Y_s, Fx1_s, Fy1_s, norm1, cmap='coolwarm', lw=2, width=0.005, headwidth=3, alpha=1)
    ax1.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax1.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(title1)
    ax1.set_aspect('equal')
    ax1.grid(True)

    ax2.clear()
    ax2.quiver(X_s, Y_s, Fx2_s, Fy2_s, norm2, cmap='coolwarm', lw=2, width=0.005, headwidth=3, alpha=1)
    ax2.contour(X, Y, Fx2, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax2.contour(X, Y, Fy2, levels=(0,), colors='black', linewidths=2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(title2)
    ax2.set_aspect('equal')
    ax2.grid(True)


def show_animated_field(X, Y, params_grad, params_rot, title, filename, interval=50, nbr_arrows=20, nullclines=True):
    """Show one vector field. The vector field usually is the gradient or rotational field.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field : Gradient field or rotational field
    title : Title of the plot
    nbr_arrows : Density of the vector field

    Returns
    -------
    `fig` object of the plot
    """
    points = X.shape[1]

    step = points // nbr_arrows  # Number of arrows 

    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame):
        vector_field = JAXflow_grad_mixture(X, Y, params_grad[frame]) + JAXflow_rot_mixture(X, Y, params_rot[frame])
        draw_field(ax, X, Y, vector_field, title, step, nullclines)
        fig.suptitle("Time = " + str(frame))
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=params_grad.shape[0], interval=interval)

    writer = PillowWriter(fps=1000 // interval)
    ani.save(filename, writer=writer)

    plt.close(fig)
    
    return ani


def show_animated_fields(X, Y, params_grad, params_rot, title1, title2, filename, interval=50, nbr_arrows=20, nullclines=True):
    """Show one vector field. The vector field usually is the gradient or rotational field.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field : Gradient field or rotational field
    title : Title of the plot
    nbr_arrows : Density of the vector field

    Returns
    -------
    `fig` object of the plot
    """
    points = X.shape[1]

    step = points // nbr_arrows  # Number of arrows 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def update(frame):
        vector_field_grad = JAXflow_grad_mixture(X, Y, params_grad[frame])
        vector_field_rot = JAXflow_rot_mixture(X, Y, params_rot[frame])
        draw_fields(ax1, ax2, X, Y, vector_field_grad, vector_field_rot, title1, title2, step, nullclines)
        fig.suptitle("Time = " + str(frame))
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=params_grad.shape[0], interval=interval)

    writer = PillowWriter(fps=1000 // interval)
    ani.save(filename, writer=writer)

    plt.close(fig)
    
    return ani




def create_module_patch(params, amp_max):
    if params[2] > 0:
        color = "blue"
    else:
        color = "red"
    factor = 2 * np.sqrt(2 * np.log(10))
    ellipse = Ellipse(xy=(params[0], params[1]), width=factor * params[3], height=factor * params[4], angle=np.degrees(params[5]), fill=True, fc=color, alpha=np.abs(params[2]/amp_max), ec=None)
    return ellipse


def show_field_with_modules(X, Y, vector_field, params, title, nbr_arrows=20, nullclines=True):
    """Show one vector field. The vector field usually is the gradient or rotational field.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field : Gradient field or rotational field
    title : Title of the plot
    nbr_arrows : Density of the vector field

    Returns
    -------
    `fig` object of the plot
    """
    points = X.shape[1]

    Fx1, Fy1 = vector_field


    step = points // nbr_arrows  # Number of arrows 

    X_s = X[::step, ::step]
    Y_s = Y[::step, ::step]


    Fx1_s = Fx1[::step, ::step] 
    Fy1_s = Fy1[::step, ::step]
    norm1 = np.sqrt(Fx1_s**2 + Fy1_s**2) + 1e-12
    """Fx1_s = Fx1_s / norm1
    Fy1_s = Fy1_s / norm1"""

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.quiver(X_s, Y_s, Fx1_s, Fy1_s, norm1, cmap='coolwarm', lw=2, width=0.005, headwidth=3, alpha=1)
    #ax1.quiver(X_s, Y_s, Fx_s, Fy_s, color='blue', lw=2, width=0.005, headwidth=3, alpha=1)

    if nullclines:
        plt.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
        plt.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)

    amp_max = np.max(np.abs(params[:, 2]))*1.5
    for module in params:
        ellipse = create_module_patch(module, amp_max)
        ax.add_patch(ellipse)
    
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    
    return fig


def show_streamplots(X, Y, vector_field1, vector_field2, title1, title2):
    """Show two windows of streamplots. The two streamplots usually are the gradient and rotational streamplots.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field1, vector_field2 : Gradient field and rotational field
    title1, title2 : Titles of the 2 plot

    Returns
    -------
    `fig` object of the plot
    """
    Fx1, Fy1 = vector_field1
    Fx2, Fy2 = vector_field2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.streamplot(X, Y, Fx1, Fy1, color= "blue", density = 3)

    ax1.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax1.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(title1)
    ax1.set_aspect('equal')
    ax1.grid(True)

    ax2.streamplot(X, Y, Fx2, Fy2, color= "orange", density = 3)
    ax2.contour(X, Y, Fx2, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax2.contour(X, Y, Fy2, levels=(0,), colors='black', linewidths=2)

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(title2)
    ax2.set_aspect('equal')
    ax2.grid(True)

    return fig


def show_streamplots_with_modules(X, Y, vector_field1, vector_field2, params1, params2, title1, title2):
    """Show two windows of streamplots. The two streamplots usually are the gradient and rotational streamplots.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field1, vector_field2 : Gradient field and rotational field
    title1, title2 : Titles of the 2 plot

    Returns
    -------
    `fig` object of the plot
    """
    Fx1, Fy1 = vector_field1
    Fx2, Fy2 = vector_field2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.streamplot(X, Y, Fx1, Fy1, color= "blue", density = 3)

    ax1.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax1.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)

    amp_max = np.max(np.abs(params1[:, 2]))*1.5
    for module in params1:
        ellipse = create_module_patch(module, amp_max)
        ax1.add_patch(ellipse)
    
    ax1.set_xlim(X.min(), X.max())
    ax1.set_ylim(Y.min(), Y.max())
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(title1)
    ax1.set_aspect('equal')
    ax1.grid(True)

    ax2.streamplot(X, Y, Fx2, Fy2, color= "orange", density = 3)
    ax2.contour(X, Y, Fx2, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
    ax2.contour(X, Y, Fy2, levels=(0,), colors='black', linewidths=2)

    amp_max = np.max(np.abs(params2[:, 2]))*1.5
    for module in params2:
        ellipse = create_module_patch(module, amp_max)
        ax2.add_patch(ellipse)
    
    ax2.set_xlim(X.min(), X.max())
    ax2.set_ylim(Y.min(), Y.max())
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(title2)
    ax2.set_aspect('equal')
    ax2.grid(True)

    return fig



def show_streamplot(X, Y, vector_field, title, nullclines=True):
    """Show one streamplot. The streamplot usually is the gradient or rotational field.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field : Gradient field or rotational field
    title : Title of the plot
    nullclines : True of False show the nullclines

    Returns
    -------
    `fig` object of the plot
    """
    Fx, Fy = vector_field
    
    fig = plt.figure(figsize=(6, 6))

    plt.streamplot(X, Y, Fx, Fy, color= "blue", density = 3)
    if nullclines:
        plt.contour(X, Y, Fx, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
        plt.contour(X, Y, Fy, levels=(0,), colors='black', linewidths=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)

    return fig







def draw_streamplot(ax, X, Y, vector_field, title, nullclines):

    Fx1, Fy1 = vector_field

    ax.clear()
    ax.streamplot(X, Y, Fx1, Fy1, color= "blue", density = 3)
    if nullclines:
        ax.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
        ax.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.grid(True)


def draw_streamplots(ax1, ax2, X, Y, vector_field1, vector_field2, title1, title2, nullclines):

    Fx1, Fy1 = vector_field1

    ax1.clear()
    ax1.streamplot(X, Y, Fx1, Fy1, color= "blue", density = 3)
    if nullclines:
        ax1.contour(X, Y, Fx1, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
        ax1.contour(X, Y, Fy1, levels=(0,), colors='black', linewidths=2)
    ax1.set_title(title1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect('equal')
    ax1.grid(True)


    Fx2, Fy2 = vector_field2

    ax2.clear()
    ax2.streamplot(X, Y, Fx2, Fy2, color= "blue", density = 3)
    if nullclines:
        ax2.contour(X, Y, Fx2, levels=(0,), colors='dimgray', linestyles="dashed", linewidths=2)
        ax2.contour(X, Y, Fy2, levels=(0,), colors='black', linewidths=2)
    ax2.set_title(title2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect('equal')
    ax2.grid(True)



def show_animated_streamplot(X, Y, params_grad, params_rot, title, filename, interval=50, nullclines=True):
    """Show one vector field. The vector field usually is the gradient or rotational field.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field : Gradient field or rotational field
    title : Title of the plot
    nbr_arrows : Density of the vector field

    Returns
    -------
    `fig` object of the plot
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame):
        vector_field = JAXflow_grad_mixture(X, Y, params_grad[frame]) + JAXflow_rot_mixture(X, Y, params_rot[frame])
        draw_streamplot(ax, X, Y, vector_field, title, nullclines)
        fig.suptitle("Time = " + str(frame))
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=params_grad.shape[0], interval=interval)

    writer = PillowWriter(fps=1000 // interval)
    ani.save(filename, writer=writer)

    plt.close(fig)
    
    return ani


def show_animated_streamplots(X, Y, params_grad, params_rot, title1, title2, filename, interval=50, nullclines=True):
    """Show one vector field. The vector field usually is the gradient or rotational field.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    vector_field : Gradient field or rotational field
    title : Title of the plot
    nbr_arrows : Density of the vector field

    Returns
    -------
    `fig` object of the plot
    """


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def update(frame):
        vector_field_grad = JAXflow_grad_mixture(X, Y, params_grad[frame])
        vector_field_rot = JAXflow_rot_mixture(X, Y, params_rot[frame])
        draw_streamplots(ax1, ax2, X, Y, vector_field_grad, vector_field_rot, title1, title2, nullclines)
        fig.suptitle("Time = " + str(frame))
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=params_grad.shape[0], interval=interval)

    writer = PillowWriter(fps=1000 // interval)
    ani.save(filename, writer=writer)

    plt.close(fig)
    
    return ani










def show_error_display(X, Y, error_on_vect_field, title):
    """Show an error. We have a scalar field telling the error on a 2D vector field.

    Parameters
    ----------
    X, Y : Coordinates of meshgrid
    error_on_vect_field : Scalar error
    title : Title of the plot

    Returns
    -------
    `fig` object of the plot
    """
    fig = plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, error_on_vect_field, shading='auto', cmap='inferno')
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')

    return fig




def show_trajectory2(trajectory1, trajectory2, filename):
    """Show the animation of one trajectory.

    Parameters
    ----------
    trajectores : The trajectores as an array
    filename : Where we want to save and the filename

    Returns
    -------
    `ani` object of the plot (the animation)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    n_time = trajectory1.shape[1]   # 162
    n_space = trajectory1.shape[2]  # 558


    # Axis limits
    minimum_x = min(np.min(trajectory1[0]), np.min(trajectory2[0]))
    maximum_x = max(np.max(trajectory1[0]), np.max(trajectory2[0]))

    minimum_y = min(np.min(trajectory1[1]), np.min(trajectory2[1]))
    maximum_y = max(np.max(trajectory1[1]), np.max(trajectory2[1]))


    for ax in (ax1, ax2):
        ax.set_xlim(minimum_x, maximum_x)
        ax.set_ylim(minimum_y, maximum_y)
        ax.set_aspect("equal")


    # Create 558 temporal trajectories
    lines1 = []
    lines2 = []
    points1 = []
    points2 = []

    for i in range(n_space):
        line1, = ax1.plot([], [], 'b-', lw=2)
        point1, = ax1.plot([], [], 'ro')
        line2, = ax2.plot([], [], 'b-', lw=2)
        point2, = ax2.plot([], [], 'ro')

        lines1.append(line1)
        lines2.append(line2)
        points1.append(point1)
        points2.append(point2)



    def init():

        for line in lines1:
            line.set_data([], [])

        for line in lines2:
            line.set_data([], [])

        for point in points1:
            point.set_data([], [])

        for point in points2:
            point.set_data([], [])

        return lines1 + lines2 + points1 + points2


    def update(frame):

        for i in range(n_space):

            # Temporal trajectory of spatial point i
            x1 = trajectory1[0, :frame+1, i]
            y1 = trajectory1[1, :frame+1, i]

            x2 = trajectory2[0, :frame+1, i]
            y2 = trajectory2[1, :frame+1, i]


            lines1[i].set_data(x1, y1)
            lines2[i].set_data(x2, y2)

            points1[i].set_data(
                [trajectory1[0, frame, i]],
                [trajectory1[1, frame, i]]
            )

            points2[i].set_data(
                [trajectory2[0, frame, i]],
                [trajectory2[1, frame, i]]
            )


        return lines1 + lines2 + points1 + points2


    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_time,
        init_func=init,
        blit=True,
        interval=20
    )


    writer = PillowWriter(fps=1000 // 20)
    ani.save(filename, writer=writer)

    plt.close(fig)
    return ani 






def show_trajectory(
    trajectory1,
    trajectory2,
    title,
    filename,
    fps=50,
    dpi=100,
    linewidth=1.0,
    point_size=16,
    frame_step=1,
):
    """
    Anime deux ensembles de trajectoires.

    Parameters
    ----------
    trajectory1, trajectory2 : array-like, shape (2, n_time, n_space)
        trajectory[0] contient les coordonnées x.
        trajectory[1] contient les coordonnées y.

    filename : str or Path
        Fichier de sortie, généralement ".gif" ou ".mp4".

    fps : int, optional
        Nombre d'images affichées par seconde.

    dpi : int, optional
        Résolution de l'animation sauvegardée.

    linewidth : float, optional
        Épaisseur des trajectoires.

    point_size : float, optional
        Taille des points courants.

    frame_step : int, optional
        Utilise une image temporelle sur `frame_step`.
        Par exemple, frame_step=2 divise presque par deux le nombre
        d'images à calculer et encoder.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation générée.
    """

    # Conversion une seule fois vers NumPy.
    # Important si les entrées sont des tableaux JAX.
    trajectory1 = np.asarray(trajectory1)
    trajectory2 = np.asarray(trajectory2)

    if trajectory1.ndim != 3 or trajectory1.shape[0] != 2:
        raise ValueError(
            "trajectory1 doit avoir la forme (2, n_time, n_space). "
            f"Forme reçue : {trajectory1.shape}"
        )

    if trajectory2.shape != trajectory1.shape:
        raise ValueError(
            "trajectory1 et trajectory2 doivent avoir la même forme. "
            f"Formes reçues : {trajectory1.shape} et {trajectory2.shape}"
        )

    if frame_step < 1:
        raise ValueError("frame_step doit être supérieur ou égal à 1.")

    n_time = trajectory1.shape[1]

    # Passage de (2, n_time, n_space) à (n_time, n_space, 2).
    # Ainsi, coords[frame] contient directement tous les points au temps frame.
    coords1 = np.ascontiguousarray(np.moveaxis(trajectory1, 0, -1))
    coords2 = np.ascontiguousarray(np.moveaxis(trajectory2, 0, -1))

    # ------------------------------------------------------------------
    # Limites communes aux deux graphiques
    # ------------------------------------------------------------------

    minimum_x = min(
        np.nanmin(coords1[..., 0]),
        np.nanmin(coords2[..., 0]),
    )
    maximum_x = max(
        np.nanmax(coords1[..., 0]),
        np.nanmax(coords2[..., 0]),
    )

    minimum_y = min(
        np.nanmin(coords1[..., 1]),
        np.nanmin(coords2[..., 1]),
    )
    maximum_y = max(
        np.nanmax(coords1[..., 1]),
        np.nanmax(coords2[..., 1]),
    )

    # Petite marge pour éviter que les trajectoires touchent les axes.
    delta_x = maximum_x - minimum_x
    delta_y = maximum_y - minimum_y

    margin_x = 0.02 * delta_x if delta_x > 0 else 1.0
    margin_y = 0.02 * delta_y if delta_y > 0 else 1.0

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16)


    for ax in (ax1, ax2):
        ax.set_xlim(minimum_x - margin_x, maximum_x + margin_x)
        ax.set_ylim(minimum_y - margin_y, maximum_y + margin_y)
        ax.set_aspect("equal", adjustable="box")

    # ------------------------------------------------------------------
    # Quatre objets graphiques seulement
    # ------------------------------------------------------------------

    lines1 = LineCollection(
        [],
        colors="tab:blue",
        linewidths=linewidth,
        antialiaseds=False,
        zorder=2
    )

    lines2 = LineCollection(
        [],
        colors="tab:blue",
        linewidths=linewidth,
        antialiaseds=False,
        zorder=2
    )

    ax1.add_collection(lines1)
    ax2.add_collection(lines2)

    points1 = ax1.scatter(
        [],
        [],
        s=point_size,
        c="tab:red",
        edgecolors="none",
        zorder=3
    )

    points2 = ax2.scatter(
        [],
        [],
        s=point_size,
        c="tab:red",
        edgecolors="none",
        zorder=3
    )

    artists = (lines1, lines2, points1, points2)

    empty_offsets = np.empty((0, 2))

    def init():
        lines1.set_segments([])
        lines2.set_segments([])

        points1.set_offsets(empty_offsets)
        points2.set_offsets(empty_offsets)

        return artists

    def update(frame):
        # coords[:frame + 1] :
        #     (temps, espace, xy)
        #
        # swapaxes(0, 1) :
        #     (espace, temps, xy)
        #
        # Chaque élément correspond alors à une trajectoire spatiale.
        segments1 = coords1[:frame + 1].swapaxes(0, 1)
        segments2 = coords2[:frame + 1].swapaxes(0, 1)

        lines1.set_segments(segments1)
        lines2.set_segments(segments2)

        # Tous les points sont mis à jour en une seule opération.
        points1.set_offsets(coords1[frame])
        points2.set_offsets(coords2[frame])

        return artists

    frame_indices = np.arange(0, n_time, frame_step)

    # S'assurer que la dernière image temporelle est incluse.
    if frame_indices[-1] != n_time - 1:
        frame_indices = np.append(frame_indices, n_time - 1)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        blit=True,
        interval=1000 / fps,
        repeat=False,
        cache_frame_data=False,
    )

    filename = Path(filename)
    extension = filename.suffix.lower()

    if extension == ".gif":
        writer = animation.PillowWriter(fps=fps)

    elif extension in {".mp4", ".m4v", ".mov"}:
        writer = animation.FFMpegWriter(fps=fps)

    else:
        plt.close(fig)
        raise ValueError(
            "Extension non prise en charge. Utiliser notamment '.gif' ou '.mp4'."
        )

    ani.save(filename, writer=writer, dpi=dpi)

    plt.close(fig)

    return ani





def show_trajectories(trajec1, trajec2, title=None, dpi=350, save=False, filename=None):
    """Show two trajectories side by side.

    Parameters
    ----------
    trajectories : The trajectories as arrays
    dpi : The quality of the plot

    Returns
    -------
    `fit` object of the plot
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
    fig.suptitle(title, fontsize=16)

    # Configuration des axes (optionnel mais utile)
    ax1.set_title("Trajectoire 1")
    ax2.set_title("Trajectoire 2")
    minimum_x = np.min([np.min(trajec1[0,:,:]), np.min(trajec2[0,:,:])])
    maximum_x = np.max([np.max(trajec1[0,:,:]), np.max(trajec2[0,:,:])])
    minimum_y = np.min([np.min(trajec1[1,:,:]), np.min(trajec2[1,:,:])])
    maximum_y = np.max([np.max(trajec1[1,:,:]), np.max(trajec2[1,:,:])])
    ax1.set_xlim(minimum_x, maximum_x)
    ax1.set_ylim(minimum_y, maximum_y)
    ax2.set_xlim(minimum_x, maximum_x)
    ax2.set_ylim(minimum_y, maximum_y)

    ax1.plot(trajec1[0,:,:], trajec1[1,:,:], color="black")
    ax2.plot(trajec2[0,:,:], trajec2[1,:,:], color="blue")
    if save:
        plt.savefig(filename)
    return fig



def plot_parameters_over_time(params, title, dpi=350):
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=dpi)
    t = np.arange(params.shape[0])
    fig.suptitle("Parameters evolution" + " " + title, fontsize=16)

    # X0 and Y0
    axes[0, 0].plot(t, params[:, 0], label="X0", color="black")
    axes[0, 0].plot(t, params[:, 1], label="Y0", color="green")
    axes[0, 0].legend(loc='upper left')

    # sx and sy
    axes[1, 0].plot(t, params[:, 3], label="Sigma x", color="black")
    axes[1, 0].plot(t, params[:, 4], label="Sigma y", color="green")
    axes[1, 0].legend(loc='upper left')

    # amp
    axes[0, 1].plot(t, params[:, 2], label="Amplitude", color="orange")
    axes[0, 1].legend(loc='upper left')

    # theta
    axes[1, 1].plot(t, params[:, 5], label="Theta", color="red")
    axes[1, 1].legend(loc='upper left')

    plt.tight_layout()

    return fig