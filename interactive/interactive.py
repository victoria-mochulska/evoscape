import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initialize the figure and two subplots (left and right panels)
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Variables to store circles and their properties
circles = []
radii = []
alphas = []
active_circle = None
dragging = False


# Function to update the contour plot
def update_contour():
    ax_right.clear()
    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    Z = np.zeros_like(X)

    for i, circle in enumerate(circles):
        circle_x, circle_y = circle.center
        Z += alphas[i] * np.exp(-((X - circle_x) ** 2 + (Y - circle_y) ** 2) / (2 * radii[i] ** 2))

    ax_right.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax_right.set_xlim([-1, 1])
    ax_right.set_ylim([-1, 1])
    plt.draw()


# Event handler for mouse double-click to add a circle
def on_double_click(event):
    global circles, radii, alphas
    if event.inaxes == ax_left:
        circle = plt.Circle((event.xdata, event.ydata), 0.05, color='blue', alpha=0.5, picker=True)
        ax_left.add_patch(circle)
        circles.append(circle)
        radii.append(0.05)
        alphas.append(0.5)
        update_contour()


# Event handler for mouse click to select a circle
def on_click(event):
    global active_circle
    if event.inaxes == ax_left:
        for i, circle in enumerate(circles):
            if circle.contains_point((event.x, event.y)):
                active_circle = i
                radius_slider.set_val(radii[i])
                alpha_slider.set_val(alphas[i])
                return
        active_circle = None


# Event handler for mouse motion to drag a circle
def on_motion(event):
    global dragging
    if active_circle is not None and event.inaxes == ax_left and dragging:
        circles[active_circle].center = (event.xdata, event.ydata)
        update_contour()
        plt.draw()


# Event handler for mouse button release to stop dragging
def on_release(event):
    global dragging
    dragging = False


# Event handler for mouse button press to start dragging
def on_press(event):
    global dragging
    if active_circle is not None:
        dragging = True


# Slider update functions
def update_radius(val):
    if active_circle is not None:
        radii[active_circle] = val
        circles[active_circle].set_radius(val)
        update_contour()
        plt.draw()


def update_alpha(val):
    if active_circle is not None:
        alphas[active_circle] = val
        circles[active_circle].set_alpha(val)
        update_contour()
        plt.draw()


# Set up sliders for radius and alpha
ax_radius = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_alpha = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

radius_slider = Slider(ax_radius, 'Radius', 0.01, 0.2, valinit=0.05)
alpha_slider = Slider(ax_alpha, 'Alpha', 0.1, 1.0, valinit=0.5)

radius_slider.on_changed(update_radius)
alpha_slider.on_changed(update_alpha)

# Connect event handlers
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_press_event', on_double_click)

# Initial axis limits and plot setup
ax_left.set_xlim([-1, 1])
ax_left.set_ylim([-1, 1])
ax_left.set_aspect('equal', 'box')
ax_right.set_xlim([-1, 1])
ax_right.set_ylim([-1, 1])
ax_right.set_aspect('equal', 'box')

plt.show()