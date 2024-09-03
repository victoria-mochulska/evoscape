import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initialize the figure with an additional invisible axes for the buttons
fig = plt.figure(figsize=(12, 6))
ax_buttons = fig.add_axes([0.05, 0.25, 0.1, 0.6], frameon=False)
ax_left = fig.add_axes([0.2, 0.25, 0.35, 0.6])
ax_right = fig.add_axes([0.6, 0.25, 0.35, 0.6])

# Variables to store circles and their properties
circles = []
radii = []
alphas = []
colors = []
active_circle = None
dragging = False
addition_mode = None
delete_mode = False


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


# Event handler for mouse clicks
def on_click(event):
    global active_circle, addition_mode, delete_mode

    if event.inaxes == ax_buttons:  # Clicked on a button
        for i, button in enumerate(buttons):
            if button.contains_point((event.x, event.y)):
                addition_mode = button.get_facecolor()
                delete_mode = False
                return

        if delete_button.contains_point((event.x, event.y)):
            addition_mode = None
            delete_mode = True
            return

    elif event.inaxes == ax_left:  # Clicked on the left panel
        if addition_mode is not None:  # Add a new circle
            circle = plt.Circle((event.xdata, event.ydata), 0.05, color=addition_mode, alpha=0.5, picker=True)
            ax_left.add_patch(circle)
            circles.append(circle)
            radii.append(0.05)
            alphas.append(0.5)
            colors.append(addition_mode)
            active_circle = len(circles) - 1  # Set the new circle as active
            update_contour()
            addition_mode = None  # Reset addition mode after adding a circle
            radius_slider.set_val(radii[active_circle])
            alpha_slider.set_val(alphas[active_circle])

        elif delete_mode:  # Delete a circle
            for i, circle in enumerate(circles):
                if circle.contains_point((event.x, event.y)):
                    circle.remove()
                    del circles[i]
                    del radii[i]
                    del alphas[i]
                    del colors[i]
                    update_contour()
                    plt.draw()
                    delete_mode = False  # Reset delete mode after deleting a circle
                    return

        else:  # Select a circle
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
    if active_circle is not None and not addition_mode and not delete_mode:
        dragging = True


# Event handler for mouse scroll to change radius
def on_scroll(event):
    if active_circle is not None and event.inaxes == ax_left:
        change = 0.01 if event.button == 'up' else -0.01
        new_radius = radii[active_circle] + change
        if 0.01 <= new_radius <= 0.2:
            radii[active_circle] = new_radius
            circles[active_circle].set_radius(new_radius)
            update_contour()
            plt.draw()
            radius_slider.set_val(new_radius)


# Event handler for key press to change alpha
def on_key(event):
    if active_circle is not None:
        if event.key == 'up':
            new_alpha = min(alphas[active_circle] + 0.05, 1.0)
        elif event.key == 'down':
            new_alpha = max(alphas[active_circle] - 0.05, 0.1)
        else:
            return

        alphas[active_circle] = new_alpha
        circles[active_circle].set_alpha(new_alpha)
        update_contour()
        plt.draw()
        alpha_slider.set_val(new_alpha)


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
ax_radius = plt.axes([0.2, 0.15, 0.35, 0.03], facecolor='lightgoldenrodyellow')
ax_alpha = plt.axes([0.2, 0.1, 0.35, 0.03], facecolor='lightgoldenrodyellow')

radius_slider = Slider(ax_radius, 'Radius', 0.01, 0.2, valinit=0.05)
alpha_slider = Slider(ax_alpha, 'Alpha', 0.1, 1.0, valinit=0.5)

radius_slider.on_changed(update_radius)
alpha_slider.on_changed(update_alpha)

# Create color buttons in the ax_buttons
button_colors = ['green', 'blue', 'pink', 'purple']
buttons = []
button_height = 0.15
button_spacing = 0.05

for i, color in enumerate(button_colors):
    button_y = 0.85 - i * (button_height + button_spacing)
    button = plt.Circle((0.5, button_y), 0.07, color=color, transform=ax_buttons.transAxes, clip_on=False)
    ax_buttons.add_patch(button)
    buttons.append(button)

# Create delete button in the ax_buttons below the color buttons
delete_button_y = 0.85 - len(button_colors) * (button_height + button_spacing) - button_spacing
delete_button = plt.Rectangle((0.35, delete_button_y), 0.3, button_height, color='red', transform=ax_buttons.transAxes, clip_on=False)
ax_buttons.add_patch(delete_button)
ax_buttons.text(0.5, delete_button_y + button_height / 2, 'Delete', ha='center', va='center', fontsize=10, color='white')

# # Create color buttons in the ax_buttons
# button_colors = ['green', 'blue', 'pink', 'purple']
# buttons = []
# for i, color in enumerate(button_colors):
#     button = plt.Circle((0.5, 0.8 - i * 0.2), 0.1, color=color, transform=ax_buttons.transAxes, clip_on=False)
#     ax_buttons.add_patch(button)
#     buttons.append(button)
#
# # Create delete button in the ax_buttons
# delete_button = plt.Rectangle((0.3, 0.1), 0.4, 0.2, color='red', transform=ax_buttons.transAxes, clip_on=False)
# ax_buttons.add_patch(delete_button)
# ax_buttons.text(0.5, 0.2, 'Delete', ha='center', va='center', fontsize=10, color='white')

# Hide the ax_buttons axis
ax_buttons.set_xlim([0, 1])
ax_buttons.set_ylim([0, 1])
ax_buttons.axis('off')

# Connect event handlers
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial axis limits and plot setup
ax_left.set_xlim([-1, 1])
ax_left.set_ylim([-1, 1])
ax_left.set_aspect('equal', 'box')
ax_right.set_xlim([-1, 1])
ax_right.set_ylim([-1, 1])
ax_right.set_aspect('equal', 'box')
ax_buttons.set_aspect('equal')

plt.show()


