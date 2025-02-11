import numpy as np
import matplotlib.pyplot as plt
# import time
import cmcrameri.cm as cm
from matplotlib.widgets import Slider

fig = plt.figure(figsize=(18, 6))
ax_buttons = fig.add_axes([-0.1, 0.25, 0.2, 0.5], frameon=False)  # 0.08, 0.6
ax_left = fig.add_axes([0.12, 0.25, 0.2, 0.5])
# ax_right = fig.add_axes([0.4, 0.25, 0.25, 0.6])
ax_flow = fig.add_axes([0.35, 0.25, 0.2, 0.5])  # Gradient potential
ax_gradient = fig.add_axes([0.55, 0.25, 0.2, 0.5])  # Rotational potential
ax_rotational = fig.add_axes([0.75, 0.25, 0.2, 0.5])  # Flow field


ax_gradient.set_title('Gradient Potential')
ax_rotational.set_title('Rotational Potential')
ax_flow.set_title('Flow Field')
for ax in [ax_gradient, ax_rotational, ax_flow]:
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

color_amplitude = {
    'tab:green': -1.0,
    'hotpink': 1.0,
    'tab:blue': 1.0,
    'tab:purple': -1.0
}

circles = []
radii = []
alphas = []
colors = []
active_circle = None
dragging = False
addition_mode = None
delete_mode = False
npoints = 101
X, Y = np.meshgrid(np.linspace(-1, 1, npoints), np.linspace(-1, 1, npoints))

gradient_contributions = []
rotational_contributions = []
flow_contributions_x = []
flow_contributions_y = []

def update_gradient_potential():
    ax_gradient.clear()
    ax_gradient.set_title('Gradient Potential')
    if sum([np.sum(term) for term in gradient_contributions]) == 0:
        Z_gradient = 0
    else:
        Z_gradient = sum(gradient_contributions)
        ax_gradient.contourf(X, Y, Z_gradient, levels=10, cmap=cm.cork_r)
    ax_gradient.set_xticks([])
    ax_gradient.set_yticks([])
    plt.draw()

def update_rotational_potential():
    ax_rotational.clear()
    ax_rotational.set_title('Rotational Potential')
    if sum([np.sum(term) for term in rotational_contributions]) == 0:
        Z_rotational = 0
    else:
        Z_rotational = sum(rotational_contributions)
        ax_rotational.contourf(X, Y, Z_rotational, levels=10, cmap='RdBu_r')
    ax_rotational.set_xticks([])
    ax_rotational.set_yticks([])
    plt.draw()


def update_flow():
    ax_flow.clear()
    ax_flow.set_title('Flow Field')
    if circles:
        dX = sum(flow_contributions_x)
        dY = sum(flow_contributions_y)
        s = 5
        # ax_flow.streamplot(X[::s], Y[::s], dX[::s], dY[::s], linewidth=1, arrowsize=1, arrowstyle='->', color='darkgrey')
        ax_flow.quiver(X[::s, ::s], Y[::s, ::s], dX[::s, ::s], dY[::s, ::s], color='#a9a9a9', lw=2, width=0.005, headwidth=3, alpha=1)
        ax_flow.contour(X, Y, dX, levels=(0,), colors='steelblue')
        ax_flow.contour(X, Y, dY, levels=(0,), colors='k')
    ax_flow.set_xticks([])
    ax_flow.set_yticks([])
    plt.draw()


def update_circle_contribution(index):
    circle_x, circle_y = circles[index].center
    weight = alphas[index] * color_amplitude[colors[index]] * np.exp(
        -((X - circle_x) ** 2 + (Y - circle_y) ** 2) / (2 * radii[index] ** 2)
    )

    if colors[index] in ['tab:green', 'tab:blue']:
        gradient_contributions[index] = weight/radii[index]**2
        update_gradient_potential()
        flow_contributions_x[index] = weight*(X - circle_x)
        flow_contributions_y[index] = weight*(Y - circle_y)
    elif colors[index] in ['hotpink', 'tab:purple']:
        rotational_contributions[index] = weight/radii[index]**2
        update_rotational_potential()
        flow_contributions_x[index] = -weight*(Y - circle_y)
        flow_contributions_y[index] = weight*(X - circle_x)
    update_flow()


def on_click(event):
    global active_circle, addition_mode, delete_mode
    for b in buttons:
        b.set_linewidth(0)
        delete_button.set_linewidth(0)

    if event.inaxes == ax_buttons:  # Clicked on a button

        for i, button in enumerate(buttons):
            if button.contains_point((event.x, event.y)):
                addition_mode = button_colors[i]  #.get_facecolor()
                delete_mode = False
                button.set_edgecolor('black')  # Highlight color
                button.set_linewidth(2)
                plt.draw()
                return

        if delete_button.contains_point((event.x, event.y)):
            addition_mode = None
            delete_mode = True
            delete_button.set_edgecolor('black')  # Highlight color
            delete_button.set_linewidth(2)
            plt.draw()
            return

    elif event.inaxes == ax_left:

        if addition_mode is not None:
            color = addition_mode
            circle = plt.Circle((event.xdata, event.ydata), 0.3, color=color, alpha=0.5, picker=True)
            ax_left.add_patch(circle)
            circles.append(circle)
            radii.append(0.3)
            alphas.append(0.5)
            colors.append(color)

            # Add to the appropriate contribution list
            contribution = np.zeros_like(X)  # Placeholder for the contribution
            flow_contributions_x.append(contribution)
            flow_contributions_y.append(contribution)
            if color in ['tab:green', 'tab:blue']:
                gradient_contributions.append(contribution)
                rotational_contributions.append(0)
            elif color in ['hotpink', 'tab:purple']:
                rotational_contributions.append(contribution)
                gradient_contributions.append(0)

            # Update contributions for the new circle
            update_circle_contribution(len(circles) - 1)
            active_circle = len(circles) - 1
            addition_mode = None
            radius_slider.set_val(radii[active_circle])
            alpha_slider.set_val(alphas[active_circle])
            update_slider_colors()
            for button in buttons:
                button.set_linewidth(0)
            return

        if delete_mode:
            for i, circle in enumerate(circles):
                contains, _ = circle.contains(event)
                if contains:
                    circle.remove()
                    del circles[i]
                    del radii[i]
                    del alphas[i]
                    del gradient_contributions[i]
                    del rotational_contributions[i]
                    del flow_contributions_x[i]
                    del flow_contributions_y[i]

                    if colors[i] in ['tab:green', 'tab:blue']:
                        update_gradient_potential()
                    elif colors[i] in ['hotpink', 'tab:purple']:
                        update_rotational_potential()
                    update_flow()

                    del colors[i]
                    delete_mode = False
                    active_circle = None
                    update_slider_colors()
                    delete_button.set_edgecolor('none')
                    delete_button.set_linewidth(0)
                    plt.draw()
                    return

        else:  # Select a circle
            for i, circle in enumerate(circles):
                contains, _ = circle.contains(event)
                if contains:
                    active_circle = i
                    radius_slider.set_val(radii[i])
                    alpha_slider.set_val(alphas[i])
                    update_slider_colors()
                    return
            active_circle = None
            update_slider_colors()


def on_motion(event):
    global dragging, drag_start_x, drag_start_y, circle_start_x, circle_start_y, delete_mode, active_circle
    if active_circle is not None and event.inaxes == ax_left and dragging and not delete_mode:
        if drag_start_x is None or drag_start_y is None:
            # Safety check in case on_press didn't set the initial positions
            drag_start_x, drag_start_y = event.xdata, event.ydata
            circle_start_x, circle_start_y = circles[active_circle].center

        delta_x = event.xdata - drag_start_x
        delta_y = event.ydata - drag_start_y

        # Update circle center based on displacement
        new_center = (circle_start_x + delta_x, circle_start_y + delta_y)
        circles[active_circle].center = new_center
        update_circle_contribution(active_circle)
        plt.draw()

def on_release(event):
    global dragging, drag_start_x, drag_start_y
    dragging = False
    drag_start_x, drag_start_y = None, None


def on_press(event):
    global dragging, drag_start_x, drag_start_y, circle_start_x, circle_start_y, active_circle
    if event.inaxes == ax_left:
        for i, circle in enumerate(circles):
            contains, _ = circle.contains(event)
            if contains:
                active_circle = i
                drag_start_x, drag_start_y = event.xdata, event.ydata
                circle_start_x, circle_start_y = circles[active_circle].center
                dragging = True
                return
    dragging = False


# Event handler for mouse scroll to change radius
def on_scroll(event):
    if active_circle is not None and event.inaxes == ax_left:
        change = 0.01 if event.button == 'up' else -0.01
        new_radius = radii[active_circle] + change
        if 0.1 <= new_radius <= 1.:
            radii[active_circle] = new_radius
            circles[active_circle].set_radius(new_radius)
            # update_plots()
            update_circle_contribution(active_circle)
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
        # update_plots()
        update_circle_contribution(active_circle)
        plt.draw()
        alpha_slider.set_val(new_alpha)


# Slider update functions
def update_radius(val):
    if active_circle is not None:
        radii[active_circle] = val
        circles[active_circle].set_radius(val)
        update_circle_contribution(active_circle)
        plt.draw()


def update_alpha(val):
    if active_circle is not None:
        alphas[active_circle] = val
        circles[active_circle].set_alpha(val)
        # update_plots()
        update_circle_contribution(active_circle)
        plt.draw()


def update_slider_colors():
    if active_circle is not None:
        active_color = colors[active_circle]
        radius_slider.poly.set_fc(active_color)
        alpha_slider.poly.set_fc(active_color)
    else:
        default_color = 'lightgrey'
        radius_slider.poly.set_fc(default_color)
        alpha_slider.poly.set_fc(default_color)
    plt.draw()


ax_radius = plt.axes([0.15, 0.15, 0.25, 0.03])
ax_alpha = plt.axes([0.15, 0.1, 0.25, 0.03])

radius_slider = Slider(ax_radius, 'Radius', 0.1, 1., valinit=0.5)
alpha_slider = Slider(ax_alpha, 'Alpha', 0.1, 1., valinit=0.5)

radius_slider.on_changed(update_radius)
alpha_slider.on_changed(update_alpha)

# Create buttons
button_colors = ['tab:green', 'tab:blue', 'hotpink', 'tab:purple']
buttons = []
button_size = 0.08  # Increased size
button_spacing = 0.12  # Increased spacing to prevent overlap

for i, color in enumerate(button_colors):
    button_y = 0.9 - i * (button_spacing + button_size)
    button = plt.Circle((0.9, button_y), button_size, color=color, transform=ax_buttons.transAxes, clip_on=False, picker=True,
                        edgecolor='w', linewidth=0)
    ax_buttons.add_patch(button)
    buttons.append(button)

delete_button_height = 0.14
delete_button_y = 0.85 - len(button_colors) * (button_spacing + button_size) - 0.05
delete_button = plt.Rectangle((0.9-delete_button_height/2, delete_button_y), delete_button_height, delete_button_height, color='red', transform=ax_buttons.transAxes, clip_on=False)
ax_buttons.add_patch(delete_button)
ax_buttons.text(0.9, delete_button_y + delete_button_height / 2, 'x', ha='center', va='center', fontsize=30, color='white', weight='bold')
ax_buttons.axis('off')

update_slider_colors()

# Connect event handlers
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('key_press_event', on_key)

ax_left.set_xlim([-1, 1])
ax_left.set_ylim([-1, 1])
ax_buttons.set_aspect('equal', 'box')
ax_left.set_aspect('equal', 'box')
ax_left.set_title('Modules')
ax_flow.set_xlim([-1, 1])
ax_flow.set_ylim([-1, 1])
ax_flow.set_aspect('equal', 'box')
for ax in (ax_left, ax_gradient, ax_rotational, ax_flow):
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
