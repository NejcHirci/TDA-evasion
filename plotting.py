import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors

from evasion import *

import matplotlib
# matplotlib.use("TkAgg")


def sort_path(path, time_dim):
    """Sorts the path according to time, x, y."""
    new_path = []
    for t in range(time_dim):
        p = [p for p in path if p[0] == t]
        if t == 0:
            p = sorted(p, key=lambda x: (x[0], x[1], x[2]))
        else:
            last_pos = new_path[-1]
            # Check if last position is highest or lowest in new path
            if last_pos[1] >= p[0][1] and last_pos[2] >= p[0][2]:
                p = sorted(p, key=lambda x: (x[0], -x[1], -x[2]))
            else:
                p = sorted(p, key=lambda x: (x[0], x[1], x[2]))
        new_path.extend(p)
    return new_path


def visualize(sensor_locs, path, grid_size=4):
    """
    Visualize the path of the intruder through the grid.
    path: list of tuples (time, x, y)
    sensor_locs: list of lists of tuples (x, y) for every time step
    path: list of tuples (time, x, y)
    """
    path = sort_path(path, len(sensor_locs))

    fig, ax = plt.subplots(dpi=300)
    ax.set_title("Evasion")
    ax.set_aspect('equal')

    data = np.zeros((grid_size, grid_size))
    for ind in range(0, len(sensor_locs[0]), 4):
        loc = sensor_locs[0][ind]
        data[loc[0]][loc[1]] = ind + 2
        loc = sensor_locs[0][ind + 1]
        data[loc[0]][loc[1]] = ind + 2
        loc = sensor_locs[0][ind + 2]
        data[loc[0]][loc[1]] = ind + 2
        loc = sensor_locs[0][ind + 3]
        data[loc[0]][loc[1]] = ind + 2
    data[path[0][2]][path[0][1]] = 1

    # Create a custom colormap for all sensors
    colors = ['white', (0, 1, 0)]
    for i in range(0, len(sensor_locs)):
        # Generate random red color with different shades
        color = (1, np.random.rand() * 0.8, np.random.rand() * 0.8)
        colors.append(color)

    cmap = matplotlib.colors.ListedColormap(colors)
    cax = ax.pcolor(data[::-1], cmap=cmap, edgecolors='k', linewidths=1)

    time = 0
    frame = 0

    def animate(i):
        nonlocal time, frame
        ax.clear()
        ax.set_title(f"Evasion {time}/{len(sensor_locs)}")
        new_data = np.zeros(grid_size)

        time = time % len(sensor_locs)
        for ind in range(0, len(sensor_locs[time]), 4):
            loc = sensor_locs[time][ind]
            new_data[loc[0]][loc[1]] = ind + 2
            loc = sensor_locs[time][ind + 1]
            new_data[loc[0]][loc[1]] = ind + 2
            loc = sensor_locs[time][ind + 2]
            new_data[loc[0]][loc[1]] = ind + 2
            loc = sensor_locs[time][ind + 3]
            new_data[loc[0]][loc[1]] = ind + 2

        if path[frame][0] == time:
            new_data[path[frame][2]][path[frame][1]] = 1
            if path[(frame + 1) % len(path)][0] != time:
                time += 1
        else:
            new_data[path[frame].y][path[frame].x] = 1

        cax = ax.pcolor(new_data[::-1], cmap=cmap, edgecolors='k', linewidths=1)
        frame = (frame + 1) % len(path)
        return cax,

    anim = animation.FuncAnimation(fig, animate, frames=len(path), interval=1000, blit=False)
    anim.save('evasion.gif', writer='imagemagick', fps=1)


def visualize_sensors(sensors, period, grid_size=4):
    """
    Visualize only the sensor movements.
    sensors: list of sensors
    period: period of the sensors
    grid_size: size of the grid
    """
    locs = coveredSpace(sensors, period)
    fig, ax = plt.subplots()
    ax.set_title("Evasion")
    ax.set_aspect('equal')

    data = np.zeros((grid_size, grid_size))
    for ind in range(0, len(locs[0]), 4):
        loc = locs[0][ind]
        data[loc[0]][loc[1]] = ind + 2
        loc = locs[0][ind + 1]
        data[loc[0]][loc[1]] = ind + 2
        loc = locs[0][ind + 2]
        data[loc[0]][loc[1]] = ind + 2
        loc = locs[0][ind + 3]
        data[loc[0]][loc[1]] = ind + 2

    # Create a custom colormap for all sensors
    colors = ['white']
    for i in range(0, len(locs)):
        # Generate random red color with different shades
        color = (1, np.random.rand() * 0.8, np.random.rand() * 0.8)
        colors.append(color)

    cmap = matplotlib.colors.ListedColormap(colors)
    cax = ax.pcolor(data[::-1], cmap=cmap, edgecolors='k', linewidths=1)

    def animate(time):
        ax.clear()
        ax.set_title(f"Evasion {time}/{len(locs)}")
        new_data = np.zeros((grid_size, grid_size))

        for ind in range(0, len(locs[time]), 4):
            loc = locs[time][ind]
            new_data[loc[0]][loc[1]] = ind + 2
            loc = locs[time][ind + 1]
            new_data[loc[0]][loc[1]] = ind + 2
            loc = locs[time][ind + 2]
            new_data[loc[0]][loc[1]] = ind + 2
            loc = locs[time][ind + 3]
            new_data[loc[0]][loc[1]] = ind + 2

        cax = ax.pcolor(new_data[::-1], cmap=cmap, edgecolors='k', linewidths=1)
        return cax,

    anim = animation.FuncAnimation(fig, animate, frames=len(locs), interval=1000, blit=False)
    plt.show()


def visualize_space(space, path):
    """Visualization for visualizing from space.
    space: ndarray of shape (time, x, y) of type bool if 0 no sensor
    """

    # Sort the path appropriately
    frameNum = space.shape[0]
    if path is not None:
        path = sort_path(path, space.shape[0])
        frameNum = len(path)
    space = space.astype(int)

    fig, ax = plt.subplots(dpi=300)
    ax.set_title("Evasion")
    ax.set_aspect('equal')

    # Create a custom colormap for all sensors
    colors = ['red', 'white', 'green']

    cmap = matplotlib.colors.ListedColormap(colors)
    cax = ax.pcolor(space[0], cmap=cmap, edgecolors='k', linewidths=1)

    time = 0
    frame = 0

    print(space)

    def animate(i):
        nonlocal time, frame
        ax.clear()
        ax.set_title(f"Evasion {time}/{space.shape[0]}")
        time = time % space.shape[0]
        new_data = np.array(space[time])

        if path is not None:
            if path[frame][0] == time:
                new_data[path[frame][1]][path[frame][2]] = 2
                if path[(frame + 1) % len(path)][0] != time:
                    time += 1
            else:
                new_data[path[frame].y][path[frame].x] = 2
        else:
            time += 1

        cax = ax.pcolor(new_data[::-1], cmap=cmap, edgecolors='k', linewidths=1)
        frame = (frame + 1) % frameNum
        return cax,

    anim = animation.FuncAnimation(fig, animate, frames=frameNum, interval=1000, blit=False)
    plt.show()
    anim.save('what.gif', writer='imagemagick', fps=1)

