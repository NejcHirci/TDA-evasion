from findPaths import *
from generator import naive_generator
import matplotlib
from matplotlib import animation

def sort_path(path, time_dim):
    """Sorts the path according to time, x, y."""
    new_path = []
    for t in range(time_dim):
        p = [p for p in path if p[0] == t]
        if len(new_path) > 0 and len(p) > 1:
            last_pos = new_path[-1]
            # Find index of the point with same coordinates as last_pos
            ind = [i for i in range(len(p)) if p[i][1] == last_pos[1] and p[i][2] == last_pos[2]][0]
            path_time = [p[ind]]
            visited = np.zeros(len(p))
            visited[ind] = 1
            cur = p[ind]
            while np.sum(visited) < len(p):
                for ind in range(len(p)):
                    if visited[ind] == 1:
                        continue
                    point = p[ind]
                    if abs(cur[1] - point[1]) + abs(cur[2] - point[2]) == 1:
                        cur = point
                        path_time.append(cur)
                        visited[ind] = 1
                        break
            new_path.extend(path_time)
        else:
            new_path.extend(p)
    return new_path


def visualize(sensor_locs, path, grid_size=4, needsSorting=True, fname='evasion.gif'):
    """
    Visualize the path of the intruder through the grid.
    path: list of tuples (time, x, y)
    sensor_locs: list of lists of tuples (x, y) for every time step
    path: list of tuples (time, x, y)
    """
    if needsSorting:
        path = sort_path(path, len(sensor_locs))

    fig, ax = plt.subplots(dpi=300)
    ax.set_title("Evasion")
    ax.set_aspect('equal')

    sensorCount = len(sensor_locs[0]) // 4
    # Create a custom colormap for all sensors
    colors = [(1,1,1), (0, 1, 0)]
    for i in range(0, sensorCount):
        # Generate random red color with different shades
        color = (np.random.rand(), 0, 0)
        colors.append(color)
    cMap = matplotlib.colors.ListedColormap(colors)

    data = np.zeros((grid_size, grid_size), dtype=int)
    for ind in range(0, len(sensor_locs[0]), 4):
        loc = sensor_locs[0][ind]
        data[loc[0]][loc[1]] = ind + 2
        loc = sensor_locs[0][ind + 1]
        data[loc[0]][loc[1]] = ind + 2
        loc = sensor_locs[0][ind + 2]
        data[loc[0]][loc[1]] = ind + 2
        loc = sensor_locs[0][ind + 3]
        data[loc[0]][loc[1]] = ind + 2
    data[path[0][1]][path[0][2]] = 1

    cax = ax.pcolor(data[::-1], cmap=cMap, edgecolors='k', linewidths=1, animated=True)
    time = 0
    frame = 0

    def animate(i):
        nonlocal time, frame, cax
        ax.clear()
        ax.set_title(f"Evasion {time}/{len(sensor_locs)}")
        new_data = np.zeros((grid_size, grid_size), dtype=int)

        time = time % len(sensor_locs)
        s = 2
        for ind in range(0, len(sensor_locs[time]), 4):
            loc = sensor_locs[time][ind]
            new_data[loc[0]][loc[1]] = s
            loc = sensor_locs[time][ind + 1]
            new_data[loc[0]][loc[1]] = s
            loc = sensor_locs[time][ind + 2]
            new_data[loc[0]][loc[1]] = s
            loc = sensor_locs[time][ind + 3]
            new_data[loc[0]][loc[1]] = s
            s += 1

        if path[frame][0] == time:
            new_data[path[frame][1]][path[frame][2]] = 1
            if path[(frame + 1) % len(path)][0] != time:
                time += 1
        else:
            new_data[path[frame][1]][path[frame][2]] = 1

        cMap = matplotlib.colors.ListedColormap(colors)
        cax = ax.pcolor(new_data[::-1], cmap=cMap, edgecolors='k', linewidths=1)
        frame = (frame + 1) % len(path)
        return cax,

    anim = animation.FuncAnimation(fig, animate, frames=len(path), interval=1000, blit=True)
    anim.save(f'output/{fname}', writer='ima', fps=2)
    plt.show()


def visualize_sensors(sensors, period, grid_size=4, fname="sensors.gif"):
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
    anim.save(f'output/{fname}', writer='Pillow', fps=2)
    plt.show()


def visualize_space(space, path, fname='space.gif'):
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
    colors = ['white', 'red', 'green']

    cmap = matplotlib.colors.ListedColormap(colors)
    cax = ax.pcolor(space[0], cmap=cmap, edgecolors='k', linewidths=1)

    time = 0
    frame = 0

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
    anim.save(f'output/{fname}', writer='Pillow', fps=2)


if __name__ == "__main__":
    sensors, p = naive_generator(8)
    all_nodes, locs = build_node_tree(sensors, period=p, grid_size=8)
    space = constructSpace(sensors, p, dim=8)
    node_path = find_path(all_nodes)
    print("Path found: ", node_path is not None)
    visualize_sensors(sensors, p, grid_size=8, fname="generated_example_3_sensors.gif")
    if node_path is not None:
        node_path = [[node.time, node.y, node.x] for node in node_path]
        visualize(locs, node_path, needsSorting=False, grid_size=8, fname="generated_example_3.gif")
    print(p)
    print("Path found: ", node_path is not None)
    print("TREE SEARCH DONE")
    paths = narrowPaths(space, onePath=True)

    print("Number of paths found: ", len(paths))
    if len(paths) > 0:
        specialCoords = paths[0]
        visualize(locs, specialCoords, grid_size=8, fname="generated_example_3_topo.gif")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    spaceShape = np.shape(space)
    # nariši najdeno pot, (Če je več poti lahko prikaže tudi dele drugih poti)
    if len(paths) > 0:
        for coord in specialCoords:
            draw_cube(ax, coord, size = 1)

    ax.set_xlim(0, spaceShape[0])
    ax.set_ylim(0, spaceShape[1])
    ax.set_zlim(0, spaceShape[2])
    ax.set_xlabel('T')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.set_aspect('equal')
    ax.set_title("Path %i"%pathId)
    plt.show()
    fig.savefig("output/generated_example_3_path.png", dpi=300)
