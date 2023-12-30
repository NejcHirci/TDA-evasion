import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from queue import PriorityQueue

from evasion import *


class Node:
    def __init__(self, x, y, time):
        self.siblings = []
        self.x = x
        self.y = y
        self.time = time
        self.parent = None
        self.children = []

    def __repr__(self):
        return "Node({}, {}, {})".format(self.x, self.y, self.time)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __le__(self, other):
        return (self.x, self.y) <= (other.x, other.y)

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def l1(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

    def l2(self, other):
        return (self.x - other.x)**2 + (self.y - other.y)**2


def build_node_tree(sensors, period=8, grid_size=4):
    """Build the tree of nodes based on the sensor locations."""
    all_nodes = []
    locs = coveredSpace(sensors, period)

    for t in range(period):
        grid = []
        for x in range(grid_size):
            for y in range(grid_size):
                # Check if the node is in the sensor range
                if [y, x] not in locs[t]:
                    grid.append(Node(x, y, t))
        # Add siblings to each node
        for node in grid:
            for other_node in grid:
                # Connect only the 4 adjacent nodes
                if node != other_node and \
                   abs(node.x - other_node.x) + abs(node.y - other_node.y) == 1:
                    node.siblings.append(other_node)
        # # Try to connect the nodes to the nodes at the previous time step
        # if t == 0:
        #     all_nodes.append(grid)
        #     continue
        # for node in grid:
        #     for prev_node in all_nodes[t-1]:
        #         if node.x == prev_node.x and node.y == prev_node.y:
        #             node.parent = prev_node
        #             prev_node.children.append(node)
        #             break
        all_nodes.append(grid)

    return all_nodes, locs


def check_siblings(node, grid, visited, start_pos):
    """Check if any sibling has a lower l1 distance to start_pos."""
    for other_node in node.siblings:
        #print(other_node, node, other_node.l2(start_pos), node.l2(start_pos))
        if other_node in grid and other_node not in visited and \
                other_node.l2(start_pos) < node.l2(start_pos):
            return False
    return True

def find_next_pos(pos, grid, t, start_pos=None):
    """Find path between pos and any valid position in grid."""

    # If no path to start position exists try to find a path to any valid position
    queue = PriorityQueue()
    queue.put((pos.l2(start_pos), pos))
    visited = []
    while not queue.empty():
        node = queue.get()[1]
        # If the node is valid intruder can remain at this position

        if node in grid:
            # Build path from last node to start node
            last_node = node
            path = [node]
            while node.parent is not None and check_siblings(node, grid, visited, start_pos):
                node = node.parent
                path.append(node)
            last_node = grid[grid.index(last_node)]
            return last_node, path[-2:0:-1]

        visited.append(node)
        for neghb in node.siblings:
            if neghb not in visited:
                neghb.parent = node
                queue.put((neghb.l2(start_pos), neghb))

    print("No path found at time {}".format(t))
    return None, []


def find_path(all_grid, debug=False):
    """
    Find the path of the intruder through the grid.
    """
    i = 0
    while i < len(all_grid[0]):
        pos = all_grid[0][i]
        path = [pos]
        if debug:
            print("Time: {}, Position: {}".format(0, pos))
            print("Path: {}".format(path))
        for t in range(1, len(all_grid)):
            # If the current position is still valid the intruder does not need to move
            pos, new_path = find_next_pos(pos, all_grid[t], t - 1, start_pos=path[0])
            if pos is None:
                i += 1
                break
            path.extend(new_path)
            path.append(pos)
            if debug:
                print("Time: {}, Position: {}".format(t, pos))
                print("Path: {}".format(path))

        # If the path is valid we can return it
        if pos is not None:
            # Check if there exists a path from last position to start position
            pos, _ = find_next_pos(pos, all_grid[0], len(all_grid) - 1, start_pos=path[0])
            if pos is not None:
                path.append(pos)
                return path
            else:
                i += 1
                print(f"Path did not end at {path[0]}")
    return None



def visualize(sensor_locs, path, grid_size=(4, 4)):
    """
    Visualize the path of the intruder through the grid.
    """
    fig, ax = plt.subplots()
    ax.set_title("Evasion")
    ax.set_aspect('equal')

    data = np.zeros(grid_size)
    for ind in range(0, len(sensor_locs[0]), 4):
        loc = sensor_locs[0][ind]
        data[loc[0]][loc[1]] = ind + 2
        loc = sensor_locs[0][ind + 1]
        data[loc[0]][loc[1]] = ind + 2
        loc = sensor_locs[0][ind + 2]
        data[loc[0]][loc[1]] = ind + 2
        loc = sensor_locs[0][ind + 3]
        data[loc[0]][loc[1]] = ind + 2
    data[path[0].y][path[0].x] = 1

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

        if path[frame].time == time:
            new_data[path[frame].y][path[frame].x] = 1
            if path[(frame + 1) % len(path)].time != time:
                time += 1
        else:
            new_data[path[frame].y][path[frame].x] = 1

        cax = ax.pcolor(new_data[::-1], cmap=cmap, edgecolors='k', linewidths=1)
        frame = (frame + 1) % len(path)
        return cax,

    anim = animation.FuncAnimation(fig, animate, frames=len(path), interval=1000, blit=False)
    #plt.show()
    anim.save('evasion.gif', writer='imagemagick', fps=1)


if __name__ == "__main__":
    """
    First we must build the tree based on the inital setting of sensors.
    """
    """
    Coordinate system:
    (0, 0) is the top left corner
    (dim, dim) is the bottom right corner
    """
    """
    # Basic Example 1
    path = [
        [0, 0],
        [1, 0],
        [2, 0],
        [2, 1],
        [2, 2],
        [1, 2],
        [0, 2],
        [0, 1]
    ]
    sensor = Sensor((4, 4), path)

    all_nodes, locs = build_node_tree([sensor], period=8, grid_size=4)

    node_path = find_path(all_nodes)
    visualize(locs, node_path)
    """
    # Example 2
    dim = (8, 8)
    endpoints = [[0, 0], [5, 0]]
    start_pos = [0, 0]
    start_dir = [1, 0]
    sens_1 = DirectionalSensor(dim, endpoints, start_pos, start_dir)

    endpoints = [[0, 1], [0, 6]]
    start_pos = [0, 5]
    start_dir = [0, 1]
    sens_2 = DirectionalSensor(dim, endpoints, start_pos, start_dir)

    endpoints = [[2, 2], [4, 2]]
    start_pos = [4, 2]
    start_dir = [-1, 0]
    sens_3 = DirectionalSensor(dim, endpoints, start_pos, start_dir)

    endpoints = [[2, 4], [4, 4]]
    start_pos = [3, 4]
    start_dir = [1, 0]
    sens_4 = DirectionalSensor(dim, endpoints, start_pos, start_dir)

    endpoints = [[2, 6], [6, 6]]
    start_pos = [4, 6]
    start_dir = [1, 0]
    sens_5 = DirectionalSensor(dim, endpoints, start_pos, start_dir)

    endpoints = [[6, 0], [6, 4]]
    start_pos = [6, 3]
    start_dir = [0, -1]
    sens_6 = DirectionalSensor(dim, endpoints, start_pos, start_dir)

    sensors = [sens_1, sens_2, sens_3, sens_4, sens_5, sens_6]
    all_nodes, locs = build_node_tree(sensors, period=40, grid_size=8)

    node_path = find_path(all_nodes)
    if node_path is None:
        print("No path found")
        exit(-1)
    visualize(locs, node_path, grid_size=dim)

