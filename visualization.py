import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def find_next_pos(pos, grid, t):
    """Find path between pos and any valid position in grid."""
    # Check if the current position is valid
    if pos in grid:
        node = grid[grid.index(pos)]
        return node

    # Use BFS to find a valid position
    queue = [pos]
    visited = []
    while queue:
        node = queue.pop(0)
        # If the node is valid intruder can remain at this position
        if node in grid:
            # Get node at new time from grid
            node = grid[grid.index(node)]
            return node
        visited.append(node)
        for neghb in node.siblings:
            if neghb not in visited:
                queue.append(neghb)

    print("No path found at time {}".format(t))
    return None


def visualize(sensor_locs, path, grid_size=(4, 4)):
    """
    Visualize the path of the intruder through the grid.
    """
    fig, ax = plt.subplots()
    ax.set_title("Evasion")
    ax.set_aspect('equal')

    data = np.zeros(grid_size)
    cmap = matplotlib.colors.ListedColormap(['white', 'red', 'green'])
    cax = ax.pcolor(data[::-1], cmap=cmap, edgecolors='k', linewidths=3)

    def animate(i):
        i = i % len(sensor_locs)
        ax.clear()
        ax.set_title(f"Evasion {i}/{len(sensor_locs)}")
        new_data = np.zeros(grid_size)
        for loc in sensor_locs[i]:
            new_data[loc[0]][loc[1]] = 1
        new_data[path[i].y][path[i].x] = 2
        cax = ax.pcolor(new_data[::-1], cmap=cmap, edgecolors='k', linewidths=3)
        return cax,

    anim = animation.FuncAnimation(fig, animate, frames=len(sensor_locs), interval=1000, blit=False)
    plt.show()


def find_path(all_grid):
    """
    Find the path of the intruder through the grid.
    """
    i = 0
    while i < len(all_grid[0]):
        pos = all_grid[0][i]
        path = [pos]
        for t in range(1, len(all_grid)):
            # If the current position is still valid the intruder does not need to move
            pos = find_next_pos(pos, all_grid[t], t - 1)
            if pos is None:
                i += 1
                break
            path.append(pos)
        # If the path is valid we can return it
        if pos is not None:
            # Check if there exists a path from last position to start position
            pos = find_next_pos(pos, all_grid[0], len(all_grid) - 1)
            if pos is not None:
                path.append(pos)
                return path
    return None


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
    visualize(locs, node_path, grid_size=dim)

