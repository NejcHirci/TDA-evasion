import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


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


class Sensor:
    def __init__(self, start_x, start_y, end_x, end_y):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

    def __repr__(self):
        return "Sensor({}, {}, {}, {})".format(self.start_x, self.start_y, self.end_x, self.end_y)


def build_node_tree():
    """
    Build the tree of nodes based on the sensor locations.

    TODO: Implement a function to build the tree of nodes based on the sensor locations.
    """
    pass


def generate_configuration():
    """
    TODO: room generator
    Generate a room with random sensor locations and movements, so that
    the sensors cover the whole room in one period.
    :return:
    """


def find_next_pos(pos, grid, t):
    """Find path between pos and any valid position in grid."""
    # Check if the current position is valid
    if pos in grid:
        return pos

    # Use BFS to find a valid position
    queue = [pos]
    visited = []
    while queue:
        node = queue.pop(0)
        if node in grid:
            return node
        visited.append(node)
        for neghb in node.siblings:
            if neghb not in visited:
                queue.append(neghb)

    return None


def visualize(sensor_locs, path):
    """
    Visualize the path of the intruder through the grid.
    """
    fig, ax = plt.subplots()
    ax.set_title("Evasion")

    graph = ax.scatter([], [])

    def update(frame):
        # Plot sensors
        offsets = []
        cols = []
        for i in range(len(sensor_locs[frame])):
            sensor = sensor_locs[frame][i]
            offsets.append([sensor[1], sensor[0]])
            cols.append('r')
        offsets.append([path[frame].x, path[frame].y])
        cols.append('b')
        graph.set_offsets(offsets)
        graph.set_facecolors(cols)
        graph.set_sizes([100] * len(offsets))
        return graph

    ani = animation.FuncAnimation(fig, update, interval=1000)
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
            pos = find_next_pos(pos, all_grid[t], t)
            if pos is None:
                i += 1
                break
            path.append(pos)
        # If the path is valid we can return it
        if pos is not None:
            # Check if there exists a path from last position to start position
            pos = find_next_pos(pos, all_grid[0], 0)
            if pos is not None:
                path.append(pos)
                return path
    return None


if __name__ == "__main__":
    """
    First we must build the tree based on the inital setting of sensors.
    """

    loc_0 = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    loc_1 = [
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1]
    ]
    loc_2 = [
        [2, 0],
        [2, 1],
        [3, 0],
        [3, 1]
    ]
    loc_3 = [
        [2, 1],
        [2, 2],
        [3, 1],
        [3, 2]
    ]
    loc_4 = [
        [2, 2],
        [2, 3],
        [3, 2],
        [3, 3]
    ]
    loc_5 = [
        [1, 2],
        [1, 3],
        [2, 2],
        [2, 3]
    ]
    loc_6 = [
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3]
    ]
    loc_7 = [
        [0, 1],
        [0, 2],
        [1, 1],
        [1, 2]
    ]

    locs = [loc_0, loc_1, loc_2, loc_3, loc_4, loc_5, loc_6, loc_7]

    all_nodes = []
    for t in range(7):
        grid = []
        for x in range(4):
            for y in range(4):
                # Check if the node is in the sensor range
                if [y, x] not in locs[t]:
                    grid.append(Node(x, y, t))
        # Add siblings to each node
        for node in grid:
            for other_node in grid:
                if abs(node.x - other_node.x) <= 1 and abs(node.y - other_node.y) <= 1:
                    node.siblings.append(other_node)
        # Try to connect the nodes to the nodes at the previous time step
        if t == 0:
            all_nodes.append(grid)
            continue
        for node in grid:
            for prev_node in all_nodes[t-1]:
                if node.x == prev_node.x and node.y == prev_node.y:
                    node.parent = prev_node
                    prev_node.children.append(node)
                    break
        all_nodes.append(grid)

    node_path = find_path(all_nodes)
    visualize(locs, node_path)
