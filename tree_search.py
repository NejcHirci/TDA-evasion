from queue import PriorityQueue

from plotting import visualize

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
        all_nodes.append(grid)

    return all_nodes, locs


def check_siblings(node, grid, visited, start_pos):
    """Check if any sibling has a lower l1 distance to start_pos."""
    for other_node in node.siblings:
        if other_node in grid and \
                other_node not in visited \
                and other_node.l2(start_pos) < node.l2(start_pos):
            return False
    return True


def find_next_pos(pos, grid, t, start_pos=None):
    """Find path between pos and any valid position in grid."""
    queue = PriorityQueue()
    pos.parent = None
    queue.put((pos.l2(start_pos), pos))
    visited = []
    while not queue.empty():
        node = queue.get()[1]
        visited.append(node)
        # If the node is valid intruder can remain at this position
        if node in grid and check_siblings(node, grid, visited, start_pos):
            # Build path from last node to start node
            last_node = node
            path = [node]
            while node.parent is not None:
                node = node.parent
                path.append(node)
            last_node = grid[grid.index(last_node)]
            return last_node, path[-2:0:-1]
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
                print("Path broken")
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
            if pos is not None and pos == path[0]:
                path.append(pos)
                return path
            else:
                i += 1
                print(f"Path did not end at {path[0]}")
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

    periods = [sensor.timePeriod for sensor in sensors]
    period = smallest_common_multiple([sensor.timePeriod for sensor in sensors])
    all_nodes, locs = build_node_tree(sensors, period=period, grid_size=8)

    node_path = find_path(all_nodes)
    if node_path is None:
        print("No path found")
        exit(-1)

    path_list = [(node.time, node.x, node.y) for node in node_path]
    visualize(locs, node_path, grid_size=dim)

