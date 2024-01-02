import numpy as np

from evasion import *

from plotting import visualize_sensors


def naive_generator(grid_size):
    """
    Generate a room with random sensor locations and movements, so that
    the sensors cover the whole room in one period.
    :return: sensors
    """
    coverage = np.full((grid_size, grid_size), False)
    sensors = []

    it = 0
    while not np.all(coverage):
        it += 1
        zero_locs = [a for a in np.argwhere(~coverage)]
        start = zero_locs[np.random.randint(len(zero_locs))]
        start[0] = np.clip(start[0], 0, grid_size - 2)
        start[1] = np.clip(start[1], 0, grid_size - 2)
        start = np.array(start)

        # Sum the amount of zeros above start point but sensor has 2x2 area
        sumUp = np.sum(~coverage[:start[0], start[1]:start[1] + 2])
        # Sum the amount of zeros below start point
        sumDown = np.sum(~coverage[start[0] + 2:, start[1]:start[1] + 2])
        # Sum the amount of zeros left of start point
        sumLeft = np.sum(~coverage[start[0]:start[0]+2, :start[1]])
        # Sum the amount of zeros right of start point
        sumRight = np.sum(~coverage[start[0]:start[0]+2, start[1] + 2:])

        # Choose direction with the most zeros
        # direction = np.argmax([sumUp, sumDown, sumLeft, sumRight])
        # dir = [[-1, 0], [1, 0], [0, -1], [0, 1]][direction]
        # dir = np.array(dir)

        # Choose direction by sampling according to the amount of zeros
        sumAll = sumUp + sumDown + sumLeft + sumRight
        if sumAll == 0:
            # Create stationary sensor
            sensor = DirectionalSensor((grid_size, grid_size), [start, start], start, [0, 1])
            sensors.append(sensor)
            coverage = np.logical_or(coverage, sensor.getSensorAreaWithoutTime())
            print(np.sum(coverage) / (grid_size ** 2))
            continue
        direction = np.random.choice([0, 1, 2, 3], p=[sumUp / sumAll, sumDown / sumAll, sumLeft / sumAll, sumRight / sumAll])
        dir = [[-1, 0], [1, 0], [0, -1], [0, 1]][direction]
        dir = np.array(dir)

        # Choose endpoint
        endpoint = start
        if direction == 0: # Up
            endY = np.argwhere(~coverage[:start[0], start[1]])[0][0]
            endpoint = np.array([endY, start[1]])
        elif direction == 1: # Down
            endY = np.argwhere(~coverage[start[0]:, start[1]])[-1][0]
            endpoint = np.array([start[0] + endY, start[1]])
        elif direction == 2: # Left
            endX = np.argwhere(~coverage[start[0], :start[1]])[0][0]
            endpoint = np.array([start[0], endX])
        else: # Right
            endX = np.argwhere(~coverage[start[0], start[1]:])[-1][0]
            endpoint = np.array([start[0], start[1] + endX])

        endpoint[0] = np.clip(endpoint[0], 0, grid_size - 2)
        endpoint[1] = np.clip(endpoint[1], 0, grid_size - 2)

        # Choose a random point on the line between start and endpoint
        start_loc = start
        if abs(endpoint[0] - start[0]) > 1 or abs(endpoint[1] - start[1]) > 1:
            path = [start]
            pos = np.array(start)
            while pos[0] != endpoint[0] or pos[1] != endpoint[1]:
                pos += dir
                path.append(np.array(pos))

            if len(path) > 2:
                start_loc = path[np.random.randint(1, len(path) - 1)]

        # Create a sensor with the selected path
        sensor = DirectionalSensor((grid_size, grid_size), [start, endpoint], start_loc, dir)
        sensors.append(sensor)

        # Update coverage
        coverage = np.logical_or(coverage, sensor.getSensorAreaWithoutTime())
        print(np.sum(coverage) / (grid_size ** 2))
    p = smallest_common_multiple([sensor.timePeriod for sensor in sensors])
    return sensors, p


if __name__ == "__main__":
    size = 8
    sensors, p = naive_generator(size)
    print(sensors)
    visualize_sensors(sensors, p, size)
