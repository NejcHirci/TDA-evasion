import numpy as np


class Sensor:
    def __init__(self, spaceDim, sensorPath):
        self.sensorPath = sensorPath  # [[x,y],[x,y],[x,y]]
        self.spaceDim = spaceDim
        self.timePeriod = len(sensorPath)

    def getSensorArea(self):
        """Returns coverage of sensor in spaceDim."""
        n_frames = len(self.sensorPath)
        space = np.zeros((n_frames, *self.spaceDim))

        for t in range(n_frames):
            y, x = self.sensorPath[t]
            space[t][y][x] = 1
            space[t][y][x + 1] = 1
            space[t][y + 1][x] = 1
            space[t][y + 1][x + 1] = 1
        return space

    def getSensorAreaWithoutTime(self):
        """Returns coverage of sensor in spaceDim."""
        n_frames = len(self.sensorPath)
        space = np.full(self.spaceDim, False)

        for t in range(n_frames):
            y, x = self.sensorPath[t]
            space[y][x] = True
            space[y][x + 1] = True
            space[y + 1][x] = True
            space[y + 1][x + 1] = True
        return space

    def getCorners(self, t):
        t = t % self.timePeriod
        return [
            [self.sensorPath[t][0], self.sensorPath[t][1]],
            [self.sensorPath[t][0], self.sensorPath[t][1] + 1],
            [self.sensorPath[t][0] + 1, self.sensorPath[t][1]],
            [self.sensorPath[t][0] + 1, self.sensorPath[t][1] + 1]
        ]


# Sensor moves in straight lines
class SimpleSensor:
    def __init__(self, spaceDim, start, stop):  # start stop naj bosta [*, 0], [*, 0] ali [0, *], [0, *]
        self.start = np.array(start)
        self.stop = np.array(stop)
        self.spaceDim = spaceDim
        self.timePeriod = 2 * np.max(np.abs(self.start - self.stop))

    def getSensorArea(self):
        if self.start[0] == self.stop[0]:
            direction = np.array([0, 1]) * (((self.start[1] - self.stop[1]) < 0) * 2 - 1)
        elif self.start[1] == self.stop[1]:
            direction = np.array([1, 0]) * (((self.start[0] - self.stop[0]) < 0) * 2 - 1)
        space = np.zeros((self.timePeriod, *self.spaceDim))
        rangee = int(self.timePeriod / 2)
        for t in range(rangee + 1):
            y, x = self.start + t * direction
            space[t][y][x] = 1
            space[t][y][x + 1] = 1
            space[t][y + 1][x] = 1
            space[t][y + 1][x + 1] = 1
        for t in range(0, rangee - 1):
            y, x = self.stop - (t + 1) * direction
            space[rangee + 1 + t][y][x] = 1
            space[rangee + 1 + t][y][x + 1] = 1
            space[rangee + 1 + t][y + 1][x] = 1
            space[rangee + 1 + t][y + 1][x + 1] = 1
        return space


class DirectionalSensor(Sensor):
    def __init__(self, spaceDim, endpoints, startPos, startDir):
        self.startPos = np.array(startPos)
        self.endpoints = np.array(endpoints)
        self.startDir = np.array(startDir)
        super().__init__(spaceDim, self.initSensorPath())


    def initSensorPath(self):
        """Creates a path the sensor will follow."""
        dir = self.startDir
        pos = self.startPos
        path = [pos.tolist()]

        self.timePeriod = 2 * np.max(np.abs(self.endpoints[0] - self.endpoints[1]))


        for t in range(1, self.timePeriod):
            pos += dir
            if np.array_equal(pos, self.endpoints[0]) or np.array_equal(pos, self.endpoints[1]):
                dir = -dir
            path.append(pos.tolist())

        return np.array(path)

    def __repr__(self):
        return "Sensor({}, {}, {})".format(self.endpoints, self.startPos, self.startDir)


class SensorsSpace:
    """SensorsSpace takes a list of start and end points of sensors and creates one space."""
    def __init__(self, spaceDim, sensorStartStopList):
        self.sensorStartStopList = sensorStartStopList  # [[[startX,startY],[stopX,*]], [[*,*],[*,*]], [[*,*],[*,*]] ]
        self.spaceDim = spaceDim  # [*,*]
        self.sensors = []
        self.timePeriod = 0
        self.getCommonTimePeriod()

    def getCommonTimePeriod(self):
        periods = []
        for startStop in self.sensorStartStopList:
            sensor = SimpleSensor(self.spaceDim, startStop[0], startStop[1])
            self.sensors.append(sensor)
            periods.append(sensor.timePeriod)

        self.timePeriod = smallest_common_multiple(periods)

    def createSpace(self):
        space = np.zeros((self.timePeriod, *self.spaceDim))
        for sensor in self.sensors:
            n_roundtrips = int(self.timePeriod / sensor.timePeriod)

            space += np.tile(sensor.getSensorArea(), (n_roundtrips, 1, 1))
        return space > 0


def gcd(a, b):
    """Return greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """Return lowest common multiple."""
    return abs(a * b) // gcd(a, b)


def smallest_common_multiple(numbers):
    """Return smallest common multiple of numbers."""
    result = 1
    for num in numbers:
        result = lcm(result, num)
    return result


def giveSpaceTimeDirection(space, step = 0.1):
    n_slices = np.shape(space)[0]
    sliceValues = np.arange(0.1, (n_slices+1)*step, step)
    tmin, tmax1 = sliceValues[[0, -1]]
    t_inf = tmax1*10
    sliceValues = np.reshape(sliceValues, (n_slices,1,1))
    newSpaceInf = space*t_inf
    newSpaceElse = (~space) * sliceValues
    newSpace = np.round(newSpaceInf + newSpaceElse, 2)
    return newSpace, tmin, tmax1, t_inf


def giveSpaceNoTimeDirection(space):
    return space*5 + 0.1, 0.1, 0.1, 5


def coveredSpace(sensors, period):
    space = []
    for t in range(period):
        locs = []
        for sensor in sensors:
            locs.extend(sensor.getCorners(t))
        space.append(locs)
    return space


def constructSpace(sensors, period, dim=8):
    space = np.ones((period, dim, dim), dtype=bool)
    for t in range(period):
        for sensor in sensors:
            locs = sensor.getCorners(t)
            for loc in locs:
                space[t][loc[0]][loc[1]] = 0
    return space


if __name__ == "__main__":
    print("Testing")
    endpoints = [[6, 0], [6, 4]]
    startPos = [6, 3]
    startDir = [0, -1]
    sensor = DirectionalSensor((8, 8), endpoints, startPos, startDir)
    print(sensor.sensorPath)