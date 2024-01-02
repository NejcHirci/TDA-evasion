import gudhi as gd  
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tree_search import *
# Plotting stuff

# function to get solid cube coordinates
def getCubeCoordinates(space):
    z, y, x = np.shape(space)
    points = []
    for i in range(z):
        for j in range(y):
            for k in range(x):
                if space[i][j][k] == 0:
                    points.append([i, j, k])
    return np.array(points)

def draw_cube(ax, coords, size):
    """
    Draw a cube in 3D space.

    Parameters:
    - ax: Matplotlib 3D axis
    - size: Size of the cube
    """
    # Define cube vertices
    vertices = [
        [coords[0], coords[1], coords[2]],
        [coords[0] + size, coords[1], coords[2]],
        [coords[0] + size, coords[1] + size, coords[2]],
        [coords[0], coords[1] + size, coords[2]],
        [coords[0], coords[1], coords[2] + size],
        [coords[0] + size, coords[1], coords[2] + size],
        [coords[0] + size, coords[1] + size, coords[2] + size],
        [coords[0], coords[1] + size, coords[2] + size],
    ]

    # Define cube faces
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[7], vertices[6], vertices[2], vertices[3]],
        [vertices[0], vertices[4], vertices[7], vertices[3]],
        [vertices[1], vertices[5], vertices[6], vertices[2]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[2], vertices[3]]
    ]

    # Plot the cube
    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))

def plotCubes(space):
    """
    Draws cubes in 3D space, that are 0 in the input space.
    """
    spaceDim = np.shape(space)
    coordsList = getCubeCoordinates(space)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for coord in coordsList:
        draw_cube(ax, coord, size = 1)

    ax.set_xlim(0, spaceDim[0])
    ax.set_ylim(0, spaceDim[1])
    ax.set_zlim(0, spaceDim[2])
    ax.set_xlabel('T')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.set_title("Whole space")
    fig.savefig("output/test_example_space.png", dpi=300)

def find_islands(matrix):
    """
    Finds islands of ones in a matrix of zeros and ones.
    """
    labeled_matrix, num_labels = measure.label(matrix, connectivity=1, return_num=True)
    return labeled_matrix, num_labels

def labelIslands(space):
     # Finds all important islands
    spaceShape = np.shape(space)
    newSpace, t_inf = giveSpaceNoTimeDirection(space)
    modulatedSpace = newSpace.copy()
    labeled_matrix_hist = []
    allImportantLs = []
    for tt in range(0, spaceShape[0]):
        labeled_matrix, num_labels = find_islands(~space[tt])
        labeled_matrix_hist.append(labeled_matrix)
        lls = np.array([i for i in range(1, num_labels+1)])
        importantLs = []
        for ord in range(num_labels):
            modulatedSpace = newSpace.copy()
            for l in np.append(lls[ord:], lls[:ord]):
                modulatedSpace[tt][labeled_matrix == l] = 1
                path = doesPathExists(modulatedSpace, t_inf)
                if not path:
                    modulatedSpace[tt][labeled_matrix == l] = 0
                    if l not in importantLs:
                        importantLs.append(l)
        allImportantLs.append(importantLs)
    return labeled_matrix_hist, allImportantLs

def giveSpaceNoTimeDirection(space):
    """
    Instead of zeros and ones, the space is given as a matrix of 0.1 and 1.
    Needed for persistance.
    """
    newSpace = space.copy()
    newSpace = ~newSpace*0.1 + newSpace
    t_inf = 1
    return newSpace, t_inf

def giveSpaceTimeLastSlice(space, timeDelay = 0, yDelay = 0, xDelay = 0, timeStep = 0.1, ):
    """
    Changes space numbers from zero to 0.1 and 1 to bigger number t_inf. One time slice also get bigger value and one cube 
    in that time slice gets even bigger number. 0.1 < tmax1 < tmax2 < t_inf.
    """
    spaceShape = np.shape(space)

    addSpace = np.ones(spaceShape)*timeStep # all space 0 to 0.1
    addSpace[timeDelay] = timeStep*spaceShape[0] # the chosen time slice gets value tmax1

    addSpace[timeDelay][yDelay][xDelay] = (spaceShape[0]+1)*timeStep # the chosen cube in chosen time slice gets value tmax2
    tmin = timeStep
    tmax1 = (spaceShape[0])*timeStep
    tmax2 = (spaceShape[0]+1)*timeStep
    t_inf = np.max(addSpace)*10 # senzor values are the biggest
    
    # combine all together
    newSpaceInf = space*t_inf
    newSpaceElse = (~space) * (addSpace)
    newSpace = np.round(newSpaceInf + newSpaceElse, 4)
    return newSpace, tmax1, tmax2, t_inf

def computeIdOfFinalCubeInLoop(newSpace, tmax1, tmax2, t_inf): # basicaly does path exists
    """
    Computes coordinates and id (kinda useless) of last cube that created path.
    """

    spaceShape = np.shape(newSpace)
    coords = np.array([[u, v, z] for z in range(spaceShape[2]) for v in range(spaceShape[1]) for u in range(spaceShape[0])]) 
    filts = np.array([newSpace[u, v, z] for z in range(spaceShape[2]) for v in range(spaceShape[1]) for u in range(spaceShape[0])])
        
    cc_density_crater = gd.PeriodicCubicalComplex(
        dimensions = np.shape(newSpace),
        vertices = filts, # top_dimensional_cells esm zamenju z vertices ker... je delal...?
        periodic_dimensions = [1,0,0] # periodično v z dimenziji
        )
    persistence = cc_density_crater.persistence() # izračunaj persistenco
    zadnjiCube = cc_density_crater.vertices_of_persistence_pairs() # vrne pare cubov (birth, death) posameznih komplexov....
    listOfCubeCoords = []
    listOfCubeIds = []
    for p in persistence:
        # print(p)
        if p[0] == 1 and (p[1][1] == float("inf") and p[1][0] < t_inf):
            listOfCubeIds.append(zadnjiCube[1][1][0])
            listOfCubeCoords.append(coords[zadnjiCube[1][1][0]])

        elif p[0]==1 and (tmax1 <= p[1][0] <= tmax2) and  int(p[1][1]) == int(t_inf):
            listOfCubeIds.append(zadnjiCube[0][1][0][0])
            listOfCubeCoords.append(coords[zadnjiCube[0][1][0][0]])
    return [listOfCubeIds, listOfCubeCoords]

def doesPathExists(newSpace, t_inf):
    """
    Checks if path (loop) exists.
    """
    spaceShape = np.shape(newSpace)
    coords = np.array([[u, v, z] for z in range(spaceShape[2]) for v in range(spaceShape[1]) for u in range(spaceShape[0])]) 
    filts = np.array([newSpace[u, v, z] for z in range(spaceShape[2]) for v in range(spaceShape[1]) for u in range(spaceShape[0])])
        
    cc_density_crater = gd.PeriodicCubicalComplex(
        dimensions = np.shape(newSpace),
        vertices = filts, # top_dimensional_cells esm zamenju z vertices ker... je delal...?
        periodic_dimensions = [1,0,0] # periodično v z dimenziji
        )
    persistence = cc_density_crater.persistence() # izračunaj persistenco
    for p in persistence:
        # print(p)
        if p[0] == 1 and (p[1][1] == float("inf") and p[1][0] < t_inf):
            return True
    return False

def getAllIds4(space, rand = False):
    """
    In every time slice it deletes one cube and then checks if path exists. 
    """
    spaceShape = np.shape(space)
    ids = []
    coords = []
    # newSpace, tmax1, tmax2, t_inf = giveSpaceTimeLastSlice(space)
    # listOfCubeIds, listOfCubeCoords = computeIdOfFinalCubeInLoop(newSpace, tmax1, tmax2, t_inf)
    newSpace, t_inf = giveSpaceNoTimeDirection(space)
    exists = doesPathExists(newSpace, t_inf)
    if not exists:
        return np.array(ids), np.array(coords)
    spaceModulated = space.copy()
    idCount = 0

    ycoords = np.arange(0, spaceShape[1])
    xcoords = np.arange(0, spaceShape[2])
    for t in range(spaceShape[0]):
        if rand:
            ycoords = np.random.choice(np.arange(0, spaceShape[1]), size=spaceShape[1], replace=False)
            xcoords = np.random.choice(np.arange(0, spaceShape[2]), size=spaceShape[2], replace=False)
        for y in ycoords:
            for x in xcoords:
                if spaceModulated[t, y, x] == 1:
                    continue
                spaceModulated[t, y, x] = 1
                newSpace, t_inf = giveSpaceNoTimeDirection(spaceModulated)
                exists = doesPathExists(newSpace, t_inf)
                if not exists:
                    ids.append(idCount)
                    coords.append([t, y, x])
                    spaceModulated[t, y, x] = 0
                idCount += 1

    return np.array(ids), np.array(coords)

def findAllPaths(space):
    """
    Finds all paths in space. You know.. creates islands removes islands checks if island was part of path, permutate through
    all islands and finds all paths... that kind of shjiet... Also ignores time traveling paths...hopefully....???
    """
    spaceShape = np.shape(space)

    # Finds all important islands
    newSpace, t_inf = giveSpaceNoTimeDirection(space)
    modulatedSpace = newSpace.copy()
    labeled_matrix_hist = []
    allImportantLs = []
    for tt in range(0, spaceShape[0]):
        labeled_matrix, num_labels = find_islands(~space[tt])
        labeled_matrix_hist.append(labeled_matrix)
        lls = np.array([i for i in range(1, num_labels+1)])
        importantLs = []
        for ord in range(num_labels):
            modulatedSpace = newSpace.copy()
            for l in np.append(lls[ord:], lls[:ord]):
                modulatedSpace[tt][labeled_matrix == l] = 1
                path = doesPathExists(modulatedSpace, t_inf)
                if not path:
                    modulatedSpace[tt][labeled_matrix == l] = 0
                    if l not in importantLs:
                        importantLs.append(l)
        allImportantLs.append(importantLs)


    # Island permutation
    maxVrednosti = [len(allImportantLs[i]) for i in range(spaceShape[0])]
    stevec = [0 for i in range(spaceShape[0])]
    paths = []
    while True:
        modulatedSpace = newSpace.copy()
        for t in range(spaceShape[0]):
            modulatedSpace[t][labeled_matrix_hist[t] != allImportantLs[t][stevec[t]]] = 1
        path = doesPathExists(modulatedSpace, t_inf)
        if path:
            paths.append(stevec.copy())
        stevec[-1] += 1
        for s in range(spaceShape[0] - 1, 0, -1):
            if stevec[s] == maxVrednosti[s]:
                stevec[s] = 0
                stevec[s-1] += 1
        if stevec[0] == maxVrednosti[0]:
            break

    # convert list of island combinations to cube meshes
    cubePathsList = []
    for p in range(len(paths)):
        cubeList = np.empty((0,3))
        for t2 in range(len(paths[p])):
            lmh = np.ravel(labeled_matrix_hist[t2])
            ids = np.where(lmh == (allImportantLs[t2][paths[p][t2]]))[0]
            ycoords = ids//spaceShape[2]
            xcoords = ids%spaceShape[2]
            tcoords = np.ones(len(xcoords))*t2
            cubeList = np.append(cubeList, np.array([tcoords, ycoords, xcoords]).T, axis = 0)
        cubePathsList.append(cubeList)
    if len(cubePathsList) == 0:
        print("Just time traveling, nothing to see here...???")
    return cubePathsList


def findOnePath(space):
    """
    Finds all paths in space. You know.. creates islands removes islands checks if island was part of path, permutate through
    all islands and finds all paths... that kind of shjiet... Also ignores time traveling paths...hopefully....???
    """
    spaceShape = np.shape(space)

    # Finds all important islands
    newSpace, t_inf = giveSpaceNoTimeDirection(space)
    modulatedSpace = newSpace.copy()
    labeled_matrix_hist = []
    allImportantLs = []
    for tt in range(0, spaceShape[0]):
        labeled_matrix, num_labels = find_islands(~space[tt])
        labeled_matrix_hist.append(labeled_matrix)
        lls = np.array([i for i in range(1, num_labels+1)])
        importantLs = []
        for ord in range(num_labels):
            modulatedSpace = newSpace.copy()
            for l in np.append(lls[ord:], lls[:ord]):
                modulatedSpace[tt][labeled_matrix == l] = 1
                path = doesPathExists(modulatedSpace, t_inf)
                if not path:
                    modulatedSpace[tt][labeled_matrix == l] = 0
                    if l not in importantLs:
                        importantLs.append(l)
        allImportantLs.append(importantLs)


    # Island permutation
    maxVrednosti = [len(allImportantLs[i]) for i in range(spaceShape[0])]
    stevec = [0 for i in range(spaceShape[0])]
    paths = []
    while len(paths) == 0:
        modulatedSpace = newSpace.copy()
        for t in range(spaceShape[0]):
            modulatedSpace[t][labeled_matrix_hist[t] != allImportantLs[t][stevec[t]]] = 1
        path = doesPathExists(modulatedSpace, t_inf)
        if path:
            paths.append(stevec.copy())
        stevec[-1] += 1
        for s in range(spaceShape[0] - 1, 0, -1):
            if stevec[s] == maxVrednosti[s]:
                stevec[s] = 0
                stevec[s-1] += 1
        if stevec[0] == maxVrednosti[0]:
            break

    # convert list of island combinations to cube meshes
    cubePathsList = []
    for p in range(len(paths)):
        cubeList = np.empty((0,3))
        for t2 in range(len(paths[p])):
            lmh = np.ravel(labeled_matrix_hist[t2])
            ids = np.where(lmh == (allImportantLs[t2][paths[p][t2]]))[0]
            ycoords = ids//spaceShape[2]
            xcoords = ids%spaceShape[2]
            tcoords = np.ones(len(xcoords))*t2
            cubeList = np.append(cubeList, np.array([tcoords, ycoords, xcoords]).T, axis = 0)
        cubePathsList.append(cubeList)
    if len(cubePathsList) == 0:
        print("Just time traveling, nothing to see here...???")
    return cubePathsList


def narrowPaths(space, onePath = True):
    """
    Combines functions findAllPaths and getAllIds4.
    From findAllPaths gets all paths (that are pretty shit) and reduces them to 1D tube nice paths with getAllIds4.
    Returns all nice paths.
    """
    spaceShape = np.shape(space)
    newSpace, t_inf = giveSpaceNoTimeDirection(space)
    path = doesPathExists(newSpace, t_inf)
    if not path:
        print("No path")
        return []
    # cubePathsList = findAllPaths(space)
    cubePathsList = findAllPaths2(space, onePath)
    narrowPaths = []
    for path in range(len(cubePathsList)):
        space = np.ones((spaceShape), dtype = bool)
        for cube in cubePathsList[path]:
            space[int(cube[0]), int(cube[1]), int(cube[2])] = 0
        ids, specialCoords = getAllIds4(space, True)
        narrowPaths.append(specialCoords)
    return narrowPaths

def doInslandsConnect(i1, i2): # sosednja time slica vsak z le eno grupo, Grupa naj bo označena z 1
    return np.max(i1+i2) > 1

def findAllConnections(space):

    spaceShape = np.shape(space)
    newSpace, t_inf = giveSpaceNoTimeDirection(space)
    labeled_matrix_hist, allImportantLs = labelIslands(space)
    allConnections = []

    for tt in range(spaceShape[0]):
        slice1 = labeled_matrix_hist[tt]
        n_islands1 = np.max(slice1)
        slice2 = labeled_matrix_hist[(tt+1)%spaceShape[0]]
        n_islands2 = np.max(slice2)
        sliceConnections = []
        for i1 in range(1, n_islands1+1):
            for i2 in range(1, n_islands2+1):
                if i1 not in allImportantLs[tt] or i2 not in allImportantLs[(tt+1)%spaceShape[0]]:
                    continue
                island1 = slice1.copy()
                island1[slice1 != i1] = 0
                island1[island1==i1] = 1
                island2 = slice2.copy()
                island2[slice2 != i2] = 0
                island2[island2==i2] = 1

                areConnected = doInslandsConnect(island1, island2)
                if areConnected:
                    sliceConnections.append([i1, i2])
        allConnections.append(np.array(sliceConnections))
    
    return allConnections, labeled_matrix_hist

class Node:
    def __init__(self, time, id):
        self.time = time
        self.id = id
        self.children = []

def getStartingNodes(connections):
    """Groups in first slice are starting nodes"""
    startingNodes = []
    for c in connections[0]:
        startingNodes.append(Node(0, c[0]))
    return startingNodes

def giveNodeChildren(node, connections):
    """Creates node tree from first node"""
    if node.time == len(connections):
        return 
    slice = connections[node.time]
    children = slice[:,1][slice[:,0] == node.id]
    # print(children)
    if len(children) == 0:
        return
    for c in children:
        c_node = Node(node.time+1, c)
        giveNodeChildren(c_node, connections)
        node.children.append(c_node)

def getPath(node, id_start, t_stop):
    # print(node.id, node.time)
    if node.time == t_stop and node.id == id_start:
        return [node.id]
    elif len(node.children) == 0:
        return 0
    for c in node.children:
        path = getPath(c, id_start, t_stop)
        if path == 0:
            continue
        else:
            return [node.id] + path
    return 0

def findAllPaths2(space, onePath = True):
    spaceShape = np.shape(space)
    # We search for all possible connections between two time slices.
    # Two groups are connected when they cover eachother
    allConnections, labeled_matrix_hist = findAllConnections(space)

    n_slices = len(labeled_matrix_hist)
    # now we just find path that comes to the same node id
    # first we build node tree. Every node (group) has children(connected groups)
    startingNodes = getStartingNodes(allConnections)
    for sn in startingNodes:
        giveNodeChildren(sn, allConnections)
    paths = []
    t_stop = len(allConnections)
    for node0 in startingNodes:
        # recursion go brrrrr
        nekej = getPath(node0, node0.id, t_stop)
        if nekej == 0:
            print("Time traveling")
            continue
        paths.append(nekej)
        # when we find first path end 
        if onePath:
            break
    
    # same as before, we get path cube coordinates
    cubePathsList = []
    for p in range(len(paths)):
        cubeList = np.empty((0,3))
        for t2 in range(n_slices):
            lmh = np.ravel(labeled_matrix_hist[t2])
            ids = np.where(lmh == (paths[p][t2]))[0]
            ycoords = ids//spaceShape[2]
            xcoords = ids%spaceShape[2]
            tcoords = np.ones(len(xcoords))*t2
            cubeList = np.append(cubeList, np.array([tcoords, ycoords, xcoords]).T, axis = 0)
        cubePathsList.append(cubeList)
    return cubePathsList



if __name__ == "__main__":
    
    #Testni primeri
    # ################## Test 1 
    # spaceShape = (7,10,9)
    # space = np.ones(spaceShape, dtype=bool)
    # space[:, 2:5, 2:4] = 0
    # space[4:6, 5:6, 5] = 0

    ######################### TEST 2
    spaceShape = (7,10,9)
    space = np.ones(spaceShape, dtype=bool)
    space[0,1,0] = 0
    space[0,1,1] = 0
    space[0,0,0] = 0
    space[1,1:9,0] = 0
    space[2,8,0:7] = 0
    space[3,1:9,6] = 0
    space[3,1,1:7] = 0
    space[4,1:9,1] = 0
    space[4:,8,1] = 0
    space[6,1:8,1] = 0
    space[6,1,0] = 0

    space[:, 4, 7] = 0

    ############################### TEST 3
    # spaceShape = (8,10,9)
    # space = np.ones(spaceShape, dtype=bool)
    # space[0:4, 1, 2] = 0
    # space[4, 1:5, 2] = 0
    # space[0:5, 4, 2] = 0
    # space[1, 4:6, 2] = 0
    # space[1:, 6, 2] = 0
    # space[4, 6:9, 2] = 0
    # space[6, 0:6, 2] = 0
    # space[6:, 1, 2] = 0


    ###################### TEST 4
    # spaceShape = (7,10,9)
    # space = np.ones(spaceShape, dtype=bool)

    # space[3, 1:6, 0] = 0
    # space[3, 1:6, 6] = 0
    # space[3, 1, 0:7] = 0
    # space[3, 6, 0:7] = 0

    # dim = (8, 8)
    # endpoints = [[0, 0], [5, 0]]
    # start_pos = [0, 0]
    # start_dir = [1, 0]
    # sens_1 = DirectionalSensor(dim, endpoints, start_pos, start_dir)
    #
    # endpoints = [[0, 1], [0, 6]]
    # start_pos = [0, 5]
    # start_dir = [0, 1]
    # sens_2 = DirectionalSensor(dim, endpoints, start_pos, start_dir)
    #
    # endpoints = [[2, 2], [4, 2]]
    # start_pos = [4, 2]
    # start_dir = [-1, 0]
    # sens_3 = DirectionalSensor(dim, endpoints, start_pos, start_dir)
    #
    # endpoints = [[2, 4], [4, 4]]
    # start_pos = [3, 4]
    # start_dir = [1, 0]
    # sens_4 = DirectionalSensor(dim, endpoints, start_pos, start_dir)
    #
    # endpoints = [[2, 6], [6, 6]]
    # start_pos = [4, 6]
    # start_dir = [1, 0]
    # sens_5 = DirectionalSensor(dim, endpoints, start_pos, start_dir)
    #
    # endpoints = [[6, 0], [6, 4]]
    # start_pos = [6, 3]
    # start_dir = [0, -1]
    # sens_6 = DirectionalSensor(dim, endpoints, start_pos, start_dir)
    #
    # sensors = [sens_1, sens_2, sens_3, sens_4, sens_5, sens_6]
    # period = smallest_common_multiple([sensor.timePeriod for sensor in sensors])
    # all_nodes, locs = build_node_tree(sensors, period=period, grid_size=8)
    # space = constructSpace(sensors, period, dim=8)
    # print(space.shape)
    # compute all paths
    paths = narrowPaths(space)
    print("Number of paths found: ", len(paths))
    # Kero pot si boš pogledal
    pathId = 0
    if len(paths) > 0:
        specialCoords = paths[pathId]
        #visualize(locs, specialCoords, grid_size=8, fname="test_example.gif")
        #visualize_space(space, specialCoords, fname="test_example.gif")

    # Prikaz poti
    # Prvi plot - najdena pot
    # Drugi plot - Celoten prostor

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
    fig.savefig("output/test_example_path.png", dpi=300)
    # nariši celoten prostor
    plotCubes(space)
    plt.show()