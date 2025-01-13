"""
Title: 2D demonstration
Author: Yifei Dong
Date: 14/07/2023
Description: The script demonstrates energy-biased optimal path search in a 2D C-space.
Adapted from https://github.com/ompl/ompl/blob/main/demos/OptimalPlanning.py
"""
import sys
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), '/home/yif/Downloads/omplapp-1.6.0-Source/ompl/py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
import argparse
import matplotlib.pyplot as plt
import numpy as np
# import tikzplotlib

# Hyperparameters
useGoalSpace = 0
useIncrementalCost = 0
runtime = 3.0
planner = 'BITstar' # 'BITstar'
runSingle = 0
gamma = 0.0

class ValidityChecker(ob.StateValidityChecker):
    def isValid(self, state):
        bools = [self.clearance(state, rads[i], centers[i])>0.0 for i in range(len(rads))]
        return bools.count(False) == 0

    def clearance(self, state, radius=[.2,.2], center=[.5, 0]):
        x, y = state[0], state[1]
        xc, yc = center[0], center[1]
        return (x-xc)**2/radius[0]**2 + (y-yc)**2/radius[1]**2 - 1

class minPathPotentialObjective(ob.OptimizationObjective):
    def __init__(self, si, start, useIncrementalCost):
        super(minPathPotentialObjective, self).__init__(si)
        self.si_ = si
        self.start_ = start
        self.useIncrementalCost_ = useIncrementalCost

    def combineCosts(self, c1, c2):
        if self.useIncrementalCost_:
            return ob.Cost(c1.value() + c2.value())
        else:
            return ob.Cost(max(c1.value(), c2.value()))

    def motionCost(self, s1, s2):
        if self.useIncrementalCost_:
            return ob.Cost(abs(s2[1] - s1[1]))
        else:
            return ob.Cost(s2[1] - self.start_[1])

def getPotentialObjective(si, start, useIncrementalCost):
    obj = minPathPotentialObjective(si, start, useIncrementalCost)
    return obj

def getThresholdPathLengthObj(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1.51))
    return obj

def getBalancedObjective(si, start, useIncrementalCost):
    lengthObj = ob.PathLengthOptimizationObjective(si)
    potentialObj = minPathPotentialObjective(si, start, useIncrementalCost)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, gamma)
    opt.addObjective(potentialObj, 1)
    return opt

def getPathLengthObjective(si):
    return ob.PathLengthOptimizationObjective(si)

def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj

def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        planner = og.BITstar(si)
        planner.params().setParam("rewire_factor", "0.5")
        planner.params().setParam("samples_per_batch", "2000")
        return planner
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        planner = og.InformedRRTstar(si)
        # planner.params().setParam("range", "0.01") # controls the maximum distance between a new state and its nearest neighbor in the tree (for max potential gain)
        # planner.params().setParam("rewire_factor", "0.01") # controls the radius of the ball used during the rewiring phase (for max potential gain)
        return planner
    elif plannerType.lower() == "prmstar":
        planner = og.PRMstar(si)
        # planner = og.PRM(si)
        return planner
    elif plannerType.lower() == "rrtstar":
        planner = og.RRTstar(si)
        planner.params().setParam("range", "0.01")
        planner.params().setParam("rewire_factor", "0.01")
        return planner
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    elif plannerType.lower() == "lbtrrt":
        planner = og.LBTRRT(si)
        # epsilon: smaller values of epsilon tend to explore the space more thoroughly but can be slower, while larger values of epsilon tend to be faster
        planner.params().setParam("epsilon", "0.01")
        return planner
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

def allocateObjective(si, objectiveType, start, useIncrementalCost):
    if objectiveType.lower() == "pathpotential":
        return getPotentialObjective(si, start, useIncrementalCost)
    elif objectiveType.lower() == "pathlength":
        return getPathLengthObjWithCostToGo(si)
    elif objectiveType.lower() == "thresholdpathlength":
        return getThresholdPathLengthObj(si)
    elif objectiveType.lower() == "weightedlengthandpotential":
        return getBalancedObjective(si, start, useIncrementalCost)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")

def state_to_list(state):
    return [state[i] for i in range(2)]

def plot_ellipse(center, radius, ax):
    u = center[0]    # x-position of the center
    v = center[1]    # y-position of the center
    a = radius[0]    # radius on the x-axis
    b = radius[1]    # radius on the y-axis

    t = np.linspace(0, 2*np.pi, 100)
    x = u + a * np.cos(t)
    y = v + b * np.sin(t)

    ax.fill(x, y, alpha=0.7, color='#f4a63e')
    ax.plot(x, y, alpha=0.7, color='#f4a63e')

def plot(sol_path_list, cost):
    fig, ax = plt.subplots()
    if sol_path_list is not None:
        # Plot the solution path
        xpath = [state[0] for state in sol_path_list]
        ypath = [state[1] for state in sol_path_list]
        ax.plot(xpath, ypath, color='#31a354')
        ax.scatter(xpath[0], ypath[0], color='r', s=100, label='Start')

    # create a circle object
    for i in range(len(rads)):
        plot_ellipse(centers[i], rads[i], ax)

    # set axis limits and aspect ratio
    if cost is not None:
        ax.text(0.02, 0.98, '$\\overline{{C}}(\\sigma^*)={:.2f}$'.format(cost), transform=ax.transAxes, verticalalignment='top', fontsize=17)
    ax.set_xlim([0, 1.])
    ax.set_ylim([0, 1.])
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(direction='in', length=6, width=1, colors='k', grid_color='k', grid_alpha=0.5, labelsize=10)
    # ax.set_title('Escape path - objective of lowest incremental potential energy gain')
    plt.savefig("png/escape-id-{:04d}.png".format(j), dpi=200)
    # tikzplotlib.save("gamma-{}-{}.tex".format(gamma, j))
    # plt.show()

def plan(runTime, plannerType, objectiveType, fname, start_pos, goal_pos, useIncrementalCost, visualize=0):
    # Construct the robot state space in which we're planning. We're
    # planning in [0,1]x[0,1], a subset of R^2.
    space = ob.RealVectorStateSpace(2)

    # Set the bounds of space to be in [0,1].
    space.setBounds(0.0, 1.0)

    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)
    si.setup()

    # Set the object used to check which states in the space are valid
    validityChecker = ValidityChecker(si)
    si.setStateValidityChecker(validityChecker)
    si.setup()

    # TODO: start and goal
    start = ob.State(space)
    start[0], start[1] = start_pos[0], start_pos[1]

    goal = ob.State(space)
    goal[0], goal[1] = goal_pos[0], goal_pos[1]

    # Energy of start and goal
    Es, Eg = start[1], goal[1]

    # Create a problem instance
    pdef = ob.ProblemDefinition(si)

    # Set the start and goal states
    if useGoalSpace:
        # GoalSpace works with RRT*/PRM*/InformedRRT*, not with BIT*
        goal_space = ob.GoalSpace(si)
        s = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, .8)
        bounds.setHigh(0, 1.)
        bounds.setLow(1, .0)
        bounds.setHigh(1, .1)
        s.setBounds(bounds)
        goal_space.setSpace(s)

        # set start and goal
        pdef.addStartState(start)
        # pdef.setGoalState(goal, threshold)
        pdef.setGoal(goal_space)
    else:
        threshold = .001 # TODO: does not seem to impact anything now
        pdef.setStartAndGoalStates(start, goal, threshold)

    # Create the optimization objective specified by our command-line argument.
    pdef.setOptimizationObjective(allocateObjective(si, objectiveType, start, useIncrementalCost))

    # Construct the optimal planner specified by our command line argument.
    optimizingPlanner = allocatePlanner(si, plannerType)
    print(optimizingPlanner.params())

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # Attempt to solve the planning problem in the given runtime
    solved = optimizingPlanner.solve(runTime)

    if solved:
        # Output the length of the path found
        sol_path_geometric = pdef.getSolutionPath()
        objValue = sol_path_geometric.cost(pdef.getOptimizationObjective()).value()
        # sumEnergyGain = (objValue - (Es-Eg)) / 2
        pathLength = sol_path_geometric.length()
        sol_path_states = sol_path_geometric.getStates()
        sol_path_list = [state_to_list(state) for state in sol_path_states]
        sol_path_ys = [state[1] for state in sol_path_states]
        pathPotentialCost = max(sol_path_ys) - start_pos[1]
        totalCost = gamma*pathLength + pathPotentialCost
        cost = totalCost
        
        print("pathPotentialCost: ", pathPotentialCost)
        print("pathLengthCost: ", pathLength)
        # print('Normalized cost, c_bar = gamma*pathLengthCost/referencePotentialCost + 1: ', cost)
        print('{0} found solution of path length {1:.4f} with an optimization ' \
            'objective value of {2:.4f}'.format( \
            optimizingPlanner.getName(), \
            pathLength, \
            objValue))
        
        # plot the map and path
        # if visualize:
        #     plot(sol_path_list, totalCost)

        if fname:
            with open(fname, 'w') as outFile:
                outFile.write(pdef.getSolutionPath().printAsMatrix())
    else:
        print("No solution found.")
        # if visualize:
        #     plot(None, None)
        return None, None
    
    print('===================================')
    return pathPotentialCost, sol_path_list

# Function to generate random ellipsoid dimensions and positions forming a U-shape
def generate_u_shape():
    # Randomize side dimensions of the U-shape
    left_height = np.random.uniform(0.1, 0.4)
    right_height = np.random.uniform(0.1, 0.4)
    bottom_width = np.random.uniform(0.2, 0.4)

    # Randomize radii of ellipsoids
    rads = np.zeros((3, 2))
    rads[0] = [np.random.uniform(0.05, 0.1), left_height]  # Left side
    rads[1] = [bottom_width, np.random.uniform(0.05, 0.1)]  # Bottom side
    rads[2] = [np.random.uniform(0.05, 0.1), right_height]  # Right side

    # Randomize positions of ellipsoids
    centers = np.zeros((3, 2))
    centers[0] = [np.random.uniform(0.1, 0.2), np.random.uniform(0.4, 0.5)]  # Left side
    centers[1] = [0.5, np.random.uniform(0.4, 0.5)]  # Bottom side
    centers[2] = [np.random.uniform(0.8, 0.9), np.random.uniform(0.4, 0.5)]  # Right side

    return centers, rads

def post_process_path(sol_path_list, pathPotentialCost):
    """Cut the path segment once it reaches below y=centers[1][1] - rads[1][1]"""
    for i in range(len(sol_path_list)):
        if sol_path_list[i][1] < centers[1][1] - rads[1][1]:
            sol_path_list = sol_path_list[:i]
            break
    sol_path_list = downsample_path(sol_path_list, pathPotentialCost, num_points=20)
    plot(sol_path_list, pathPotentialCost)

    return sol_path_list

from scipy.interpolate import CubicSpline
def downsample_path(path, cost, num_points=20):
    """
    Downsample a 2D path to `num_points` with smooth interpolation.
    
    Args:
        path (list of list): Original path as a list of [x, y] points.
        num_points (int): Number of points in the downsampled path.
    
    Returns:
        list of list: Downsampled smooth path with `num_points` points.
    """
    path = np.array(path)
    x, y = path[:, 0], path[:, 1]

    # Parameterize the path using cumulative distances
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_distances = np.hstack(([0], np.cumsum(distances)))

    # Create a cubic spline for each coordinate
    cs_x = CubicSpline(cumulative_distances, x)
    cs_y = CubicSpline(cumulative_distances, y)

    # Create evenly spaced points along the path
    uniform_distances = np.linspace(0, cumulative_distances[-1], num_points)
    smooth_x = cs_x(uniform_distances)
    smooth_y = cs_y(uniform_distances)

    # Combine x and y into the downsampled path
    downsampled_path = np.vstack((smooth_x, smooth_y)).T
    return downsampled_path

import joblib
def save_dataset(filename, dataset):
    """Save the dataset to a file."""
    # Save the object to a joblib file
    joblib.dump(dataset, filename)

    print(f"Data has been saved to {filename}")

    # Optionally, load the object back to verify
    loaded_data = joblib.load(filename)
    # print("Loaded Data:", loaded_data)
    print(loaded_data.keys())

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Optimal motion planning demo program.')

    # Add a filename argument
    parser.add_argument('-t', '--runtime', type=float, default=runtime, help=\
        '(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.')
    parser.add_argument('-p', '--planner', default=planner, \
        choices=['LBTRRT', 'BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', \
        'SORRTstar'], \
        help='(Optional) Specify the optimal planner to use, defaults to RRTstar if not given.')
    parser.add_argument('-o', '--objective', default='PathPotential', \
        choices=['PathPotential', 'PathLength', 'ThresholdPathLength', \
        'WeightedLengthAndPotential'], \
        help='(Optional) Specify the optimization objective, defaults to PathLength if not given.')
    parser.add_argument('-f', '--file', default=None, \
        help='(Optional) Specify an output path for the found solution path.')
    parser.add_argument('-i', '--info', type=int, default=1, choices=[0, 1, 2], \
        help='(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG.' \
        ' Defaults to WARN.')

    # Parse the arguments
    args = parser.parse_args()

    # Check that time is positive
    if args.runtime <= 0:
        raise argparse.ArgumentTypeError(
            "argument -t/--runtime: invalid choice: %r (choose a positive number greater than 0)" \
            % (args.runtime,))

    # Set the log level
    if args.info == 0:
        ou.setLogLevel(ou.LOG_WARN)
    elif args.info == 1:
        ou.setLogLevel(ou.LOG_INFO)
    elif args.info == 2:
        ou.setLogLevel(ou.LOG_DEBUG)
    else:
        ou.OMPL_ERROR("Invalid log-level integer.")

    # Solve the planning problem
    num_envs = 2
    costs = []
    paths = []
    ellipse_centers = []
    ellipse_radii = []
    object_starts = []
    for j in range(num_envs):
        # Generate U-shape configuration
        centers, rads = generate_u_shape()

        # Randomize the "start" position inside the U-shape
        start_x = np.random.uniform(centers[0][0] + rads[0][0], centers[2][0] - rads[2][0])
        start_y = np.random.uniform(centers[1][1] + rads[1][1], 0.7)
        start_pos = (start_x, start_y)
        goal_pos = (0.5, 0.0)

        pathPotentialCost, sol_path_list = plan(args.runtime, args.planner, args.objective, args.file, start_pos, goal_pos, useIncrementalCost, visualize=1)
        if pathPotentialCost is not None:
            sol_path_list = post_process_path(sol_path_list, pathPotentialCost)
            costs.append(pathPotentialCost)
            paths.append(sol_path_list)
            object_starts.append(start_pos)
            ellipse_centers.append(centers)
            ellipse_radii.append(rads)
    dataset = {"costs": np.array(costs), 
               "paths": np.array(paths), 
               "object_starts": np.array(object_starts), 
               "ellipse_centers": np.array(ellipse_centers), 
               "ellipse_radii": np.array(ellipse_radii)}
    save_dataset(f"dataset_escape_from_u_2d_{num_envs}_envs.joblib", dataset)
