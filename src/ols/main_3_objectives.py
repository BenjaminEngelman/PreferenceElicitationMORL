# import sys
# sys.path.insert(0, '..')

from src.utils import plot_ccs, plot_ccs_2
from src.env import BountyfulSeaTreasureEnv, DeepSeaTreasureEnv
from src.agents import QLearningAgent
from src.solver import Solver
from recordclass import recordclass
from time import time
import numpy as np
import queue
import math



##########################################
# TODO :

##########################################

MIN_IMPROVEMENT = 0.1

# A V_P contains: - the value of a policy (for each objective)
#                 - the range for which in the current CCS
V_P = recordclass("V_P", ["obj1", "obj2", "obj3" "start_w0", "end_w0", "start_w1", "end_w1"])

# A CornerWeight contains: - the value (w0, w1) of the 2 first weights
#                          - its maximum possible improvement
CornerWeight = recordclass("CornerWeight", ["val", "improvement"])

Point = recordclass("Point", ["x", "y", "z"])


def scalarize(V_PI, w0, w1):
    w2 = 1 - w1 - w0
    return w0 * V_PI.obj1 + w1 * V_PI.obj2 + w2 * V_PI.obj3


def intersection(points):
    p = np.array(points)
    # r1*w1 + r2*w2 + r3*(1-w1-w2) = u. Unknowns: w1, w2, u
    a = np.array((p[:,0]-p[:,2], p[:,1]-p[:,2], -np.ones(p.shape[0]))).T
    b = -p[:,2]
    x = np.linalg.solve(a, b)
    return x


def hasImprovement(w1, V_PI, S):
    if len(S) == 0 or w1.improvement == -math.inf:
        return True

    currentHeight = None
    for V in S:
        if V.start.x == w1.val:
            currentHeight = V.start.y
            break
    x, y = intersection(Point(w1.val, 0), Point(w1.val, currentHeight), Point(0, V_PI.obj1), Point(1, V_PI.obj2))
    if y > currentHeight:
        return True
    else:
        return False


def removeObseleteValueVectors(V_PI, S):
    for V in S:
        s = V.start.x
        e = V.end.x

        if (scalarize(V_PI, s) > scalarize(V, s)) and (scalarize(V_PI, e) > scalarize(V, e)):
            S.remove(V)


def removeObseleteWeights(Q, s, e):
    for item in Q.queue:
        # each item is a tupe (priority, cornerweight)
        cornerWeight = item[1]
        if (s < cornerWeight.val < e) and (cornerWeight.improvement > -math.inf):
            Q.queue.remove(item)


def newCornerWeights(V_PI, S):
    """
    Find the new corner weights by computing the intersection
    between V_PI with all the Vs in the Partial CCS (S)
    """
    cornerWeights = []

    for V in S:
        w0, w1, U = intersection(V_PI.start, V_PI.end, V.start, V.end)
        # If intersection is in the range of line (p1, p2) and (p3, p4)
        if not (cornerW > V_PI.end.x or cornerW < V_PI.start.x or cornerW > V.end.x or cornerW < V.start.x):
            if V_PI.obj1 > V.obj1:
                V.start.x = cornerW
                V.start.y = Y
                V_PI.end.x = cornerW
                V_PI.end.y = Y
            else:
                V.end.x = cornerW
                V.end.y = Y
                V_PI.start.x = cornerW
                V_PI.start.y = Y

            cornerWeights.append(CornerWeight(val=cornerW, improvement=None))

    return cornerWeights


def estimateImprovement(cornerWeight, S):
    """
    Compute the maximum improvement given the current estimation of the CSS and
    a corner weight
    """
    for V in S:
        if V.start.x == cornerWeight:
            lastPoint = V.end
            cornerPoint = V.start
        elif V.end.x == cornerWeight:
            firstPoint = V.start
    _, height = intersection(firstPoint, lastPoint,
                             Point(cornerWeight, 1000), cornerPoint)
    return height - cornerPoint.y


def ols(env, agent):
    start = time()


    solver = Solver(env, agent)

    S = []  # Partial CCS
    W = []  # Visited weights
    Q = queue.PriorityQueue()  # To prioritize the weights

    # Add the two extremum values for the weights in the Queue
    # With infinite priority
    Q.put((-math.inf, CornerWeight(val=0.0, improvement=-math.inf)))
    Q.put((-math.inf, CornerWeight(val=1.0, improvement=-math.inf)))

    num_iter = 0
    while not Q.empty():
        print("\nITERATION: %d" % num_iter)
        print(Q.queue)
        # Get corner weight from Queue
        weight = Q.get()
        w1 = weight[1]
        W.append(w1.val)

        # Call solver with this weight
        w = np.array([1 - w1.val, w1.val])
        print("Solving for weights: ", w)
        obj1, obj2 = solver.solve(w)
        # Get V_PI from solver
        V_PI = V_P(obj1=obj1, obj2=obj2, start=Point(
            x=0.0, y=obj1), end=Point(x=1.0, y=obj2))
        print(V_PI)

        W.append(w1.val)
        print(f"W1: {w1}\nV_PI: {V_PI}\nS: {S}")
        if V_PI not in S and hasImprovement(w1, V_PI, S):
            # Remove obseletes Vs from S
            removeObseleteValueVectors(V_PI, S)

            # plot CSS + New value vector
            plot_ccs_2(S + [V_PI])

            # Find new cornerweights
            W_V_PI = newCornerWeights(V_PI, S)
            
            S.append(V_PI)
            removeObseleteWeights(Q, V_PI.start.x, V_PI.end.x)
            for cornerWeight in W_V_PI:
                cornerWeight.improvement = estimateImprovement(
                    cornerWeight.val, S)
                if cornerWeight.improvement > MIN_IMPROVEMENT and cornerWeight.val not in W:
                    # priority = -improvement because high priority = small number
                    Q.put((-cornerWeight.improvement, cornerWeight))

        num_iter += 1

    total_time = time() - start
    print("Number of iterations: %d" % num_iter)
    print("Time (s): %.2f" % total_time)
    plot_ccs_2(S)