import math
import queue
import numpy as np
from time import time
from recordclass import recordclass
from solver import Solver
from agents import QLearningAgent
from env import DeepSeaTreasure
from utils import plot_ccs

##########################################
# TODO :

##########################################

MIN_IMPROVEMENT = 0.01

# A V_P contains: - the value of a policy (for each objective)
#                 - the range for which in the current CCS
V_P = recordclass("V_P", ["obj1", "obj2", "start", "end"])

# A CornerWeight contains: - the value (w1) of the weight
#                          - its maximum possible improvement
CornerWeight = recordclass("CornerWeight", ["val", "improvement"])

Point = recordclass("Point", ["x", "y"])

def scalarize(V_PI, w1):
    w0 = 1 - w1
    return w0 * V_PI.obj1 + w1 + V_PI.obj2

def intersection(p1, p2, p3, p4):
    """
    Compute the intersection of the lines defined as
    (p1, p2) and (p2, p3)
    """
    numX = (p1.x*p2.y-p1.y*p2.x)*(p3.x-p4.x)-(p1.x-p2.x)*(p3.x*p4.y-p3.y*p4.x)
    numY = (p1.x*p2.y-p1.y*p2.x)*(p3.y-p4.y)-(p1.y-p2.y)*(p3.x*p4.y-p3.y*p4.x)
    denum = (p1.x-p2.x)*(p3.y-p4.y)-(p1.y-p2.y)*(p3.x-p4.x)

    px = numX / denum
    py = numY / denum
    return Point(px, py)

def hasImprovement(w1, V, S):
    if len(S) == 0 or w1.improvement == -math.inf:
        return True

    currentHeight = None
    for V in S:
        if V.start.x == w1:
            currentHeight = V.start.y
            break
    x, y = intersection(Point(w1, 0), Point(w1, currentHeight), Point(0, V_PI.obj1), Point(1, V_PI.obj2))
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
    for cornerWeight in Q:
        if (s < cornerWeight.val < e) and (cornerWeight.improvement > -math.inf):
            Q.remove(cornerWeight)

def newCornerWeights(V_PI, S):
    """
    Find the new corner weights by computing the intersection
    between V_PI with all the Vs in the Partial CCS (S)
    """
    cornerWeights = []

    for V in S:
        cornerW, Y = intersection(V_PI.start, V_PI.end, V.start, V.end)
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
    _, height = intersection(firstPoint, lastPoint, Point(cornerWeight, 1000), cornerPoint)
    return height - cornerPoint.y


if __name__ == "__main__":

    start = time()

    env = DeepSeaTreasure()
    n_actions = env.action_spec[2][1]
    agent = QLearningAgent(epsilon=0.7, n_actions=n_actions, n_states=100)
    solver = Solver(env, agent)

    S = set()  # Partial CCS
    W = set()  # Visited weights
    Q = queue.PriorityQueue()  # To prioritize the weights

    # Add the two extremum values for the weights in the Queue
    # With infinite priority
    Q.put((-math.inf, CornerWeight(val=0, improvement=-math.inf)))
    Q.put((-math.inf, CornerWeight(val=1, improvement=-math.inf)))

    num_iter = 0 
    while not Q.empty():
        # Get corner weight from Queue
        w1 = Q.get()[1]
        # Call solver with this weight
        w = np.array([1 - w1.val, w1.val])
        obj1, obj2 = solver.solve(w)
        # Get V_PI from solver
        V_PI = V_P(obj1=obj1, obj2=obj2, start=Point(x=0, y=obj1), end=Point(x=1, y=obj2))

        W.add(w)

        if V_PI not in S: #and hasImprovement(w1, V_PI, S)
            # Remove obseletes Vs from S
            removeObseleteValueVectors(V_PI, S)

            # Find new cornerweights
            W_V_PI = newCornerWeights(V_PI, S)
            S.add(V_PI)
            removeObseleteWeights(Q, V_PI.start, V_PI.end)

            for cornerWeight in W_V_PI:
                cornerWeight.improvement = estimateImprovement(cornerWeight.val, S)
                if cornerWeight.improvement > MIN_IMPROVEMENT:
                    # priority = -improvement because high priority = small number
                    Q.put((-cornerWeight.improvement, cornerWeight.val))
        
        num_iter += 1
    

    total_time = time() - start
    print("Number of iterations: %d" % num_iter)
    print("Time (s): total_time")
    plot_ccs(S)