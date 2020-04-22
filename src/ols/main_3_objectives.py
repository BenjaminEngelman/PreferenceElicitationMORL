from recordclass import recordclass
from time import time
import numpy as np
import queue
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from itertools import combinations 
from src.utils import get_best_sol
from src.ols.utils import create_3D_pareto_front, plot_3D_pareto_front, compare_3D_pareto_fronts
from src.solver import Solver



MIN_IMPROVEMENT = 0.0

# A CornerWeight contains: - the value (w0, w1) of the 2 first weights
#                          - its maximum possible improvement
CornerWeight = recordclass("CornerWeight", ["w0", "w1", "improvement"])



def triangle_for_point(r):
    r = np.array(r)
    w = np.identity(len(r)).T
    u = np.inner(w,r)

    triang = tri.Triangulation(w[:,0], w[:,1], triangles=np.arange(3)[None,:])
    return triang, u

def scalarize(V_PI, weights):
    return np.dot(weights, V_PI)


def intersection(points):
    p = np.array(points)
    # r1*w1 + r2*w2 + r3*(1-w1-w2) = u. Unknowns: w1, w2, u
    a = np.array((p[:,0]-p[:,2], p[:,1]-p[:,2], -np.ones(p.shape[0]))).T
    b = -p[:,2]
    x = np.linalg.solve(a, b)
    return x


def newCornerWeights(V_PI, S, W):
    """
    V_PI : The new policy (list of 3 scalars)
    S: The policies contained in the current approximation of the CCS
    W: The weights for which we already have executed the sub-routine
    """

    cornerWeights = []

    # Need a least 3 Policies in S
    if len(S) < 2:
        return cornerWeights

    combs = combinations(S, 2) 

    # Compute the intersection of V_PI with all the combinations of 2 policies in S
    for comb in combs:
        w0, w1, U = intersection([comb[0], comb[1], V_PI])

        # If the intersection has already been used
        # Don't keep it
        intersec_w = np.array([w0, w1, 1-w0-w1])
        if list(intersec_w) in W or w0 < 0 or w0 > 1 or w1 < 0 or w1 > 1:
            continue
        
        # Check if the intersection is higher/lower than the current CCS
        # We can only add the intersection if there is no policy above it
        can_add_intersec = True
        for V in S:
            if V not in comb:
                if scalarize(V, intersec_w) > U:
                    can_add_intersec = False

        if can_add_intersec:
            # Compute the maximum improvement
            ceiling_plane = [
                max(comb[0][0], comb[1][0], V_PI[0]),
                max(comb[0][1], comb[1][1], V_PI[1]),
                max(comb[0][2], comb[1][2], V_PI[2]),
            ]
            u_ceiling_plane = scalarize(ceiling_plane, intersec_w)
            improv = u_ceiling_plane - U

            cornerWeights.append(CornerWeight(w0=w0, w1=w1, improvement=improv))

    return cornerWeights


def ols(S=[], W=[], Q=queue.PriorityQueue(), num_iter=0, run_name="ols_run"):
    # pareto_front = create_3D_pareto_front(size=10, seed=0)
    solver = Solver()
    start = time()

    # Add the two extremum values for the weights in the Queue
    # With infinite priority
    if num_iter == 0:
        Q.put((-math.inf, CornerWeight(w0=1.0, w1=0.0, improvement=-math.inf)))
        Q.put((-math.inf, CornerWeight(w0=0.0, w1=1.0, improvement=-math.inf)))
        Q.put((-math.inf, CornerWeight(w0=0.0, w1=0.0, improvement=-math.inf)))


    while not Q.empty():
        print("\nITERATION: %d" % num_iter)
        print(Q.queue)
        # Get corner weight from Queue
        weight = Q.get()
        w_values = weight[1] # weights[0] is the priority, [1] is the cornerweight obj

        # Call solver with this weight
        w = np.array([w_values.w0, w_values.w1, 1 - w_values.w0 - w_values.w1])

        # print("Solving for weights: ", w)
        W.append(list(w))
        obj1, obj2, obj3 = solver.solve("minecart", w)
        
        # obj1, obj2, obj3 = get_best_sol(pareto_front, w)
        V_PI = [obj1, obj2, obj3]

        print(f"RETURNS : {V_PI}")

        if V_PI not in S: #and hasImprovement(w_values, V_PI, S):

            # Find new cornerweights
            W_V_PI = newCornerWeights(V_PI, S, W)
            
            S.append(V_PI)
            for cornerWeight in W_V_PI:

                if cornerWeight.improvement > MIN_IMPROVEMENT:
                    # priority = -improvement because high priority = small number
                    Q.put((-cornerWeight.improvement, cornerWeight))

        num_iter += 1

        # Save S, Q, W, num_iter
        with open(f'runs_ols/{run_name}/S.pkl', 'wb') as handle:
            pickle.dump(S, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'runs_ols/{run_name}/Q.pkl', 'wb') as handle:
            pickle.dump(Q.queue, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'runs_ols/{run_name}/W.pkl', 'wb') as handle:
            pickle.dump(W, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'runs_ols/{run_name}/num_iter.pkl', 'wb') as handle:
            pickle.dump(num_iter, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(Q.queue)

    total_time = time() - start

    print("Number of iterations: %d" % num_iter)
    print("Time (s): %.2f" % total_time)
    # compare_3D_pareto_fronts(pareto_front, np.array(S))
    

if __name__ == "__main__":
    import os
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir('runs_ols'):
        os.mkdir("runs_ols")
    
    if not args.resume:
        if not os.path.isdir(f'runs_ols/{args.name}'):
            os.mkdir(f'runs_ols/{args.name}')
        ols(run_name=args.name)
    
    else:
        path = f'runs_ols/{args.name}'
        S = pickle.load(open(f'{path}/S.pkl', 'rb'))
        W = pickle.load(open(f'{path}/W.pkl', 'rb'))
        num_iter = pickle.load(open(f'{path}/num_iter.pkl', 'rb'))
        # Hack because we  cannot pickle PrioritiQueue
        # We picke its its internal queue which is a list
        _queue = pickle.load(open(f'{path}/Q.pkl', 'rb'))
        Q = queue.PriorityQueue()
        Q.queue = _queue
    
        ols(S=S, W=W, Q=Q, num_iter=num_iter, run_name=args.name)
