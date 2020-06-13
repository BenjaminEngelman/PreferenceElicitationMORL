from time import time
from numpy.random import RandomState
import numpy as np
from src.solver import Solver
from src.utils import get_best_sol_BST
from src.constants import WEIGHTS_COMP_BST
import pickle
seed = 42
# weights_list = {
#     # Key = w0, value = return
#     # 0.0: [5, -1],
#     # 0.04: [80, -3],
#     # 0.07: [120, -5],
#     # 0.15: [140, -7],
#     # 0.22: [150, -9],
#     # 0.25: [163, -13],
#     # 0.28: [166, -14],
#     # 0.45: [173, -17],
#     [0.4497, 0.5503],
#     [0.3452, 0.6548],
    

random_state = RandomState(seed)

# Setup of the environment and agent
n_obj = 2
solver = Solver()

all_weights = random_state.uniform(0, 1, (100, 2))
all_weights /= all_weights.sum(axis=1)[:,None]

res = {}
for i, weights in enumerate(all_weights):#WEIGHTS_COMP_BST:
    weights = np.array(weights)
    optimal_returns = get_best_sol_BST(weights)
    returns = solver.solve("bst", weights, random_state=random_state)

    res[i] = {"opt": optimal_returns, "ret": returns}

    with open(f'experiments/bst/results.pickle', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    