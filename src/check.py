from time import time
from numpy.random import RandomState
import numpy as np
from src.env import BountyfulSeaTreasureEnv
from src.agents import MOQlearning
from src.solver import Solver
from src.user import User



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
    
# }
weights_list = [
    # [0.5, 0.5],
    [0.84340659, 0.15659341],
]
random_state = RandomState(seed)

# Setup of the environment and agent
n_obj = 2
env = BountyfulSeaTreasureEnv()

for weights in weights_list:
    # weights = np.array([weight, 1-weight])
    agent = MOQlearning(env, decay=0.999997, random_state=random_state)
    # Used to train and evaluate the agent on the environements
    solver = Solver()
    res = solver.solve(agent, np.array(weights))
    print(weights, res)
