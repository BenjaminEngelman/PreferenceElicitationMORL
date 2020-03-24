from time import time
from numpy.random import RandomState
import numpy as np
from env import BountyfulSeaTreasureEnv, DeepSeaTreasureEnv, RewardWrapper
from agents import QLearningAgent
from solver import Solver
from user import User



seed = 42
weights_list = {
    # Key = w0, value = return
    # 0.0: [5, -1],
    # 0.04: [80, -3],
    # 0.07: [120, -5],
    # 0.15: [140, -7],
    # 0.22: [150, -9],
    # 0.25: [163, -13],
    # 0.28: [166, -14],
    # 0.45: [173, -17],
    [0.05516022, 0.94483978]
}
random_state = RandomState(seed)

# Setup of the environment and agent
n_obj = 2
env = BountyfulSeaTreasureEnv()
n_actions = env.nA
n_states = env.nS

for weight in weights_list:
    weights = np.array([weight, 1-weight])
    agent = QLearningAgent(n_actions=n_actions, n_states=n_states,
                        decay=0.999997, random_state=random_state)
    # Used to train and evaluate the agent on the environements
    solver = Solver(env, agent)
    res = solver.solve(weights)
    print(weight, res)
