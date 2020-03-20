import sys
sys.path.insert(0, '..')

from user import User
from solver import Solver
from agents import QLearningAgent
from env import BountyfulSeaTreasureEnv, DeepSeaTreasureEnv, RewardWrapper
import numpy as np
from numpy.random import RandomState
from time import time

seed = 42
weights = np.array([0.43418691, 0.56581309])
random_state = RandomState(seed)

# Setup of the environment and agent
n_obj = 2
env = BountyfulSeaTreasureEnv()
n_actions = env.nA
n_states = env.nS
agent = QLearningAgent(n_actions=n_actions, n_states=n_states,
                        decay=0.999997, random_state=random_state)
# Used to train and evaluate the agent on the environements
solver = Solver(env, agent)
res = solver.solve(weights)
print(res)


