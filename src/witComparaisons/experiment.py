import sys
sys.path.insert(0, '..')

from time import time
from numpy.random import RandomState
import numpy as np

from env import BountyfulSeaTreasureEnv, DeepSeaTreasureEnv, RewardWrapper
from agents import QLearningAgent
from solver import Solver
from user import User
import matplotlib.pyplot as plt





def main(method="opti"):
    seed = 42
    random_state = RandomState(seed)

    # Setup of the environment and agent
    n_obj = 2
    env = BountyfulSeaTreasureEnv()
    n_actions = env.nA
    n_states = env.nS
    agent = QLearningAgent(n_actions=n_actions, n_states=n_states, decay=0.999997, random_state=random_state)
    solver = Solver(env, agent) # Used to train and evaluate the agent on the environements
    history = []

    # Create a user that will return noisy comparisons
    user = User(num_objectives=n_obj, std_noise=1, random_state=random_state)
    print("Hidden weights: " + str(user.hidden_weights) +"\n")

    # Data points seen 
    # X contains the returns obtained so far
    # y contains the noisy user utility for those returns 
    X, y = [], []

    # We start with an estimate of the weights weights
    weights = np.ones(n_obj) / n_obj
    history.append(weights[0])
    prev_weights = weights + 1

    # When the difference between two successive estimated weights is below eps. we stop
    epsilon = 1e-3

    it = 0
    while it < 15:# or not (np.abs(prev_weights - weights)<epsilon).all():
        print("Iteration %d" % it)
        print("Current weights estimates :" + str(weights))
        # Solve the environement for the random weights
        w0_to_solve = random_state.uniform(0, 1)
        w_solve = np.array([w0_to_solve, 1 - w0_to_solve])
        returns = solver.solve(w_solve)
        print("Returns for current weights: " + str(returns))

        # Get a noisy estimate of the user utility
        u = user.get_utility(returns)
        print("Utility of the user: " + str(u) + "\n")

        # Add those to our dataset
        X.append(returns)
        y.append(u)

        # Estimate new weights
        prev_weights = weights
        if method == "opti":
            weights = estimateWeightsOpti(X, y, current_estimate=weights)

        elif method == "reg":
            weights = estimateWeightsReg(X, y)
        else:
            print("Wrong method.")
        
        history.append(weights[0])

        it += 1
    
    f, ax = plt.subplots()
    ax.plot(history,  marker='o')
    ax.hlines(user.hidden_weights[0], xmin=0, xmax=len(history))
    # ax.set_yticks(list(np.arange(0, 1, 0.1)))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("w0 estimate")
    plt.savefig(f"figures/exp_method_{method}.png")
