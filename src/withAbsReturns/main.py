import sys
sys.path.insert(0, '..')

from time import time
from numpy.random import RandomState
import numpy as np

from env import BountyfulSeaTreasureEnv, DeepSeaTreasureEnv, RewardWrapper
from agents import QLearningAgent
from solver import Solver
from user import User

from scipy.optimize import minimize, lsq_linear
import matplotlib.pyplot as plt


def estimateWeightsOpti(X, y, current_estimate=None):
    """
    Estimates the weights minimizing the sum of squared errors
    with some constrains on the weights
        - sum of weights = 1
        - each weight is between 0 and 1
    """
    def mse(weights, X, y):
        """
        Sum of squared errors  
        """ 
        return np.sum(np.square((np.dot(X, weights) - y)))

    # Sum of weights must be equal to 1 and each weight must be between 0 and 1
    cons = ({'type': 'eq','fun' : lambda w: np.sum(w) - 1.0})
    bnds = [(0, 1)  for  _ in range(X[0].shape[0])]

    x0 = [current_estimate]
    opt = minimize(
        mse,
        x0,
        method='SLSQP',
        constraints=cons,
        bounds=bnds,
        args=(np.array(X), np.array(y)),
    )
    weights = np.array(opt.x)

    return weights

def estimateWeightsReg(X, y):
    """
    Estimate the weights with Linear Regression and projects 
    them on the 0-1 weight simplex
    """
    res = lsq_linear(X, y, bounds=(0, 1), lsmr_tol='auto', verbose=0)

    # Sum must be between 0 and 1 
    weights = res.x / np.sum(res.x)
    # print(res.x)
    # print(weights)
    return weights



def findWeightsWithAbsReturns(user, env, seed, method="opti"):
    random_state = RandomState(seed)

    # Setup of the environment and agent
    n_obj = 2
    n_actions = env.nA
    n_states = env.nS
    agent = QLearningAgent(n_actions=n_actions, n_states=n_states, decay=0.999997, random_state=random_state)
    solver = Solver(env, agent) # Used to train and evaluate the agent on the environements
    logs = {
        "returns": [],
        "weights": []
    }

    # Data points seen 
    # X contains the returns obtained so far
    # y contains the noisy user utility for those returns 
    X, y = [], []

    # We start with an estimate of the weights weights
    weights = np.ones(n_obj) / n_obj
    # history.append(weights[0])
    prev_weights = weights + 1

    # When the difference between two successive estimated weights is below eps. we stop
    epsilon = 1e-3

    it = 0
    while it < 5:# or not (np.abs(prev_weights - weights)<epsilon).all():
        # print("Iteration %d" % it)
        # print("Current weights estimates: " + str(weights))
        # Solve the environement for the random weights
        # w0_to_solve = random_state.uniform(0, 1)
        # w_solve = np.array([w0_to_solve, 1 - w0_to_solve])
        returns = solver.solve(weights)
        # print("Returns for the weights " + str(w_solve) + ": " + str(returns))
        logs["returns"].append(returns)
        logs["weights"].append(weights[0])

        # Get a noisy estimate of the user utility
        u = user.get_utility(returns)
        # print("Utility of the user: " + str(u) + "\n")

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
        
        # history.append(weights[0])

        it += 1

    return logs
    
    # f, ax = plt.subplots()
    # ax.plot(history,  marker='o')
    # ax.hlines(user.hidden_weights[0], xmin=0, xmax=len(history))
    # # ax.set_yticks(list(np.arange(0, 1, 0.1)))
    # ax.set_xlabel("Iteration")
    # ax.set_ylabel("w0 estimate")
    # plt.savefig(f"figures/exp_method_{method}.png")


if __name__ == "__main__":
    pass
    # main(method="opti")