import sys
sys.path.insert(0, '..')

from time import time
import logging
import datetime
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


    it = 0
    while it < 6:# or not (np.abs(prev_weights - weights)<epsilon).all():
        # print("Iteration %d" % it)
        # print("Current weights estimates: " + str(weights))
        # Solve the environement for some random weights
        w0_to_solve = random_state.uniform(0, 1)
        w_to_solve = np.array([w0_to_solve, 1 - w0_to_solve])
        returns = solver.solve(w_to_solve)
        # print("Returns for the weights " + str(w_solve) + ": " + str(returns))
        logs["weights"].append(weights[0])
        logs["returns"].append(solver.solve(weights)) # Log the returns for the current weight estimate


        # Get a noisy estimate of the user utility
        u = user.get_utility(returns)
        logging.info("Utility of the user: " + str(u) + "\n")
        

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
    ts = datetime.datetime.now().timestamp()
    logging.basicConfig(
        format='%(message)s', filename=f'logs/experiment_{ts}.log', level=logging.INFO)

    env = BountyfulSeaTreasureEnv()
    seed = 1
    rs = RandomState(seed)
    user = User(num_objectives=2, std_noise=0.001, random_state=rs, weights=[0.25, 0.75])
    logs = findWeightsWithAbsReturns(user, env, seed=seed)
    print(logs)