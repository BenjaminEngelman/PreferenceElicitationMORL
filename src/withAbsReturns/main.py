from time import time
import logging
import datetime
from numpy.random import RandomState
import numpy as np

from src.env import BountyfulSeaTreasureEnv
from src.agents import Qlearning
from src.solver import Solver
from src.user import User

from scipy.optimize import minimize, lsq_linear
import matplotlib.pyplot as plt

N_STEPS = 4


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



def findWeightsWithAbsReturns(user, env_name, seed, method="opti"):
    random_state = RandomState(seed)

    # Setup of the environment and agent
    n_obj = user.num_objectives
    logs = {
        "returns": [],
        "weights": []
    }

    # Data points seen 
    # X contains the returns obtained so far
    # y contains the noisy user utility for those returns 
    X, y = [], []

    # We start with an estimate of the weights weights
    if env_name in ["synt", "minecart"]:
        weights = np.array([0.75, 0.1, 0.15])
    else:
        weights = np.ones(n_obj) / n_obj


    it = 0
    while it < N_STEPS:
        solver = Solver() # Used to train and evaluate the agent on the environements

        logging.info("Iteration %d" % it)
        logging.info("Current weights estimates: " + str(weights))

        # # Solve the environement for some random weights
        # w_to_solve = random_state.uniform(0.0, 1, n_obj)
        # w_to_solve /= np.sum(w_to_solve)
        w_to_solve = weights

        returns = solver.solve(env_name, w_to_solve, random_state=random_state)
        logging.info("Returns for the weights " + str(w_to_solve) + ": " + str(returns))

        logs["weights"].append(weights)
        # Solve the env for the current weight estimate and add to logs
        # logs["returns"].append(solver.solve(env_name, weights, random_state=random_state)) # Log the returns for the current weight estimate
        logs["returns"].append(returns) # Log the returns for the current weight estimate


        # Get a noisy estimate of the user utility
        u = user.get_utility(returns)
        logging.info("Utility of the user: " + str(u) + "\n")
        

        # Add those to our dataset
        X.append(np.array(returns))
        y.append(u)

        # if env with 3 objectives
        # don't update weights after first iteration
        # start after the second
        if env_name in ["synt_bst", "bst"] or it > 0:
            if method == "opti":
                weights = estimateWeightsOpti(X, y, current_estimate=weights)

            elif method == "reg":
                weights = estimateWeightsReg(X, y)
            else:
                print("Wrong method.")
        else:
            weights = np.array([0.1, 0.5, 0.4])
        
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
    from src.utils import plot_weight_estimations, plot_on_ternary_map, plot_2d_run
    ts = datetime.datetime.now().timestamp()
    logging.basicConfig(
        format='%(message)s', filename=f'src/withAbsReturns/logs/experiment_{ts}.log', level=logging.INFO)

    seed = 1
    rs = RandomState(seed)
    # user_w = [0.9, 0.1, 0.0]
    # user = User(num_objectives=3, noise_pct=0.001, random_state=rs, weights=user_w)
    # logs = findWeightsWithAbsReturns(user, env_name="3d_synthetic", seed=seed)
    # plot_weight_estimations(logs, user_w)

    # user_w = [0.3, 0.3, 0.4] 
    user_w = [0.2, 0.8] 

    env_name = "synt_bst"
    user = User(num_objectives=2, noise_pct=50, random_state=rs, weights=user_w)
    logs = findWeightsWithAbsReturns(user, env_name=env_name, seed=seed)
    plot_2d_run(logs, user_w)
    # plot_on_ternary_map(logs, user_w, env_name)