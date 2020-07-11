import sys
sys.path.insert(0, '..')

import queue
import datetime
import math
import logging
import numpy as np
import matplotlib.pyplot as plt

from time import time
from numpy.random import RandomState
from recordclass import recordclass

from src.env import BountyfulSeaTreasureEnv
from src.agents import Qlearning
from src.solver import Solver
from src.utils import get_best_sol_BST, get_best_sol
from src.ols.utils import create_3D_pareto_front
from src.user import User


# N_STEPS = 16

def small_unecertainty(H_fit):
    stdevs = []
    for elem in H_fit:
        stdevs.append(1.0 / math.sqrt(elem))
    # print(stdevs)

    return any(elem < 0.1 for elem in stdevs)


def plot_weights_history(user, weights_history, noise, seed):
    f, ax = plt.subplots()
    ax.plot(weights_history,  marker='o')
    ax.hlines(user.hidden_weights[1], xmin=0, xmax=len(weights_history))
    # ax.set_yticks(list(np.arange(0, 1, 0.1)))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("w0 estimate")
    plt.savefig(f"figures/exp_seed_{seed}_noise_{noise}_{user.hidden_weights}.png")


def get_returns_to_compare(returns, prefered_results, rejected_results, previous_comparisons, random_state):
    # print(f"To compare  = {returns} - Choosing between {prefered_results}")

    # Prioritze prefered results
    # If none prefered results can be compared
    # Consider rejected results
    for results in [prefered_results, rejected_results]:

        for r in results[::-1]:
            # If r and returns are different and have not been compared yet
            # [returns.tolist(), r.tolist()] not in previous_comparisons and
            if  (np.abs(returns - r) != 0).all():
                return r




def findWeightsWithComparisons(user, env_name, seed, solver_calls_budget=100, metric="solver"):
    """
    Starts by computing the reward for the policies associated to the 
    [1 0] and [0 1] weights
    Then asks for comparison between the last return and the previous best which is different.
    If no previous best return is different, choose from the rejected returns.
    """
    logging.info(f"\n \n")

    logs = {
        "returns": [],
        "weights": [],
        "solver_calls": [],
        "stds": [],
    }
    solver = Solver()

    # Check expected solution for real weights
    if env_name == "synt":
        real_sol = get_best_sol(create_3D_pareto_front(), user.hidden_weights)
    elif env_name == "synt_20":
        real_sol = get_best_sol(create_3D_pareto_front(size=20), user.hidden_weights)
    elif env_name == "synt_bst" or "bst":
        real_sol = get_best_sol_BST(user.hidden_weights)

    logging.info(f"Expected solution for hidden weights of the user is: {real_sol}")

    logging.info("Hidden weights: " + str(user.hidden_weights) + "\n")

    random_state = RandomState(seed)

    num_obj = user.num_objectives

    if num_obj == 2:
        initial_weights = [
            np.array([0, 1]),
            np.array([1, 0]),
        ]
    
    elif num_obj == 3:
        initial_weights = [
            np.array([0.65, 0.25, 0.10]),
            # np.array([0.1, 0.1, 0.8]),
            np.array([0.1, 0.5, 0.4]),
            # np.array([0.1, 0.25, 0.65]),
        ]

    
    
    initial_returns = [
        solver.solve(env_name, weights, random_state=random_state) for weights in initial_weights
    ]


    logs["weights"].extend(initial_weights)
    logs["returns"].extend(initial_returns)
    logs["solver_calls"].extend([1, 2])

    logs["stds"].extend([[0, 0], [0, 0]])


    logging.info("Comparison between: " + str(initial_returns[0]) + " and " + str(initial_returns[1]))
    prefered = user.compare(initial_returns[0], initial_returns[1])[0]
    logging.info("User prefers: " + str(prefered))

    index = [np.array_equal(prefered,x) for x in initial_returns].index(True)
    prior_w = initial_weights[index]

    weights, w_fit, H_fit = user.current_map()
    std = 1 / np.sqrt(H_fit)

    query = 2

    if metric == "user":
        def condition():
            return query < solver_calls_budget
    
    elif metric == "solver":
        def condition():
            return solver.n_calls < solver_calls_budget

    while condition():

        logging.info(f"Current weights: {weights} | Current std : {std}")
        logs["weights"].append(weights)
        logs["stds"].append(std)

        returns = solver.solve(env_name, weights, random_state=random_state)
        logging.info("Current returns: " + str(returns))
        logs["returns"].append(returns)
        logs["solver_calls"].append(solver.n_calls) 

        i = 0
        while list(returns) == list(prefered) and condition() and i < 100:
            weights = user.sample_weight_vector()
            returns = solver.solve(env_name, weights, random_state=random_state)
            i += 1
        # logs["weights"].append(weights)
        # logs["returns"].append(returns)

        # Make a new comparison
        logging.info("Comparison between: " + str(returns) + " and " + str(prefered))
        prefered = user.compare(returns, prefered)[0]
        logging.info("User prefers: " + str(prefered))
        logging.info("")
        query += 1

        # Compute the new weights MAP
        weights, w_fit, H_fit = user.current_map()
        std = 1 / np.sqrt(H_fit)

        # if small_unecertainty(H_fit):
        #     logs["weights"].append(weights)
        #     returns = solver.solve(env_name, weights, random_state=random_state)
        #     logs["returns"].append(returns)
        #     logging.info("Sure enough for: " + str(weights) +  "that returns " + str(returns))
        #     break            

        # logging.info("Current weights: " + str(weights) + '\n')


        # logs["weights"].append(weights)
    # assert(solver.n_calls == solver_calls_budget)
    print()

    return logs


if __name__ == "__main__":
    from src.constants import *
    from src.utils import plot_weight_estimations, plot_on_ternary_map, plot_2d_run

    ts = datetime.datetime.now().timestamp()
    logging.basicConfig(
        format='%(message)s', filename=f'src/withComparisons/logs/experiment_{ts}.log', level=logging.INFO)

    # for w in WEIGHTS_COMP_BST: 

    #     # for noise_pct in range(1, 11):
    #     #     logging.info(f"\n#### Noise: {noise_pct} ####\n")

    #     #     for seed in range(0, 10):
    #     #         logging.info(f"\n#### Seed: {seed} ####\n")
    #     #         main(method=1, noise=noise_pct, seed=seed)

    #     seed = 1
    #     rs = RandomState(seed)
    #     user = User(num_objectives=2, noise_pct=2.000, random_state=rs, weights=w)
    #     logs = findWeightsWithComparisons(user, env_name="synt_bst", seed=seed)
    # user_w = [0.1, 0.1, 0.8]
    # user_w = [0.15,0.3, 0.55]
    # user_w = [0.05,0.9, 0.05]

    # seed = 1
    # rs = RandomState(seed)
    # user = User(num_objectives=3, noise_pct=0.1, random_state=rs, weights=user_w, num_virtual_comps=0)
    # logs = findWeightsWithComparisons(user, env_name="synt", seed=seed, solver_calls_budget=40)
    # print(logs["returns"][-1])

    # plot_on_ternary_map(logs, user_w, "synt")

    # # user_w = [0.2, 0.8] 
    user_w = [0.5, 0.5]
    # user_w = [0.8, 0.2]
    # user_w = [0.26, 0.74]
    # user_w = [0.22, 0.78]
    # user_w = [0.82, 0.18]
    # user_w = [0.62, 0.38]

    




    seed = 1
    rs = RandomState(seed)
    user = User(num_objectives=2, noise_pct=0.1, random_state=rs, weights=user_w, num_virtual_comps=0)
    logs = findWeightsWithComparisons(user, env_name="bst", seed=seed, solver_calls_budget=6, metric="solver")
    plot_2d_run(logs, user_w)