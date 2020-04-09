import sys
sys.path.insert(0, '..')

import queue
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt

from time import time
from numpy.random import RandomState
from recordclass import recordclass

from src.env import BountyfulSeaTreasureEnv
from src.agents import MOQlearning
from src.solver import Solver
from src.utils import get_best_sol_BST
from src.user import User

Result = recordclass("Result", ["weights", "returns"])


# def get_best_sol(weights):

#     solutions = [
#         [5, -1],
#         [80, -3],
#         [120, -5],
#         [140, -7],
#         [145, -8],
#         [150, -9],
#         [163, -13],
#         [166, -14],
#         [173, -17],
#         # [175, -19],

#     ]

#     def utility(x): return weights[0] * x[0] + weights[1] * x[1]
#     best_u = -1000
#     best_sol = 0

#     for sol in solutions:
#         if utility(sol) > best_u:
#             best_u = utility(sol)
#             best_sol = sol

#     return best_sol


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

        for sol in results[::-1]:
            r = sol.returns
            # If r and returns are different and have not been compared yet
            if [returns.tolist(), r.tolist()] not in previous_comparisons and (np.abs(returns - r) != 0).all():
                return sol




def findWeightsWithComparisons(user, agent, seed):
    """
    Starts by computing the reward for the policies associated to the 
    [1 0] and [0 1] weights
    Then asks for comparison between the last return and the previous best which is different.
    If no previous best return is different, choose from the rejected returns.
    """
    logs = {
        "returns": [],
        "weights": []
    }

    # Check expected solution for real weights
    real_sol = get_best_sol_BST(user.hidden_weights)
    logging.info(f"Expected solution for hidden weights of the user is: {real_sol}")



    logging.info("Hidden weights: " + str(user.hidden_weights) + "\n")

    random_state = RandomState(seed)
    solver = Solver()

    weights_history = []
    weights = np.array([1, 0])
    weights_history.append(weights[1])

    prev_weights = weights + 1
    prefered = None

    prefered_results = []
    rejected_results = []

    previous_comparisons = []
    

    # When the difference between two successive estimated weights is below eps. we stop
    epsilon = 1e-3
    it = 0
    while it < 10:  # or not (np.abs(prev_weights - weights)<epsilon).all():
        logging.info("\nIteration %d" % it)
        logging.info("Current weights estimates :" + str(weights))

        # Q-learning
        agent.reset()
        returns = solver.solve(agent, weights)
        logging.info("Returns for current weights: " + str(returns))

        result = Result(weights, returns)
        logs["returns"].append(returns)
        logs["weights"].append(weights[1])

        if it == 0:
            # The second weight we test is [0, 1]
            # We need at least two policies before starting to compare
            prev_weights = weights
            prefered_results.append(result)
            weights = np.array([0, 1])

        else:
            # Find the return to compare with the current one
            to_compare = get_returns_to_compare(returns, prefered_results, rejected_results, previous_comparisons, random_state)
            if to_compare is None:
                break
            logging.info("Comparison between: " + str(returns) + " and " + str(to_compare.returns))

            prefered, rejected, u_pref, u_rej = user.compare(result, to_compare)
            # I Use tolist() for comparison convenience
            # (See usage of previous_coparisons in get_returns_to_compare(...))
            previous_comparisons.extend([
                [result.returns.tolist(), to_compare.returns.tolist()],
                [to_compare.returns.tolist(), result.returns.tolist()]
            ])

            logging.info("User prefers: " + str(prefered.returns))
            logging.info(f"Utility of preferd = {u_pref}")
            logging.info(f"Utility of rejected = {u_rej}")


            prefered_results.append(prefered)
            rejected_results.append(rejected)

            # Estimate new weights
            prev_weights = weights

            weights = user.current_map(prefered.weights)


        it += 1

    return logs


if __name__ == "__main__":

    ts = datetime.datetime.now().timestamp()
    logging.basicConfig(
        format='%(message)s', filename=f'src/withComparisons/logs/experiment_{ts}.log', level=logging.INFO)

    # for std_noise in range(1, 11):
    #     logging.info(f"\n#### Noise: {std_noise} ####\n")

    #     for seed in range(0, 10):
    #         logging.info(f"\n#### Seed: {seed} ####\n")
    #         main(method=1, noise=std_noise, seed=seed)

    seed = 1
    rs = RandomState(seed)

    env = BountyfulSeaTreasureEnv()
    agent = MOQlearning(env, decay=0.999997, random_state=rs)

    user = User(num_objectives=2, std_noise=0.000, random_state=rs, weights=[1, 0.0])
    logs = findWeightsWithComparisons(user, agent, seed=seed)
    print(logs)