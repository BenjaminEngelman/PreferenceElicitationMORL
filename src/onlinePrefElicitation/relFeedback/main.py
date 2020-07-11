import logging
import numpy as np
from numpy.random import RandomState
from src.rl.solver import SingleObjSolver
from src.utils import get_best_sol_BST, get_best_sol
from src.ols.utils import create_3D_pareto_front


def elicitWithRelFeedback(user, env_name, seed, solver_calls_budget=100, metric="solver"):
    logging.info(f"\n \n")

    logs = {
        "returns": [],
        "weights": [],
        "solver_calls": [],
        "stds": [],
    }
    solver = SingleObjSolver()  # Used to train and evaluate the agent on the env

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

    # Initial weights to solve the problem
    if num_obj == 2:
        initial_weights = [
            np.array([0, 1]),
            np.array([1, 0]),
        ]

    # If 3 objectives solve for two arbitrary weights
    # that are not too close from each other
    elif num_obj == 3:
        initial_weights = [
            np.array([0.65, 0.25, 0.10]),
            # np.array([0.1, 0.1, 0.8]),
            np.array([0.1, 0.5, 0.4]),
            # np.array([0.1, 0.25, 0.65]),
        ]

    # Solve the problem for the initial weights
    initial_returns = [
        solver.solve(env_name, weights, random_state=random_state) for weights in initial_weights
    ]

    logs["weights"].extend(initial_weights)
    logs["returns"].extend(initial_returns)

    # Ask the user to compare the initial solutions
    logging.info("Comparison between: " + str(initial_returns[0]) + " and " + str(initial_returns[1]))
    preferred = user.compare(initial_returns[0], initial_returns[1])[0]
    logging.info("User prefers: " + str(preferred))

    # Compute a first estimate of the user's weights
    weights, w_fit, H_fit = user.current_map()
    std = 1 / np.sqrt(H_fit)

    # Keep track of the number of queries asked to the user
    n_queries = 2

    # The stop condition could be the number of queries to the user
    # or the number of solver calls
    if metric == "user":
        def stopCondition():
            return n_queries >= solver_calls_budget
    elif metric == "solver":
        def stopCondition():
            return solver.n_calls >= solver_calls_budget

    while not stopCondition():

        logging.info(f"Current weights: {weights} | Current std : {std}")
        logs["weights"].append(weights)
        logs["stds"].append(std)

        # Solve the problem for the current estimate of the User's weights
        returns = solver.solve(env_name, weights, random_state=random_state)
        logging.info("Current returns: " + str(returns))
        logs["returns"].append(returns)
        logs["solver_calls"].append(solver.n_calls)

        n_try = 0
        # While we can still sample new solutions
        while (list(returns) == list(preferred)) and (not stopCondition()) and (n_try < 100):
            weights = user.sample_weight_vector()
            returns = solver.solve(env_name, weights, random_state=random_state)
            n_try += 1

        # Make a new comparison
        logging.info("Comparison between: " + str(returns) + " and " + str(preferred))
        preferred = user.compare(returns, preferred)[0]
        logging.info("User prefers: " + str(preferred))
        logging.info("")
        n_queries += 1

        # Compute the new weights MAP (new estimate of the user's weights)
        weights, w_fit, H_fit = user.current_map()
        std = 1 / np.sqrt(H_fit)

    return logs


if __name__ == "__main__":
    pass
    # Debugging runs
    # from src.constants import *
    # from src.utils import plot_weight_estimations, plot_on_ternary_map, plot_2d_run
    #
    # ts = datetime.datetime.now().timestamp()
    # logging.basicConfig(
    #     format='%(message)s', filename=f'src/relFeedback/logs/experiment_{ts}.log', level=logging.INFO)
    #
    # # for w in WEIGHTS_COMP_BST:
    #
    # #     # for noise_pct in range(1, 11):
    # #     #     logging.info(f"\n#### Noise: {noise_pct} ####\n")
    #
    # #     #     for seed in range(0, 10):
    # #     #         logging.info(f"\n#### Seed: {seed} ####\n")
    # #     #         main(method=1, noise=noise_pct, seed=seed)
    #
    # #     seed = 1
    # #     rs = RandomState(seed)
    # #     user = User(num_objectives=2, noise_pct=2.000, random_state=rs, weights=w)
    # #     logs = findWeightsWithComparisons(user, env_name="synt_bst", seed=seed)
    # # user_w = [0.1, 0.1, 0.8]
    # # user_w = [0.15,0.3, 0.55]
    # # user_w = [0.05,0.9, 0.05]
    #
    # # seed = 1
    # # rs = RandomState(seed)
    # # user = User(num_objectives=3, noise_pct=0.1, random_state=rs, weights=user_w, num_virtual_comps=0)
    # # logs = findWeightsWithComparisons(user, env_name="synt", seed=seed, solver_calls_budget=40)
    # # print(logs["returns"][-1])
    #
    # # plot_on_ternary_map(logs, user_w, "synt")
    #
    # # # user_w = [0.2, 0.8]
    # user_w = [0.5, 0.5]
    # # user_w = [0.8, 0.2]
    # # user_w = [0.26, 0.74]
    # # user_w = [0.22, 0.78]
    # # user_w = [0.82, 0.18]
    # # user_w = [0.62, 0.38]
    #
    # seed = 1
    # rs = RandomState(seed)
    # user = User(num_objectives=2, noise_pct=0.1, random_state=rs, weights=user_w, num_virtual_comps=0)
    # logs = elicitWithRelFeedback(user, env_name="bst", seed=seed, solver_calls_budget=6, metric="solver")
    # plot_2d_run(logs, user_w)
