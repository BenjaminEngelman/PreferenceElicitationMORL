import logging
from numpy.random import RandomState
import numpy as np
from src.rl.solver import SingleObjSolver
from scipy.optimize import minimize, lsq_linear


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
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bnds = [(0, 1) for _ in range(X[0].shape[0])]

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


def elicitWithAbsFeedback(user, env_name, seed, method="opti", solver_calls_budget=10):
    random_state = RandomState(seed)
    solver = SingleObjSolver()  # Used to train and evaluate the agent on the env

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
    if env_name in ["synt", "synt_20", "minecart"]:
        weights = np.array([0.75, 0.1, 0.15])
    else:
        weights = np.ones(n_obj) / n_obj

    it = 0
    while it < solver_calls_budget:

        logging.info("Iteration %d" % it)
        logging.info("Current weights estimates: " + str(weights))

        # Solve the problem for the current weight estimate
        w_to_solve = weights
        returns = solver.solve(env_name, w_to_solve, random_state=random_state)
        logging.info("Returns for the weights " + str(w_to_solve) + ": " + str(returns))

        logs["weights"].append(weights)
        # Solve the env for the current weight estimate and add to logs
        logs["returns"].append(returns)  # Log the returns for the current weight estimate

        # Get a noisy estimate of the user utility
        u = user.get_utility(returns)
        logging.info("Utility of the user: " + str(u) + "\n")

        # Add those to our dataset
        X.append(np.array(returns))
        y.append(u)

        # if env with 2 objectives
        if env_name in ["synt_bst", "bst"] or it > 0:
            if method == "opti":
                weights = estimateWeightsOpti(X, y, current_estimate=weights)

            elif method == "reg":
                weights = estimateWeightsReg(X, y)
            else:
                print("Wrong method.")
        # if env has 3 objectives solve it for another arbitrary weight
        # So we have have 2 initial solutions
        else:
            weights = np.array([0.1, 0.5, 0.4])

        it += 1

    return logs


if __name__ == "__main__":
    pass
    # Some Debugging runs

    # from src.utils import plot_weight_estimations, plot_on_ternary_map, plot_2d_run
    # from src.utils import get_best_sol_BST
    #
    # ts = datetime.datetime.now().timestamp()
    # # logging.basicConfig(
    # #     format='%(message)s', filename=f'src/absFeedback/logs/experiment_{ts}.log', level=logging.INFO)
    #
    # seed = 1
    # rs = RandomState(seed)
    # # user_w = [0.9, 0.1, 0.0]
    # # user = User(num_objectives=3, noise_pct=0.001, random_state=rs, weights=user_w)
    # # logs = findWeightsWithAbsReturns(user, env_name="3d_synthetic", seed=seed)
    # # plot_weight_estimations(logs, user_w)
    #
    # # user_w = [0.3, 0.3, 0.4]
    # user_w = [0.3, 0.7]
    #
    # random_state = RandomState(42)
    #
    # while True:
    #     user_w = random_state.uniform(0.0, 1, 2)
    #     user_w /= np.sum(user_w)
    #
    #     best_sol = get_best_sol_BST(user_w)
    #
    #     env_name = "synt_bst"
    #     user = User(num_objectives=2, noise_pct=10.0, random_state=rs, weights=user_w)
    #     logs = findWeightsWithAbsReturns(user, env_name=env_name, seed=seed, solver_calls_budget=10)
    #     if list(logs["returns"][-1]) != list(best_sol):
    #         print(user_w)
    #         plot_2d_run(logs, user_w)
    #         break
    #
    #     # plot_on_ternary_map(logs, user_w, env_name)
