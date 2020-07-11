import numpy as np
from numpy.random import RandomState
import pickle
from src.onlinePrefElicitation.user import User
from src.constants import *
from src.onlinePrefElicitation.relFeedback.main import elicitWithRelFeedback
from src.onlinePrefElicitation.absFeedback.main import elicitWithAbsFeedback
# from ols.main import ols
from src.utils import plot_compareMethods, plot_on_ternary_map, \
    plot_2d_run
from src.utils import get_best_sol_BST, get_best_sol
from src.ols.utils import create_3D_pareto_front


def get_utility_loss(logged_returns, optimal_weights, env_name):
    """
    :param logged_returns: The returns provided at each step by the PE method
    :param optimal_weights: The hidden weights of the user
    :param env_name: the name of the problem
    :return: The utility loss at each step
    """
    if env_name in ["synt_bst", "bst"]:
        utilities = [optimal_weights[0] * sol[0] + optimal_weights[1] * sol[1] for sol in BST_SOLUTIONS]
    elif env_name in ["synt"]:
        utilities = [optimal_weights[0] * sol[0] + optimal_weights[1] * sol[1] + optimal_weights[2] * sol[2] for sol in
                     create_3D_pareto_front()]
    elif env_name in ["synt_20"]:
        utilities = [optimal_weights[0] * sol[0] + optimal_weights[1] * sol[1] + optimal_weights[2] * sol[2] for sol in
                     create_3D_pareto_front(size=20)]

    u_losses = []
    for returns in logged_returns:
        utility = np.dot(np.array(optimal_weights), np.array(returns))
        normalized = (utility - min(utilities)) / (max(utilities) - min(utilities))
        dist = 1 - normalized

        u_losses.append(dist)

    return u_losses


def get_distances_from_optimal_weights(logged_weights, optimal_weights):
    distances = []
    for w in logged_weights:
        dist = np.linalg.norm(np.array(optimal_weights) - np.array(w))
        distances.append(dist)
    return distances


def experimentNoise(experiment_id, method, env_name):
    """
    Run a PE method for multiple levels of noise
    in a single environment
    """
    if env_name == "synt":
        num_obj = 3
        WEIGHTS_LIST = WEIGHTS_NOISE_SYNT
    elif env_name == "bst" or env_name == "synt_bst":
        num_obj = 2
        WEIGHTS_LIST = WEIGHTS_COMP_BST
    # elif env_name == "minecart":
    #     num_obj = 3
    #     WEIGHTS_LIST = WEIGHTS_NOISE_MINECART

    seed = 42
    random_state = RandomState(seed)

    n_runs = 500

    noise_values = [
        0,
        5,
        10,
        20,
        40,
        60,
        80,
        100,
    ]

    # mean_distances, std_distances = [], []
    # mean_weightEstimates, std_weightEstimates = [], []
    results = {}
    for noise in noise_values:

        all_seed_distances = []
        all_seed_weightEstimates = []

        for run in range(n_runs):
            print(f"Noise = {noise}")

            weight_vector = random_state.uniform(0.0, 1, N_OBJ[env_name])
            weight_vector /= np.sum(weight_vector)

            user = User(
                num_objectives=num_obj,
                noise_pct=noise,
                random_state=random_state,
                weights=weight_vector
            )

            if env_name == "synt":
                pf = create_3D_pareto_front()
                optimal_returns = get_best_sol(pf, weight_vector)
            else:
                optimal_returns = get_best_sol_BST(weight_vector)

            if method == "comparisons":
                logs = elicitWithRelFeedback(user, env_name, seed=seed)
            elif method == "absolute":
                logs = elicitWithAbsFeedback(user, env_name, seed=seed, solver_calls_budget=19)
            else:
                print("Incorrect method.")
                exit()

            returns = logs["returns"]
            weights = logs["weights"]

            distances = get_utility_loss(returns, optimal_returns, weight_vector, env_name)
            all_seed_distances.append(distances)
            distances_w = get_distances_from_optimal_weights(weights, weight_vector)
            all_seed_weightEstimates.append(distances_w)

        results[noise] = {"dist": all_seed_distances, "w": all_seed_weightEstimates}

        with open(f'experiments/{experiment_id}/results.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def single_run(env_name, optimal_weights, method, seed, num_virtual_comp=10, solver_calls_budget=5, low_noise=False):
    """
    Run a PE method for some weights (preferences) in one of the environment
    """
    random_state = RandomState(seed)
    if method == "absolute":
        if low_noise:
            noise_pct = 0.1
        else:
            noise_pct = 10
    else:
        noise_pct = 0.1

    user = User(num_objectives=N_OBJ[env_name], noise_pct=noise_pct, env=env_name, random_state=random_state,
                weights=optimal_weights, num_virtual_comps=num_virtual_comp)
    if method == "absolute":
        logs = elicitWithAbsFeedback(user, env_name, seed=seed, method="opti",
                                     solver_calls_budget=solver_calls_budget)
    elif method == "comparisons":
        logs = elicitWithRelFeedback(user, env_name, seed=seed, solver_calls_budget=solver_calls_budget,
                                     metric="solver")
    res = (0, logs)

    return res


def absolute_minecart(exp_name):
    env_name = "minecart"
    seed = 1

    for weights in WEIGHTS_EXP_MINECART:
        res = single_run(env_name, weights, "absolute", seed)
        with open(f'experiments/{exp_name}/logs_{weights}.pickle', 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


def comparisons_synt(exp_name):
    env_name = "synt"
    seed = 1

    for weights in WEIGHTS_COMP_SYNT:
        _, res = single_run(env_name, weights, "comparisons", seed)
        plot_on_ternary_map(res, weights, env_name, method="comparison", experiment_id=exp_name)


def comparisons_synt_bst(exp_name):
    env_name = "synt_bst"
    seed = 1

    for weights in WEIGHTS_COMP_BST:
        _, res = single_run(env_name, weights, "comparisons", seed)
        plot_2d_run(res, weights, method="comparison", experiment_id=exp_name)


def comparisons_accuracy(num_virtual_comp, env_name, solver_calls_budget):
    """
    Run the PE method using relative feedback 100 times
    using num_virtual_comp virtual comparisons and get the accuracy
    :param num_virtual_comp:
    :param env_name:
    :param solver_calls_budget:
    :return:
    """
    seed = 1
    random_state = RandomState(seed)
    n_trials = 100

    successes = 0
    solver_calls = []
    distances = []

    for step in range(n_trials):
        print(step)
        weights = random_state.uniform(0.0, 1, N_OBJ[env_name])
        weights /= np.sum(weights)

        if env_name == "synt":
            pf = create_3D_pareto_front()
            optimal_result = get_best_sol(pf, weights)
        else:
            optimal_result = get_best_sol_BST(weights)

        _, res = single_run(env_name, weights, "comparisons", seed, num_virtual_comp=num_virtual_comp,
                            solver_calls_budget=solver_calls_budget)
        obtained_res = res["returns"][-1]
        # Check if it converged
        if list(obtained_res) == list(optimal_result):
            successes += 1

        u_loss = get_utility_loss(res["returns"], optimal_result, weights, env_name)
        distances.append(u_loss)

    s_rate = (successes / n_trials) * 100

    # with open(f'experiments/test.pickle', 'wb') as handle:
    #     pickle.dump([s_rate, distances, solver_calls], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return s_rate, distances, solver_calls


def eval_PE_withComparisons(exp_name, env_name):
    """
    Evaluate the PE method using relative comparisons for different number of virtual comparisons
    :param exp_name:
    :param env_name:
    :return:
    """
    results = {}
    for solver_calls_budget in [3, 6, 9, 12, 15]:
        results[solver_calls_budget] = {}

        for num_virtual_comp in range(0, 45, 5):
            print(f"Budget: {solver_calls_budget} | n_virt_coms: {num_virtual_comp}")
            s_rate, _ = comparisons_accuracy(num_virtual_comp, env_name, solver_calls_budget)
            results[solver_calls_budget][num_virtual_comp] = s_rate
            with open(f'experiments/{exp_name}/results.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def did_it_succeed(optimal_returns, res):
    """
    Check if the last solution of the PE method was the optimal one
    :param optimal_returns:
    :param res:
    :return:
    """
    obtained_returns = res["returns"][-1]
    return list(optimal_returns) == list(obtained_returns)


def experiment1(exp_name, env_name):
    """
    Run the two Online PE methods on the problem "env_name"
    for 500 runs (500 random users)
    :param exp_name:
    :param env_name:
    :return:
    """
    seed = 42
    random_state = RandomState(seed)
    n_trials = 500
    results = {}

    all_weights = np.random.uniform(0, 1, (n_trials, N_OBJ[env_name]))
    all_weights /= all_weights.sum(axis=1)[:, None]

    for scb in range(3, 53):
        successes_abs = 0
        successes_comp = 0

        distances_abs = []
        distances_comp = []

        for step, weights in enumerate(all_weights):
            print(f"Solver call buget: {scb} | Run number {step}")
            weights = random_state.uniform(0.0, 1, N_OBJ[env_name])
            weights /= np.sum(weights)

            if env_name == "synt":
                pf = create_3D_pareto_front()
                optimal_returns = get_best_sol(pf, weights)
            elif env_name == "synt_20":
                pf = create_3D_pareto_front(size=20)
                optimal_returns = get_best_sol(pf, weights)
            else:
                optimal_returns = get_best_sol_BST(weights)

            _, res_abs = single_run(env_name, weights, "absolute", seed, solver_calls_budget=scb)
            _, res_comp = single_run(env_name, weights, "comparisons", seed, num_virtual_comp=0,
                                     solver_calls_budget=scb)

            successes_abs += int(did_it_succeed(env_name, weights, optimal_returns, res_abs))
            successes_comp += int(did_it_succeed(env_name, weights, optimal_returns, res_comp))

            distances_abs.append(
                get_utility_loss(res_abs["returns"], optimal_returns, weights, env_name)[-1])
            distances_comp.append(
                get_utility_loss(res_comp["returns"], optimal_returns, weights, env_name)[-1])

        results[scb] = {
            "abs": {"success": (successes_abs / n_trials) * 100, "distance": distances_abs},
            "comp": {"success": (successes_comp / n_trials) * 100, "distance": distances_comp}
        }

        with open(f'experiments/{exp_name}/results.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results


def abs_low_noise(exp_name, env_name):
    """
    Run the PE method using absolute feedback for with a small noise level
    :param exp_name:
    :param env_name:
    :return:
    """
    seed = 42
    random_state = RandomState(seed)
    n_trials = 500
    all_weights = random_state.uniform(0, 1, (n_trials, N_OBJ[env_name]))
    all_weights /= all_weights.sum(axis=1)[:, None]
    results = {}

    for scb in range(3, 20):
        successes = 0
        distances = []

        for step, weights in enumerate(all_weights):
            print(f"Solver call budget: {scb} | Run number {step}")

            if env_name == "synt":
                pf = create_3D_pareto_front()
                optimal_returns = get_best_sol(pf, weights)
            else:
                optimal_returns = get_best_sol_BST(weights)

            _, res = single_run(env_name, weights, "absolute", seed, solver_calls_budget=scb, low_noise=True)

            successes += int(did_it_succeed(env_name, weights, optimal_returns, res))

            distances.append(get_utility_loss(res["returns"], optimal_returns, weights, env_name)[-1])

        results[scb] = {
            "abs": {"success": (successes / n_trials) * 100, "distance": distances},
        }

        with open(f'experiments/{exp_name}/results.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results


def get_opt_return(env_name, weights):
    """
    Get the optimal solution for an environment (i.e. problem) and some weights
    :param env_name:
    :param weights:
    :return:
    """
    if env_name == "synt":
        pf = create_3D_pareto_front()
        optimal_returns = get_best_sol(pf, weights)
    elif env_name == "synt_20":
        pf = create_3D_pareto_front(size=20)
        optimal_returns = get_best_sol(pf, weights)
    else:
        optimal_returns = get_best_sol_BST(weights)
    return optimal_returns


if __name__ == "__main__":

    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--eid', help="The id of the experiment")
    parser.add_argument('--experiment', help="The name of the experiment to run")
    parser.add_argument('--method', choices=('comparisons', 'absolute', 'all'), help="The name of the method")
    parser.add_argument('--env', choices=('bst', 'synt_bst', 'minecart', 'synt', 'synt_20'),
                        help="help the name of the environement to solve")

    args = parser.parse_args()
    if args.eid == None:
        print("Please provide an experiment ID")
        exit()
    else:
        os.mkdir(f"experiments/{args.eid}/")

    if args.experiment == "noise" and args.method == None:
        print("Please specify the name of the method")
        exit()

    if args.experiment == "comp":
        # compareMethods(args.eid, args.env)
        experiment1(args.eid, args.env)

    elif args.experiment == "noise":
        if args.method != "all":
            experimentNoise(args.eid, args.method, args.env)
        else:
            experimentNoise(args.eid, "absolute", args.env)
            experimentNoise(args.eid, "comparisons", args.env)

    elif args.experiment == "minecart_abs":
        absolute_minecart(args.eid)

    elif args.experiment == "synt_comp":
        comparisons_synt(args.eid)

    elif args.experiment == "synt_bst_comp":
        comparisons_synt_bst(args.eid)

    elif args.experiment == "abs_low_noise":
        abs_low_noise(args.eid, args.env)

    elif args.experiment == "eval_comps":
        eval_PE_withComparisons(args.eid, args.env)
