import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

from src.user import User
from src.constants import *
from src.withComparisons.main import findWeightsWithComparisons
from src.withAbsReturns.main import findWeightsWithAbsReturns
# from ols.main import ols
from src.utils import plot_compareMethods, plot_experimentNoise, computeFromNestedLists
from src.utils import get_best_sol_BST, get_best_sol
from src.ols.utils import create_3D_pareto_front



def get_distances_from_optimal_returns(logged_returns, optimal_returns):

    norm = lambda x: x[0] / BST_MAX_TREASURE
    distances = []

    norm_opt = norm(optimal_returns)

    for returns in logged_returns:
        norm_ret = norm(returns)
        dist = np.absolute(norm_opt - norm_ret)

        distances.append(dist)

    return distances


def compareMethods(experiment_id, env_name):
    """
    Compare withComparaisons and withAbsReturns
    """

    # For each of those weights the optimal return is different
    # We only consider w1 as w0 = 1 - w1
    if env_name == "bst":
        num_obj = 2
        WEIGHTS_LIST = WEIGHTS_COMP_BST

    elif env_name == "synt":
        solutions = create_3D_pareto_front()
        num_obj = 3

        WEIGHTS_LIST = WEIGHTS_COMP_SYNT
    elif env_name == "minecart":
        num_obj = 3
        WEIGHTS_LIST = WEIGHTS_NOISE_MINECART

    noise = 0.001
    seed = 1
    random_state = RandomState(1)
    
    for weight in WEIGHTS_LIST:
        print("---------")
        print(f"Weight = {weight}")
        print("---------")

        # Get the optimal return for the current weight
        # So we can compare the result of the P.E. method
        weight_vector = np.array(weight)
        if env_name == "bst":
            optimal_returns = get_best_sol_BST(weight_vector)

        elif env_name == "synt":
            optimal_returns = get_best_sol(solutions, weight_vector)

        elif env_name == "minecart":
            print("Not Ready yet")
            exit()

        # Create a user with those weights (i.e. preferences)
        user = User(
            num_objectives=num_obj,
            std_noise=noise,
            random_state=random_state,
            weights=weight_vector
        )

        # P.E. Methods
        ##############

        # WithComparaisons
        logs_comps = findWeightsWithComparisons(user, env_name, seed=seed)
        # [2:] because first 2 returns are fixed to get first comparaisons
        distances_withComp = get_distances_from_optimal_returns(
            logs_comps["returns"][2:], optimal_returns)

        # withAbsReturs
        logs_abs = findWeightsWithAbsReturns(user, env_name, seed=seed, method="opti")
        distances_withAbsRet = get_distances_from_optimal_returns(
            logs_abs["returns"], optimal_returns)

        plot_compareMethods(
            experiment_id,
            distances_withComp,
            distances_withAbsRet,
            logs_comps["weights"][2:],
            logs_abs["weights"],
            weight,
            noise
        )


def experimentNoise(experiment_id, method, env_name):

    if env_name == "synt":
        num_obj = 3
        WEIGHTS_LIST = WEIGHTS_NOISE_SYNT
    elif env_name == "bst":
        num_obj = 2
        WEIGHTS_LIST = WEIGHTS_COMP_BST
    elif env_name == "minecart":
        num_obj = 3
        WEIGHTS_LIST = WEIGHTS_NOISE_MINECART


    nseed = 10

    noise_values = [
        0.001,
        0.01,
        0.1,
    ]
    for weight in WEIGHTS_LIST:
        print("---------")
        print(f"Weight = {weight}")
        print("---------")

        weight_vector = np.array(weight)

        # Get the optimal return for the current weight
        # So we can compare the result of the P.E. method
        if env_name == "bst":
            optimal_returns = get_best_sol_BST(weight_vector)

        elif env_name == "synt":
            optimal_returns = get_best_sol(solutions, weight_vector)

        elif env_name == "minecart":
            print("Not Ready yet")
            exit()

        mean_distances, std_distances = [], []
        mean_weightEstimates, std_weightEstimates = [], []
        for noise in noise_values:
            print(f"Noise = {noise}")

            all_seed_distances = []
            all_seed_weightEstimates = []

            for seed in range(nseed):
                random_state = RandomState(seed)

                print(f"Seed: {seed}")
                
                user = User(
                    num_objectives=num_obj,
                    std_noise=noise,
                    random_state=random_state,
                    weights=weight_vector
                )

                if method == "comparisons":
                    logs = findWeightsWithComparisons(user, env_name, seed=seed)
                elif  method == "absolute":
                    logs = findWeightsWithAbsReturns(user, env_name, seed=seed)
                else:
                    print("Incorrect method.")
                    exit()
                
                if method == "comparisons":
                    # 2 first are fixed to get first comparisons
                    returns = logs["returns"][2:]
                    weights = logs["weights"][2:]
                else:
                    returns = logs["returns"]
                    weights = logs["weights"]

                distances = get_distances_from_optimal_returns(returns, optimal_returns)

                all_seed_distances.append(distances)
                all_seed_weightEstimates.append(weights)

            # Compute means and stds
            mean_distances.append(computeFromNestedLists(all_seed_distances, "mean"))
            std_distances.append(computeFromNestedLists(all_seed_distances, "std"))

            mean_weightEstimates.append(computeFromNestedLists(all_seed_weightEstimates, "mean"))
            std_weightEstimates.append(computeFromNestedLists(all_seed_weightEstimates, "std"))


        plot_experimentNoise(
            experiment_id,
            mean_distances,
            std_distances,
            mean_weightEstimates,
            std_weightEstimates,
            noise_values,
            weight,
            method
        )
           

if __name__ == "__main__":

    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--eid', help="The id of the experiment")
    parser.add_argument('--experiment', choices=('comp', 'noise', 'all'), help="The name of the experiment to run")
    parser.add_argument('--method', choices=('comparisons', 'absolute', 'all'), help="The name of the method")
    parser.add_argument('--env', choices=('bst', 'minecart', 'synt'), help="help the name of the environement to solve")

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
        compareMethods(args.eid, args.env)
    
    elif args.experiment == "noise":
        if args.method != "all":
            experimentNoise(args.eid, args.method, args.env)
        else:
            experimentNoise(args.eid, "absolute", args.env)
            experimentNoise(args.eid, "comparisons", args.env)
    
    else:
        compareMethods(args.eid, args.env)
        experimentNoise(args.eid, "absolute", args.env)
        experimentNoise(args.eid, "comparisons", args.env)

    
        

    


