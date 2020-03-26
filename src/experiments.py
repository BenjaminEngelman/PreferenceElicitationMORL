import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

from user import User
from constants import BST_MAX_TIME, BST_MAX_TREASURE
from env import BountyfulSeaTreasureEnv
from witComparaisons.main import findWeightsWithComparisons
from withAbsReturns.main import findWeightsWithAbsReturns
from ols.main import ols
from utils import plot_compareMethods, plot_experimentNoise, computeFromNestedLists



def get_distances_from_optimal_returns(logged_returns, optimal_returns):

    norm = lambda x: x[0] / BST_MAX_TREASURE
    distances = []

    norm_opt = norm(optimal_returns)

    for returns in logged_returns:
        norm_ret = norm(returns)
        dist = np.absolute(norm_opt - norm_ret)

        distances.append(dist)

    return distances


def compareMethods():
    """
    Compare withComparaisons and withAbsReturns
    """

    # For each of those weights the optimal return is different
    # We only consider w0 as w1 = 1 - w0
    WEIGHTS_LIST = {
        # Key = w0, value = optimal returns
        0.0: [5, -1],
        0.04: [80, -3],
        0.07: [120, -5],
        0.15: [140, -7],
        0.22: [150, -9],
        0.25: [163, -13],
        0.28: [166, -14],
        0.45: [173, -17],
    }

    noise = 0.001
    seed = 1

    for weight in WEIGHTS_LIST:
        print("---------")
        print(f"Weight = {weight}")
        print("---------")

        weight_vector = np.array([weight, 1-weight])
        optimal_returns = WEIGHTS_LIST[weight]

        random_state = RandomState(1)
        env = BountyfulSeaTreasureEnv()
        user = User(
            num_objectives=2,
            std_noise=noise,
            random_state=random_state,
            weights=weight_vector
        )

        # WithComparaisons
        logs_comps = findWeightsWithComparisons(user, env, seed=seed)
        # [2:] because first 2 returns are fixed to get first comparaisons
        distances_withComp = get_distances_from_optimal_returns(
            logs_comps["returns"][2:], optimal_returns)

        # withAbsReturs
        logs_abs = findWeightsWithAbsReturns(
            user, env, seed=seed, method="opti")
        distances_withAbsRet = get_distances_from_optimal_returns(
            logs_abs["returns"], optimal_returns)

        plot_compareMethods(
            distances_withComp,
            distances_withAbsRet,
            logs_comps["weights"][2:],
            logs_abs["weights"],
            weight,
            noise
        )


def experimentNoise(method):

    WEIGHTS_LIST = {
        # Key = w0, value = optimal returns
        # 0.0: [5, -1],
        # 0.04: [80, -3],
        0.07: [120, -5],
        # 0.15: [140, -7],
        # 0.22: [150, -9],
        # 0.25: [163, -13],
        0.28: [166, -14],
        # 0.45: [173, -17],
    }

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

        weight_vector = np.array([weight, 1-weight])
        optimal_returns = WEIGHTS_LIST[weight]

        mean_distances, std_distances = [], []
        mean_weightEstimates, std_weightEstimates = [], []
        for noise in noise_values:
            print(f"Noise = {noise}")

            all_seed_distances = []
            all_seed_weightEstimates = []

            for seed in range(nseed):
                print(f"Seed: {seed}")
                random_state = RandomState(seed)
                env = BountyfulSeaTreasureEnv()
                user = User(
                    num_objectives=2,
                    std_noise=noise,
                    random_state=random_state,
                    weights=weight_vector
                )

                if method == "comparisons":
                    logs = findWeightsWithComparisons(user, env, seed=seed)
                elif  method == "absolute":
                    logs = findWeightsWithAbsReturns(user, env, seed=seed)
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
            mean_distances,
            std_distances,
            mean_weightEstimates,
            std_weightEstimates,
            noise_values,
            weight,
            method
        )
           

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=('comp', 'noise'), help="The name of the experiment to run")
    parser.add_argument('--method', choices=('comparisons', 'absolute', 'all') ,help="The name of the method")

    args = parser.parse_args()
    if args.experiment == "noise" and args.method == None:
        print("Please specify the name of the method")
    

    if args.experiment == "comp":
        compareMethods()
    
    elif args.experiment == "noise":
        if args.method != "all":
            experimentNoise(args.method)
        else:
            experimentNoise("absolute")
            experimentNoise("comparisons")
    
        

    


