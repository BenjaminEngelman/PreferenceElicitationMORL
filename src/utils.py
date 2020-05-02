import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from mpl_toolkits.mplot3d import axes3d
import gym
from gym import spaces
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.callbacks import BaseCallback
from src.constants import BST_SOLUTIONS
from collections import Counter


class MinecartObsWrapper(gym.ObservationWrapper):
    def observation(self, s):
        state = np.append(s['position'], [s['speed'], s['orientation'], *s['content']])
        return state


class MultiObjRewardWrapper(gym.RewardWrapper):
    """
    Transform a multi-ojective reward (= array)
    to a single scalar
    """
    def __init__(self, env, weights):
        super().__init__(env)
        self.weights = weights
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

    def reward(self, rew):
        return self.weights.dot(rew)

class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model once after num_steps steps

    :param num_steps: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_path: str, name_prefix='rl_model', num_steps=15_000_000, verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.num_steps = num_steps
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls == self.num_steps:
            path = os.path.join(self.save_path, '{}_checkpoint'.format(self.name_prefix))
            self.model.save(path)
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True

def most_occuring_sublist(list):
    return np.array(Counter(tuple(d) for d in list).most_common(1)[0][0])

def computeFromNestedLists(nested_vals, op):
    """
    Computed the mean/var a 2-D array and returns a 1-D array of all of the columns
    regardless of their dimensions.
    https://stackoverflow.com/questions/10058227/calculating-mean-of-arrays-with-different-lengths
    """
    output = []
    maximum = 0
    for lst in nested_vals:
        if len(lst) > maximum:
            maximum = len(lst)
    for index in range(maximum): # Go through each index of longest list
        temp = []
        for lst in nested_vals: # Go through each list
            if index < len(lst): # If not an index error
                temp.append(lst[index])
        if op == "mean":
            output.append(np.nanmean(temp))
        elif op == "std":
            output.append(np.std(temp))
    return output


def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x: x[1])[0]


def plot_compareMethods(distances_withComp, distances_withAbsRet, weights_withComp, weights_withAbsRet, optimal_weight, noise):
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    # Distances to optimal return
    axes[0].set_title("Distance to optimal return")
    axes[0].set_ylabel("Distance")

    axes[0].plot(distances_withComp, label="Comparaisons", marker='o')
    axes[0].plot(distances_withAbsRet, label="Absolute return", marker='o')
    axes[0].legend()

    # Weights estimate
    axes[1].set_title("Weight estimate")
    axes[1].set_ylabel("W1 estimate")

    axes[1].plot(weights_withComp, label="Comparaisons", marker='o')
    axes[1].plot(weights_withAbsRet, label="Absolute return", marker='o')
    axes[1].hlines(optimal_weight, xmin=0, xmax=len(weights_withAbsRet), label="optimal")
    axes[1].legend()

    f.savefig(f"figures/comp_methods_{optimal_weight}_noise_{noise}.png")    

def plot_experimentNoise(all_distances, std_distances, all_weightsEstimates, std_weightsEstimates, noise_values, optimal_weight, method):
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))

    print(std_distances)
    print(std_weightsEstimates)

    # Distances to optimal return
    axes[0].set_title("Distance to optimal return")
    axes[0].set_ylabel("Distance")
    
    for d, std, noise_value in zip(all_distances, std_distances, noise_values):
        axes[0].errorbar(list(range(len(d))), d, yerr=std, label=noise_value, marker='o')
    axes[0].legend()
    
    # Weights estimate
    axes[1].set_title("Weight estimate")
    axes[1].set_ylabel("W1 estimate")
    for w, std, noise_value in zip(all_weightsEstimates, std_weightsEstimates, noise_values):
        axes[1].errorbar(list(range(len(w))), w, yerr=std, label=noise_value, marker='o')

    axes[1].hlines(optimal_weight, xmin=0, xmax=8, label="optimal")
    axes[1].legend()

    f.savefig(f"figures/noise_{optimal_weight}_{method}.png")


def plot_weight_estimations(results, optimal_weight):
    weight_estimations = np.array(results["weights"])
    plt.hlines(optimal_weight[1], xmin=0, xmax=len(weight_estimations), label="Optimal")
    plt.plot(weight_estimations, marker='o', label="Estimation")
    plt.legend()
    plt.show()



def get_best_sol(pareto_front, weights):
    utility = lambda x: np.dot(weights, x)
    best_u = -np.inf
    best_sol = 0

    for sol in pareto_front:
        if utility(sol) > best_u:
            best_u = utility(sol)
            best_sol = sol

    return best_sol


def get_best_sol_BST(weights):
    return get_best_sol(BST_SOLUTIONS, weights)