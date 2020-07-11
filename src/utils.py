import pickle
import ternary
import matplotlib.pyplot as plt
import numpy as np
from src.constants import MATPLOTLIB_COLORS
from src.constants import BST_SOLUTIONS
from collections import Counter
import matplotlib

matplotlib.rcParams.update({'font.size': 22})


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


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
    for index in range(maximum):  # Go through each index of longest list
        temp = []
        for lst in nested_vals:  # Go through each list
            if index < len(lst):  # If not an index error
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


def plot_compareMethods(experiemnt_id, distances_withComp, distances_withAbsRet, weights_withComp, weights_withAbsRet,
                        optimal_weight, noise):
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    # Distances to optimal return
    axes[0].set_title("Distance to optimal return")
    axes[0].set_ylabel("Distance")

    axes[0].plot(distances_withComp, label="Comparaisons", marker='o')
    axes[0].plot(distances_withAbsRet, label="Absolute return", marker='o')
    axes[0].legend()

    # Weights estimate
    axes[1].set_title("Weight estimate")
    axes[1].set_ylabel("W1 estimate")

    axes[1].plot(np.array(weights_withComp)[:, 1], label="Comparaisons", marker='o')
    axes[1].plot(np.array(weights_withAbsRet)[:, 1], label="Absolute return", marker='o')
    axes[1].hlines(optimal_weight[1], xmin=0, xmax=len(weights_withAbsRet), label="optimal")
    axes[1].legend()

    f.savefig(f"experiments/{experiemnt_id}/comp_methods_{optimal_weight}_noise_{noise}.png")


def plot_experimentNoise(experiment_id, all_distances, std_distances, all_weightsEstimates, std_weightsEstimates,
                         noise_values, optimal_weight, method):
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

    print(std_distances)
    print(std_weightsEstimates)

    # Distances to optimal return
    axes[0].set_title("Distance to optimal utility")
    axes[0].set_ylabel("Utility Loss")
    axes[0].set_xlabel("Iteration")

    for d, std, noise_value in zip(all_distances, std_distances, noise_values):
        axes[0].errorbar(list(range(len(d))), d, yerr=std, label=f"{noise_value}%", marker='o', capsize=4)
    axes[0].legend()

    # Weights estimate
    axes[1].set_title("Distance to optimal weights")
    # axes[1].set_ylabel("$W_1$ estimate")
    axes[1].set_ylabel("$| w^* - w |$")
    axes[1].set_xlabel("Iteration")

    for w, std, noise_value in zip(all_weightsEstimates, std_weightsEstimates, noise_values):
        axes[1].errorbar(list(range(len(w))), w, yerr=std, label=f"{noise_value}%", marker='o', capsize=4)

    # axes[1].hlines(optimal_weight[1], xmin=0, xmax=len(w), label="optimal")
    axes[1].legend()

    f.savefig(f"experiments/{experiment_id}/noise_{optimal_weight}_{method}.png")


def plot_weight_estimations(results, optimal_weights):
    weight_estimations = np.array(results["weights"])
    plt.hlines(optimal_weights[1], xmin=0, xmax=len(weight_estimations), label="Optimal")
    plt.plot(np.array(weight_estimations)[:, 1], marker='o', label="Estimation")
    plt.legend()
    plt.show()


def plot_on_ternary_map(results, optimal_weights, env_name, method=None, experiment_id=None):
    weight_estimations = np.array(results["weights"])
    if env_name == "synt":
        with open('synthetic_pareto_front/pf.pickle', 'rb') as handle:
            background_points = pickle.load(handle)
    else:
        background_points = []

    # make the ternary plot
    figure, tax = ternary.figure(scale=100)
    figure.set_size_inches(50, 50)
    scatters = []
    for solution_index in background_points:
        sol, scatter_points = background_points[solution_index]

        # Round and plot
        float_formatter = "{:.3f}".format
        utility = float_formatter(np.dot(optimal_weights, np.array(sol)))
        sol = [float_formatter(x) for x in sol]
        scatter = tax.scatter(scatter_points, s=200, label=f"Utility: {utility}")
        scatters.append(scatter)

    figure.legend(loc='upper right', frameon=True, ncol=2)

    # plot the weights
    tax.plot(weight_estimations * 100, color="black", marker='o', label="Weights estimates")

    tax.scatter([weight_estimations[0] * 100], marker="s", zorder=np.inf, color="black", s=110)
    tax.scatter([weight_estimations[1] * 100], marker="s", zorder=np.inf, color="black", s=110)
    tax.scatter([np.array(optimal_weights) * 100], marker="X", zorder=np.inf, color="black", s=150,
                label="Optimal weights")

    # figure.legend(handles = others, loc='upper right', frameon=True, fontsize=11, ncol=1)

    tax.set_title("CCS")
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=5, color="blue")
    tax.ticks(axis='lbr', linewidth=1, multiple=5)
    tax.clear_matplotlib_ticks()
    tax.left_axis_label("$w_2$")
    tax.right_axis_label("$w_1$")
    tax.bottom_axis_label("$w_0$")
    tax.get_axes().axis('off')
    if experiment_id == None:
        tax.show()
    else:
        tax.savefig(f"experiments/{experiment_id}/{optimal_weights}_{method}.png")


def plot_2d_run(results, optimal_weights, method=None, experiment_id=None):
    weight_estimations = np.array(results["weights"])
    # stds =  np.array(results["stds"])[:, 1]
    trace = np.array(weight_estimations)[:, 1]

    num_iter = len(weight_estimations)
    y = np.arange(0, num_iter + 1, 0.01)
    weights_line = np.arange(0, 1.01, 0.01)

    figure, ax = plt.subplots(figsize=(15, 12))

    for i, sol in enumerate(BST_SOLUTIONS[::-1]):
        float_formatter = "{:.3f}".format
        utility = float_formatter(np.dot(optimal_weights, np.array(sol)))
        sol_string = [float_formatter(x) for x in sol]
        plt.scatter([], [], c=MATPLOTLIB_COLORS[i], label=f"Solution: {sol_string} \nUtility: {utility}")

        for time_weight in weights_line:
            full_weight = [1 - time_weight, time_weight]

            if list(get_best_sol_BST(full_weight)) == sol:
                plt.scatter(x=[time_weight] * len(y), y=y, c=MATPLOTLIB_COLORS[i], s=120)

    plt.axvline(optimal_weights[1], 0, num_iter, linestyle="dashed", color="black")

    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=-0.04, ymax=num_iter + 0.05)
    plt.yticks(list(range(len(trace))))
    plt.xlabel("$w_1$")
    plt.ylabel("iteration")

    # plt.plot(trace, list(range(len(trace))), marker='o', color="black")
    plt.errorbar(trace, list(range(len(trace))), marker='o', color="black", capsize=3)

    figure.tight_layout()
    figure.subplots_adjust(bottom=0.19)
    figure.legend(loc='lower center', frameon=True, fontsize=12, ncol=5, )
    # bbox_to_anchor=(0.5, -0.02)
    if experiment_id == None:
        plt.show()
    else:
        plt.savefig(f"experiments/{experiment_id}/{optimal_weights}_{method}.png")


def get_best_sol(pareto_front, weights):
    '''
    Get the best solution from a pareto front for a weight
    :param pareto_front:
    :param weights:
    :return:
    '''
    utility = lambda x: np.dot(weights, x)
    best_u = -np.inf
    best_sol = 0

    for sol in pareto_front:
        if utility(sol) > best_u:
            best_u = utility(sol)
            best_sol = sol

    return np.array(best_sol)


def get_best_sol_BST(weights):
    return get_best_sol(BST_SOLUTIONS, weights)
