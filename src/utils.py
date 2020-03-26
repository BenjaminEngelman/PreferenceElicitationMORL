import matplotlib.pyplot as plt
import numpy as np
from constants import BST_SOLUTIONS 

def get_best_sol_BST(weights):
    utility = lambda x: weights[0] * x[0] + weights[1] * x[1]

    best_u = -1000
    best_sol = 0

    for sol in BST_SOLUTIONS:
        if utility(sol) > best_u:
            best_u = utility(sol)
            best_sol = sol

    return best_sol[0]

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

def plot_ccs(S):
    print(S)
    f, ax = plt.subplots()
    for V_PI in S:
        x_vals = [V_PI.start.x, V_PI.end.x]
        y_vals = [V_PI.start.y, V_PI.end.y]        
        ax.plot(x_vals, y_vals)
    
    # ax.set_title("CCS approximated by OLS")
    ax.set_xlabel("w1")
    ax.set_ylabel("Vw")    
    plt.show()

def plot_ccs_2(S):
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    for V_PI in S:
        x_vals = [V_PI.start.x, V_PI.end.x]
        y_vals = [V_PI.start.y, V_PI.end.y]        
        ax[0].plot(x_vals, y_vals)
        ax[1].scatter(V_PI.obj2, V_PI.obj1)
    
    # ax.set_title("CCS approximated by OLS")
    ax[0].set_xlabel("w1")
    ax[0].set_xlim([0, 1])
    ax[0].set_ylabel("Vw")    

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Treasure")

    f.savefig(f"figures/ols.png")


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
    axes[1].set_ylabel("W0 estimate")

    axes[1].plot(weights_withComp, label="Comparaisons", marker='o')
    axes[1].plot(weights_withAbsRet, label="Absolute return", marker='o')
    axes[1].hlines(optimal_weight, xmin=0, xmax=len(weights_withAbsRet), label="optimal")
    axes[1].legend()

    f.savefig(f"../figures/comp_methods_{optimal_weight}_noise_{noise}.png")    

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
    axes[1].set_ylabel("W0 estimate")
    for w, std, noise_value in zip(all_weightsEstimates, std_weightsEstimates, noise_values):
        axes[1].errorbar(list(range(len(w))), w, yerr=std, label=noise_value, marker='o')

    axes[1].hlines(optimal_weight, xmin=0, xmax=8, label="optimal")
    axes[1].legend()

    f.savefig(f"../figures/noise_{optimal_weight}_{method}.png")

         


    # axes[0].title(f"Noise experiment - Weight = {weight_vector} - {method} method")
