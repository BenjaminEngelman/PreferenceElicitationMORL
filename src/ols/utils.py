import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState

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
        ax[0].plot(x_vals, y_vals, label=f"[{round(V_PI.start.x, 2)}, {round(V_PI.end.x, 2)}]")
        ax[1].scatter(V_PI.obj1, V_PI.obj2)
    
    # ax.set_title("CCS approximated by OLS")
    ax[0].set_xlabel("w1")
    ax[0].set_xlim([0, 1])
    ax[0].set_ylabel("Vw")    
    ax[0].legend()

    ax[1].set_xlabel("Treasure")
    ax[1].set_ylabel("Time")

    f.savefig(f"figures/ols_10.png")
    f.show()



def sample_point_on_positive_sphere(random_state):
    """
    Sample a point on the positive part of sphere
    ((x, y, z) are all positives)
    """
    ndim = 3
    vec = random_state.randn(ndim)
    vec /= np.linalg.norm(vec, axis=0)
    while not (vec>0).all():
        vec = random_state.randn(ndim)
        vec /= np.linalg.norm(vec, axis=0)
    return vec

def create_3D_pareto_front(size=100, seed=42):
    random_state = RandomState(seed)

    pareto_front = []

    while len(pareto_front) != size:
        x, y, z = sample_point_on_positive_sphere(random_state=random_state)
        if [x, y, z] not in pareto_front:
            pareto_front.append([x, y, z])

    return pareto_front

def plot_3D_pareto_front(pareto_front):
    xs = []
    ys = []
    zs = []
    for x, y, z in pareto_front:
        xs.append(x)
        ys.append(y)
        zs.append(z)
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.scatter(xs, ys, zs)
    plt.show()

def compare_3D_pareto_fronts(pf_1, pf_2):
    fig, ax = plt.subplots(1, 2, subplot_kw={'projection':'3d'})

    for i, pf in enumerate([pf_1, pf_2]):
        xs = []
        ys = []
        zs = []
        for x, y, z in pf:
            xs.append(x)
            ys.append(y)
            zs.append(z)
        ax[i].scatter(xs, ys, zs)

    ax[0].title.set_text('Original PF')
    ax[1].title.set_text('PF Discovred By OLS')
    plt.show()
