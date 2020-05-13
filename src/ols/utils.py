import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState

from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])

def intersection(p1, p2, p3, p4):
    """
    Compute the intersection of the lines defined as
    (p1, p2) and (p3, p4)
    """
    # print(p1, p2, p3, p4)
    numX = (p1.x*p2.y-p1.y*p2.x)*(p3.x-p4.x)-(p1.x-p2.x)*(p3.x*p4.y-p3.y*p4.x)
    numY = (p1.x*p2.y-p1.y*p2.x)*(p3.y-p4.y)-(p1.y-p2.y)*(p3.x*p4.y-p3.y*p4.x)
    denum = (p1.x-p2.x)*(p3.y-p4.y)-(p1.y-p2.y)*(p3.x-p4.x)

    px = numX / denum
    py = numY / denum
    # return Point(round(px, 4), round(py, 4))
    return px, py

def plot_ccs(S, new_corners=[]):
    if len(S) > 1:
        f, ax = plt.subplots()
        for i, V_PI in enumerate(S):
            print(V_PI)
            x_vals = [V_PI.start.x, V_PI.end.x]
            y_vals = [V_PI.start.y, V_PI.end.y]        
            p = ax.plot([0, 1], [V_PI.obj1, V_PI.obj2], label=f"$V^{i}$")
            ax.plot(x_vals, y_vals, linewidth=4, color=p[0].get_color())
        if len(S) == 2:
            p = ax.plot([0, 1], [S[0].start.y, S[1].end.y], linestyle="dotted", color="black")
            inter = intersection(Point(0, S[0].start.y), Point(1,  S[1].end.y), Point(new_corners[0].val,new_corners[0].y), Point(new_corners[0].val, 40))
            p = ax.plot([new_corners[0].val, new_corners[0].val], [new_corners[0].y, inter[1]], linestyle="dashed", color="black")
            ax.text(0.65, 7.8, '$\Delta$')



        for j, corner in enumerate(new_corners):
            x = corner.val
            y = corner.y
            if j == 0:
                ax.scatter(x, y, marker='x', color='black', s=50, zorder=100, label="Corner weight")
            else:
                ax.scatter(x, y, marker='x', color='black', s=50, zorder=100)
                            

        # ax.set_title("CCS approximated by OLS")

        ax.set_xlabel("$w_1$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$V_w$")    
        plt.legend()

        # f.savefig(f"figures/ols_10.png")
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

    # f.savefig(f"figures/ols_10.png")
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

def create_3D_pareto_front(size=10, seed=42):
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
