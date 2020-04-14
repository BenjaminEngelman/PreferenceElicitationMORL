import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def triangle_for_point(r):
    r = np.array(r)
    w = np.identity(len(r)).T
    u = np.inner(w,r)

    triang = tri.Triangulation(w[:,0], w[:,1], triangles=np.arange(3)[None,:])
    return triang, u

def intersection(points):
    p = np.array(points)
    # r1*w1 + r2*w2 + r3*(1-w1-w2) = u. Unknowns: w1, w2, u
    a = np.array((p[:,0]-p[:,2], p[:,1]-p[:,2], -np.ones(p.shape[0]))).T
    b = -p[:,2]
    x = np.linalg.solve(a, b)
    return x

def plot():
    points = [(1, 1, 8), (7, 2, 1), (2, 7, 3), (7, 7, 8)]
    colors = ['red', 'green', 'blue', 'yellow']

    fig = plt.figure()
    plt.gca().set_aspect('equal')
    ax = plt.axes(projection='3d')
    for i in range(len(points)):
        triang, u = triangle_for_point(points[i])
        ax.plot_trisurf(triang, u, color=colors[i], alpha=0.4)
    # compute intersection
    w1, w2, u = intersection(points[:3])
    print(w1, w2, u)
    ax.scatter([w1], [w2], [u], c='orange', s=70,)
    ax.text(w1, w2, u, 'intersection')
    plt.show()


if __name__ == "__main__":
    plot()    