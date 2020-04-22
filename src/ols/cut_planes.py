# # # def findRegions():
# # # 	planes = []
# # # 	regions = []

# # # 	for i in range(100):
# # # 		new_plane = create_plane()
# # # 		region_new_plane = [
# # # 			Constraint(x1 > 0),
# # # 			Constraint(x2 > 0)
# # # 			Constraint(x3 > 0),
# # # 			Constraint(x1 + x2 + x3 = 0)
# # # 		]

# # # 		for j in range(len(planes)):
# # # 			other_plane = planes[j]
# # # 			other_region = regions[j]   

# # # 			f = plane[j] INTERSECTION new_plane  # f est la fonction qui définit la ligne 
# # # 												 # la ligne ce sont tous les points où f(p) = 0
# # # 			p = point such that f(p) > 0
# # # 			if new_plane(p) > plane[j](p):
# # # 				region_new_plane.add_contraint(Constraint(f(x) > 0))
# # # 				regions[j].add_contraint(Constraint(f(x) < 0))
# # # 			else:
# # # 				region_new_plane.add_contraint(Constraint(f(x) < 0))
# # # 				regions[j].add_contraint(Constraint(f(x) > 0))

		
# # # 		planes.append(new_plane)
# # # 		regions.append(region_new_plane)

# # from sympy import Point3D, Plane
# # from sympy.plotting import plot, plot3d, plot3d_parametric_surface
# # import random

# # from sympy import symbols
# # from sympy.plotting import plot3d
# # x, y = symbols('x y')
# # plot3d(x*y, (x, 0, 1), (y, 0, 1))

# # # plane_1 = Plane(Point3D(1, 0, 1), Point3D(0, 1, 1), Point3D(0, 0, 8))
# # # plane_2 = Plane(Point3D(1, 0, 7), Point3D(0, 1, 2), Point3D(0, 0, 1))

# # # print(plane_1.equation())
# # # print(plane_2.equation().)

# # # intersec = plane_1.intersection(plane_2)
# # # print(intersec[0].equation())

# # # x = random.randrange(start=0, stop=1)
# # # y = random.randrange(start=m*x+b, stop=1)
# # # print(x, y)

# # # print(intersec)
# # # intersec[0].plot_interval()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# points1 = [[1, 0, 1],
#            [0, 1, 1],
#            [0, 0, 8]]

# points2 = [[1, 0, 7],
#            [0, 1, 2],
#            [0, 0, 1]]

# def get_plane_from_points(points):

#     p0, p1, p2 = points
#     x0, y0, z0 = p0
#     x1, y1, z1 = p1
#     x2, y2, z2 = p2

#     ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
#     vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

#     u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

#     point  = np.array(p0)
#     normal = np.array(u_cross_v)

#     d = -point.dot(normal)

#     xx, yy = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
    
#     z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    
#     return xx, yy, z

# xx, yy, z = get_plane_from_points(points1)
# xx2, yy2, z2 = get_plane_from_points(points2)


# # plot the surface
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, z, alpha=0.4, color="red")
# plt3d.plot_surface(xx2, yy2, z2, alpha=0.4, color="green")

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# p1 = (7, 7, 1, 8)
# p2 = (-6, -1, 1, 1)

# def get_plan(p):
#     a, b, c, d = p
#     x = np.linspace(0,1,100)
#     y = np.linspace(0,1,100)

#     X,Y = np.meshgrid(x,y)
#     Z = (d - a*X - b*Y) / c
#     return X, Y, Z



# def plane_intersect(a, b):
#     """
#     a, b   4-tuples/lists
#            Ax + By +Cz + D = 0
#            A,B,C,D in order  

#     output: 2 points on line of intersection, np.arrays, shape (3,)
#     """
#     a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

#     aXb_vec = np.cross(a_vec, b_vec)

#     A = np.array([a_vec, b_vec, aXb_vec])
#     d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

# # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

#     p_inter = np.linalg.solve(A, d).T

#     return p_inter[0], (p_inter + aXb_vec)[0]


# a, b = (-6,-1,1,1), (7,7,1,8)
# intersec = plane_intersect(a, b)
# print(intersec)

# X, Y, Z = get_plan(p1)
# X2, Y2, Z2 = get_plan(p2)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, alpha=0.4)
# ax.plot_surface(X2, Y2, Z2, alpha=0.4)
# ax.scatter([intersec[0][0], intersec[1][0]], [intersec[0][1], intersec[1][1]], [intersec[0][2], intersec[1][2]])
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

# (1, 0, 1), (0, 1, 1), (0, 0, 8)

# ax.plot_trisurf([1, 0, 0], [0, 1, 0], [1, 1, 8])
ax.plot_trisurf([1, 0, 7], [0, 1, 2], [1, 1, 1])

plt.show()