# BST_MAX_TREASURE = 173
# BST_MAX_TIME = -17

# BST_SOLUTIONS = [
#     [5, -1],
#     [80, -3],
#     [120, -5],
#     [140, -7],
#     [145, -8],
#     [150, -9],
#     [163, -13],
#     [166, -14],
#     [173, -17],
#     [175, -19],
# ]

GAMMA_BST = 0.95

BST_MAX_TREASURE = 100 * GAMMA_BST**19
BST_MAX_TIME = sum([-1 * GAMMA_BST**i for i in range(20)])

BST_SOLUTIONS = [
    [18 * GAMMA_BST**1, sum([-1 * GAMMA_BST**i for i in range(1)])],
    [26 * GAMMA_BST**2, sum([-1 * GAMMA_BST**i for i in range(2)])],
    [31 * GAMMA_BST**3, sum([-1 * GAMMA_BST**i for i in range(3)])],
    [44 * GAMMA_BST**6, sum([-1 * GAMMA_BST**i for i in range(6)])],
    [48.2 * GAMMA_BST**7, sum([-1 * GAMMA_BST**i for i in range(7)])],
    [56 * GAMMA_BST**9, sum([-1 * GAMMA_BST**i for i in range(9)])],
    [72 * GAMMA_BST**13, sum([-1 * GAMMA_BST**i for i in range(13)])],
    [76.3 * GAMMA_BST**14, sum([-1 * GAMMA_BST**i for i in range(14)])],
    [90 * GAMMA_BST**17, sum([-1 * GAMMA_BST**i for i in range(17)])],
    [100 * GAMMA_BST**19, sum([-1 * GAMMA_BST**i for i in range(19)])],
]

BST_DIRECTIONS = ["U", "R", "D", "L"]

# Learning constants
STEPS_BST = 110_000

STEPS_MINECART_COLD_START = 80_000_000
N_STEPS_BEFORE_CHECKPOINT = 15_000_000
STEPS_MINECART_HOT_START = 65_000_000

# Experiments constants

# Each of those weights have another optimal solution for 
# syntetic PF created by create_3D_pareto_front() in src.ols.utils
# These are used for the "comparison" experiment
WEIGHTS_COMP_SYNT = [
    [0.05, 0.77, 0.18],
    [0.8, 0.06, 0.14],
    [0.07, 0.77, 0.16],
    [0.11, 0.32, 0.57],
    [0.34, 0.64, 0.02],
    [0.21, 0.4, 0.39],
    [0.08, 0.04, 0.88],
    [0.32, 0.4, 0.28],
    [0.18, 0.44, 0.38],
    [0.64, 0.05, 0.31],
]

# Each of those weights have another optimal solution for 
# the BST environment
# These are used for the "comparison" experiment
WEIGHTS_COMP_BST = [
    [0.03, 0.97],
    [0.22, 0.78],
    [0.23, 0.77],
    [0.32, 0.68],
    [0.36, 0.64],
    [0.49, 0.51],
    [0.64, 0.36],
    [0.74, 0.26],
    [0.8, 0.2],
    [0.99, 0.01],
]

# These are used for the "noise" experiment
WEIGHTS_NOISE_SYNT = [
    [0.64, 0.05, 0.31],
    [0.32, 0.4, 0.28],
    [0.07, 0.77, 0.16],
]

WEIGHTS_NOISE_BST = [
    [1.0, 0.0],
    [0.5, 0.5],
    [0.1, 0.9]
]

WEIGHTS_NOISE_MINECART = [
    [0.8, 0.0, 0.2],
    [0.0, 0.9, 0.1],
    [0.1, 0.0, 0.9]
]