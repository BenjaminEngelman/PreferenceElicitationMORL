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

BST_MAX_TREASURE = 100 * GAMMA_BST**20
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
STEPS_MINECART_COLD_START = 70_000_000
# STEPS_MINECART_COLD_START = 10

STEPS_MINECART_HOT_START = 35_000_000
# STEPS_MINECART_HOT_START = 10

