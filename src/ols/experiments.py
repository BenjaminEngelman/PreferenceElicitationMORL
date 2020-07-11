from src.ols.main_2_objectives import ols
from numpy.random import RandomState
from src.constants import BST_SOLUTIONS



def run_BST():
    """
    Run OLS on Boutyfull Sea Treasure
    """
    env = "synt_bst"
    random_state = RandomState(42)
    print(BST_SOLUTIONS)
    ols(env, random_state)


# def run_minecart():
#     """
#     Run OLS on Minecart
#     """
#     env = BountyfulSeaTreasureEnv()
#     n_actions = env.nA
#     n_states = env.nS
#     random_state = RandomState(42)

#     agent = QLearningAgent(
#         n_actions=n_actions,
#         n_states=n_states,
#         decay=0.999997,
#         random_state=random_state
#     )
#     ols(env, agent)

if __name__ == "__main__":
    run_BST()