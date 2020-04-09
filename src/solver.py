import numpy as np
from numpy.random import RandomState
from src.agents import MOQlearning, MODQN
from src.env import BountyfulSeaTreasureEnv
from minecart.envs.minecart import Minecart
from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.constants import STEPS_BST, STEPS_MINECART
import matplotlib.pyplot as plt

class Solver(object):
    """
    Train and evaluate an agent in its environment
    """
    def eval_agent(self, agent):
        state = agent.env.reset()

        cnt = 0
        tot_reward_mo = 0
        terminate = False

        while not terminate:
            action, _ = agent.predict(state)
            state, reward, terminate, _ = agent.env.step(action)

            if cnt > 1000:
                terminate = True
            tot_reward_mo = tot_reward_mo + reward * np.power(agent.gamma, cnt)
            cnt = cnt + 1


        return tot_reward_mo

    def solve(self, agent, weights):
        if isinstance(agent.env, BountyfulSeaTreasureEnv):
            learning_steps = STEPS_BST
        else:
            learning_steps = STEPS_MINECART
            
        agent.learn(learning_steps, weights)
        V = self.eval_agent(agent)
        return V
       

if __name__ == "__main__":    
    random_state = RandomState(42)  
    
    env = BountyfulSeaTreasureEnv()
    agent = MOQlearning(env, decay=0.999997, random_state=random_state)
    # env = MinecartDeterministicEnv()
    # env = VecNormalize(env, norm_obs=True, norm_reward=False)
    # agent = MODQN(env)

    solver = Solver()

    # weights =  np.array([0.99, 0.0, 0.01])
    weights = np.array([1, 0])
    solver.solve(agent, weights)
    # agent.save(f"../saved_agents/minecart_{weights}")
    # agent.demonstrate()
    plt.imshow(env.render())
    plt.show()

