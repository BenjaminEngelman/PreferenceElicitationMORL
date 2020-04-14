import numpy as np
from numpy.random import RandomState
from src.agents import Qlearning
from src.env import BountyfulSeaTreasureEnv
from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.constants import STEPS_BST, STEPS_MINECART
import matplotlib.pyplot as plt
from src.utils import MultiObjRewardWrapper
from src.utils import CustomPolicy
from stable_baselines import A2C

class Solver(object):
    """
    Train and evaluate an agent in its environment
    """
    def eval_agent(self, agent, env_name):
        if env_name == "bst":
            agent.env = BountyfulSeaTreasureEnv()
        elif env_name == "minecart":
            agent.env = MinecartDeterministicEnv()
            
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

    def solve(self, env_name, weights, random_state=None):
        if env_name == "bst":
            env = MultiObjRewardWrapper(BountyfulSeaTreasureEnv(), weights)
            learning_steps = STEPS_BST
            agent = Qlearning(env, decay=0.999997, random_state=random_state)

        elif env_name == "minecart":
            env = MultiObjRewardWrapper(MinecartDeterministicEnv(), weights)
            learning_steps = STEPS_MINECART
            agent = A2C(CustomPolicy, env, verbose=1, tensorboard_log="src/tensorboard/", n_steps=500)
        
        else:
            print("Cannot solve this environment.")
            exit(1)

        agent.learn(learning_steps)
        returns = self.eval_agent(agent, env_name)
        return returns
       

if __name__ == "__main__":    
    random_state = RandomState(42)  
    
    env = BountyfulSeaTreasureEnv()
    agent = Qlearning(env, decay=0.999997, random_state=random_state)
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

