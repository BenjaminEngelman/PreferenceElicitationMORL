import numpy as np
from numpy.random import RandomState
from agents import MOQlearning, MODQN
from env import BountyfulSeaTreasureEnv, DeepSeaTreasureEnv, OtherDeepSeaTreasure
from minecart.envs.minecart import Minecart
from minecart.envs.minecart_env import MinecartDeterministicEnv
from constants import STEPS_BST, STEPS_MINECART

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
            tot_reward_mo = tot_reward_mo + reward #* np.power(self.agent.gamma, cnt)
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
    from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

    
    # env = BountyfulSeaTreasureEnv()
    # agent = MOQlearning(env, decay=0.999997, random_state=random_state)
    env = MinecartDeterministicEnv()
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=False)
    agent = MODQN(env)

    solver = Solver()

    weights =  np.array([1., 0., 0.])
    solver.solve(agent, weights)
    agent.save(f"../saved_agents/minecart_{weights}")
    agent.demonstrate()


