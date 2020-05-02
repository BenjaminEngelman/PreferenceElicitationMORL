import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

from src.agents import Qlearning
from src.env import BountyfulSeaTreasureEnv
from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.constants import STEPS_BST, STEPS_MINECART_COLD_START, STEPS_MINECART_HOT_START, N_STEPS_BEFORE_CHECKPOINT
from src.utils import MinecartObsWrapper, MultiObjRewardWrapper, most_occuring_sublist
from src.utils import get_best_sol, CheckpointCallback
from src.ols.utils import create_3D_pareto_front


from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from gym.wrappers import TimeLimit
import os

A2C_ARCH = [64, 64]

def get_pretrained_agents():
    agents = []
    dir_name = "saved_agents"
    for filename in os.listdir(dir_name):
        if filename.split['_'][-1] == "checkpoint":
            agent = A2C.load(dir_name + '/' + filename)
            weights = np.array([float(w) for w in filename.split('_')[:-1]])
            agents.append([weights, agent])

    return agents

def get_most_similar_agent(weights, trained_agents):
    min_dist = np.inf
    most_similar_agent = None
    most_similar_weights = None

    for agent in trained_agents:
        agent_weight = agent[0]
        dist = np.linalg.norm(agent_weight-weights)
        if dist < min_dist:
            min_dist = dist
            most_similar_agent = agent[1]
            most_similar_weights = agent_weight
    
    return most_similar_weights, most_similar_agent



def build_SO_minecart(weights):
    env = MinecartDeterministicEnv()
    env = MinecartObsWrapper(env)
    env = MultiObjRewardWrapper(env, weights)
    env = TimeLimit(env, max_episode_steps=1000)
    env = DummyVecEnv([lambda: env])
    return env

def build_MO_minecart():
    env = MinecartDeterministicEnv()
    env = MinecartObsWrapper(env)
    env = TimeLimit(env, max_episode_steps=1000)
    return env


class Solver(object):
    """
    Train and evaluate an agent in its environment
    """
    def eval_agent(self, agent, env_name, n_runs=1):
        if env_name == "bst":
            agent.env = BountyfulSeaTreasureEnv()
        elif env_name == "minecart":
            agent.env = build_MO_minecart()
        
        results = []

        for _ in range(n_runs):
            
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

            results.append(np.round(tot_reward_mo, 2))
        
        # Get the mode (the results observed the most)
        res = most_occuring_sublist(results)

        return res

    def solve(self, env_name, weights, random_state=None):
        if env_name == "bst":
            n_eval_runs = 1
            env = MultiObjRewardWrapper(BountyfulSeaTreasureEnv(), weights)
            learning_steps = STEPS_BST
            agent = Qlearning(env, decay=0.999997, random_state=random_state)
        
        elif env_name == "synt":
            env = create_3D_pareto_front(10)
            return get_best_sol(env, weights)


        elif env_name == "minecart":
            n_eval_runs = 100
            env = build_SO_minecart(weights)
            trained_agents = get_pretrained_agents()
            checkpoint_callback = CheckpointCallback(
                num_steps=N_STEPS_BEFORE_CHECKPOINT,
                save_path='saved_agents',
                name_prefix=f'{weights[0]}_{weights[1]}_{weights[2]}'
            )


            # Train agent from scratch
            if len(trained_agents) == 0:
                learning_steps = STEPS_MINECART_COLD_START
                agent = A2C(MlpPolicy,
                    env,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    n_steps=500,
                    max_grad_norm=50,
                    # clip_loss_value=100,
                    learning_rate=3e-4,
                    gamma=0.98,
                    policy_kwargs={'net_arch': [{'vf': A2C_ARCH, 'pi': A2C_ARCH}]},
                    # tensorboard_log="src/tensorboard/"
                )

            # Get the most similar already trained agent
            else:
                most_similar_weights, agent = get_most_similar_agent(weights, trained_agents)
                learning_steps = STEPS_MINECART_HOT_START

                # If the most similar agent was trained for the same weights
                # we don't need to learn()
                if list(most_similar_weights) == list(weights):
                    returns = self.eval_agent(agent, env_name)
                    return returns
                else:
                    agent.set_env(env)

        agent.learn(learning_steps, callback=checkpoint_callback)
        agent.save(f"saved_agents/{weights[0]}_{weights[1]}_{weights[2]}")
        returns = self.eval_agent(agent, env_name, n_runs=n_eval_runs)

        return returns
       

# if __name__ == "__main__":    
    # random_state = RandomState(42)  
    
    # env = BountyfulSeaTreasureEnv()
    # agent = Qlearning(env, decay=0.999997, random_state=random_state)
    # # env = MinecartDeterministicEnv()
    # # env = VecNormalize(env, norm_obs=True, norm_reward=False)
    # # agent = MODQN(env)

    # solver = Solver()

    # # weights =  np.array([0.99, 0.0, 0.01])
    # weights = np.array([1, 0])
    # solver.solve(agent, weights)
    # # agent.save(f"../saved_agents/minecart_{weights}")
    # # agent.demonstrate()
    # plt.imshow(env.render())
    # plt.show()

