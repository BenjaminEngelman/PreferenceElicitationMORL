import numpy as np

from src.agents import Qlearning
from src.RL_envs.deepSeaTreasures import BountyfulSeaTreasureEnv
from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.constants import STEPS_BST, STEPS_MINECART_COLD_START, N_ENVS_A2C
from src.utils import MinecartObsWrapper, MultiObjRewardWrapper
from src.utils import get_best_sol, get_best_sol_BST
from src.ols.utils import create_3D_pareto_front

from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from gym.wrappers import TimeLimit
import os

A2C_ARCH = [20, 20, 20]


def get_pretrained_agents():
    agents = []
    dir_name = "saved_agents"
    for filename in os.listdir(dir_name):
        agent = A2C.load(dir_name + '/' + filename)
        weights = np.array([float(w) for w in filename.split('_')])
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
    env = TimeLimit(env, max_episode_steps=10000)
    # env = DummyVecEnv([lambda: env])
    return env


def build_MO_minecart():
    env = MinecartDeterministicEnv()
    env = MinecartObsWrapper(env)
    env = TimeLimit(env, max_episode_steps=10000)
    return env

def highest_utility(results, w):
    utility = lambda x: np.dot(np.array(w), np.array(x))

    best_res = results[0]
    best_util = -np.inf

    for res in results:
        util = utility(res)
        if util > best_util:
            best_util = util
            best_res = res

    return best_res



class Solver(object):
    """
    Train and evaluate an agent in its environment
    """

    def __init__(self):
        self.n_calls = 0

    def eval_agent(self, agent, env_name, w, n_runs=1):
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
                tot_reward_mo = tot_reward_mo + \
                    reward * np.power(agent.gamma, cnt)
                cnt = cnt + 1

            results.append(tot_reward_mo)

        # Get the mode (the results observed the most)
        # res = most_occuring_sublist(results)
        res = highest_utility(results, w)

        return res

    def solve(self, env_name, weights, random_state=None):
        self.n_calls += 1

        if env_name == "bst":
            n_eval_runs = 1
            env = MultiObjRewardWrapper(BountyfulSeaTreasureEnv(), weights)
            learning_steps = STEPS_BST
            agent = Qlearning(env, decay=0.999997, random_state=random_state)
            
        elif env_name == "synt_bst":
            return get_best_sol_BST(weights)

        elif env_name[0:4] == "synt":
            env = create_3D_pareto_front(size=int(env_name.split('_')[-1]))
            return get_best_sol(env, weights)
        


        elif env_name == "minecart":
            n_eval_runs = 50

            trained_agents = get_pretrained_agents()
            # checkpoint_callback = CheckpointCallback(
            #     num_steps=N_STEPS_BEFORE_CHECKPOINT,
            #     save_path='saved_agents',
            #     name_prefix=f'{weights[0]}_{weights[1]}_{weights[2]}'
            # )

            # Train agent from scratch
            if len(trained_agents) == 0:
                learning_steps = STEPS_MINECART_COLD_START
                env = SubprocVecEnv([lambda i=i: build_SO_minecart(weights) for i in range(N_ENVS_A2C)])
                agent = A2C(MlpPolicy,
                            env,
                            vf_coef=0.5,
                            ent_coef=0.01,
                            n_steps=500//N_ENVS_A2C,
                            max_grad_norm=50,
                            # clip_loss_value=100,
                            learning_rate=3e-4,
                            gamma=0.99,
                            policy_kwargs={'net_arch': [
                                {'vf': A2C_ARCH, 'pi': A2C_ARCH}]},
                            # tensorboard_log="src/tensorboard/"
                            )

            # Get the most similar already trained agent
            else:
                most_similar_weights, agent = get_most_similar_agent(weights, trained_agents)
                learning_steps = STEPS_MINECART_COLD_START

                # If the most similar agent was trained for the same weights
                # we don't need to learn()

                if list(most_similar_weights) == list(weights):
                    fully_trained_agent = A2C.load(f'saved_agents/{most_similar_weights[0]}_{most_similar_weights[1]}_{most_similar_weights[2]}')
                    returns = self.eval_agent(fully_trained_agent, env_name, weights, n_runs=n_eval_runs)
                    return returns
                else:
                    env = SubprocVecEnv([lambda i=i: build_SO_minecart(weights) for i in range(N_ENVS_A2C)])
                    agent = A2C(MlpPolicy,
                                env,
                                vf_coef=0.5,
                                ent_coef=0.01,
                                n_steps=500//N_ENVS_A2C,
                                max_grad_norm=50,
                                # clip_loss_value=100,
                                learning_rate=3e-4,
                                gamma=0.99,
                                policy_kwargs={'net_arch': [
                                    {'vf': A2C_ARCH, 'pi': A2C_ARCH}]},
                                # tensorboard_log="src/tensorboard/"
                                )

        if env_name == "minecart":
            agent.learn(int(learning_steps))#, callback=checkpoint_callback)
            agent.save(f"saved_agents/{weights[0]}_{weights[1]}_{weights[2]}")
        else:
            agent.learn(learning_steps)
        returns = self.eval_agent(agent, env_name, weights, n_runs=n_eval_runs)

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
