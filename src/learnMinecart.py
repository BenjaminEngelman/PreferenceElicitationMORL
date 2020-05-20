import numpy as np
import sys
from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.utils import MultiObjRewardWrapper, MinecartObsWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from src.utils import CheckpointCallback

from stable_baselines import PPO2, A2C, DQN
from stable_baselines.common import make_vec_env
from gym.wrappers import TimeLimit
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv



if __name__ == "__main__":

    weights = np.array([float(x) for x in sys.argv[2:]])
    arch = [64, 64]
    def make_env():
        env = MinecartDeterministicEnv()
        env = MinecartObsWrapper(env)
        env = MultiObjRewardWrapper(env, weights)
        env = TimeLimit(env, max_episode_steps=10000)
        # env = DummyVecEnv([lambda: env])
        return env

    n_envs = 8
    env = SubprocVecEnv([lambda i=i: make_env() for i in range(n_envs)])


    # if sys.argv[1] == "PPO":
    #     model = PPO2(
    #         MlpPolicy,
    #         env,
    #         verbose=1,
    #         gamma=0.98,
    #         max_grad_norm=50,
    #         n_steps=500,
    #         learning_rate=3e-4,
    #         policy_kwargs={'net_arch': [{'vf': [128], 'pi': [128]}]},
    #         tensorboard_log="src/tensorboard/")

    if sys.argv[1] == "A2C":
        model = A2C(MlpPolicy,
                    env,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    n_steps=500//n_envs,
                    max_grad_norm=50,
                    # clip_loss_value=100,
                    learning_rate=3e-4,
                    gamma=0.98,
                    policy_kwargs={'net_arch': [{'vf': arch, 'pi': arch}]},
                    tensorboard_log="src/tensorboard/"
        )
   
    elif sys.argv[1] == "DQN":
        model = DQN(
            DQNMlpPolicy,
            env,
            verbose=1,
            train_freq=500,
            tensorboard_log="src/tensorboard/",
            gamma=0.98,
            prioritized_replay=True,
            policy_kwargs={'layers': arch},
            learning_rate=3e-4,
            exploration_final_eps=0.01,

        )   

    else:
        print("Wrong method")
        exit()

    # checkpoint_callback = CheckpointCallback(num_steps=15_000_000, save_path='./saved_agents', name_prefix=f'{weights[0]}_{weights[1]}_{weights[2]}')
    model.learn(total_timesteps=int(7e6))
    model.save("test_parallel")
    # model.save(f"saved_agents/{weights[0]}_{weights[1]}_{weights[2]}")
    