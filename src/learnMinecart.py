import numpy as np
import sys
from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.rl.minecartUtils import MinecartObsWrapper, MinecartMultiObjRewardWrapper
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import PPO2, A2C, DQN
from stable_baselines.logger import configure
from gym.wrappers import TimeLimit
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback

import csv
import os
import gym

class CSVLogger(gym.Wrapper):
    def __init__(self, env, logdir):
        super(CSVLogger, self).__init__(env)
        self._total_timesteps = 0
        try:
            os.mkdir(logdir)
        except FileExistsError:
            pass
        self._csv_path = logdir + '/log_progress.csv'
        with open(self._csv_path, 'w') as f:
            csv.writer(f, delimiter=',').writerow(
                ['total_steps', 'episode_steps', 'episode_reward'])

    def log_to_csv(self):
        with open(self._csv_path, 'a') as f:
            csv.writer(f, delimiter=',').writerow(
                [self._total_timesteps, self._episode_timesteps, self._episode_reward])

    def reset(self):
        obs = super(CSVLogger, self).reset()

        self._episode_timesteps = 0
        self._episode_reward = 0

        return obs

    def step(self, action):
        next_obs, reward, done, info = super(CSVLogger, self).step(action)

        self._episode_timesteps += 1
        self._episode_reward += reward

        if done:
            self._total_timesteps += self._episode_timesteps
            self.log_to_csv()
        return next_obs, reward, done, info


if __name__ == "__main__":

    import uuid

    weights = np.array([float(x) for x in sys.argv[2:5]])
    arch = [20, 20, 20]

    # configure logging dir and type
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv'
    os.environ['OPENAI_LOGDIR'] = f"runs/A2C_{str(uuid.uuid4())[:4]}_{weights}_{sys.argv[5].replace('.', '')}"
    # configure logger
    configure()

    def make_env(env_n, penalty_fac):
        env = MinecartDeterministicEnv()
        env = MinecartObsWrapper(env)
        env = MinecartMultiObjRewardWrapper(env, weights, penalty_fac)
        env = TimeLimit(env, max_episode_steps=1000)
        env = CSVLogger(env, os.environ['OPENAI_LOGDIR'] + f'_{env_n}')
        # env = DummyVecEnv([lambda: env])
        return env


    n_envs = 16


    if sys.argv[1] == "A2C":
        env = SubprocVecEnv([lambda i=i: make_env(i, float(sys.argv[5])) for i in range(n_envs)])

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
                    # full_tensorboard_log=True,
                    # tensorboard_log="src/tensorboard/",
                    # verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
        save_freq=int(50e5), save_path='checkpoints/',
        name_prefix=str(uuid.uuid4())[:4] + str(weights) + sys.argv[5].replace('.', '')
        )

    # elif sys.argv[1] == "DQN":
    #     env = make_env(0)
    #     env = DummyVecEnv([lambda: env])

    #     model = DQN(
    #         DQNMlpPolicy,
    #         env,
    #         verbose=1,
    #         # train_freq=500,
    #         # tensorboard_log="src/tensorboard/",
    #         gamma=0.98,
    #         prioritized_replay=True,
    #         policy_kwargs={'layers': arch},
    #         learning_rate=3e-4,
    #         # exploration_final_eps=0.01,

    #     )   

    #     checkpoint_callback = CheckpointCallback(
    #     save_freq=int(10e6), save_path='checkpoints/',
    #     name_prefix=str(uuid.uuid4())[:4] + str(weights)
    #     )

    else:
        print("Wrong method")
        exit()

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=int(625e5), save_path='checkpoints/',
    #     name_prefix=str(uuid.uuid4())[:4]
    # )
    model.learn(total_timesteps=int(12e7), callback=checkpoint_callback)
    model.save(f"saved_agents_last/{weights[0]}_{weights[1]}_{weights[2]}_{sys.argv[5].replace('.', '')}_ok")
