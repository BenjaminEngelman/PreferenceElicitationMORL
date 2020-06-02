import numpy as np
import sys
from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.utils import MultiObjRewardWrapper, MinecartObsWrapper, MinecartLessFuelMultiObjRewardWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from src.utils import CheckpointCallback

from stable_baselines import PPO2, A2C, DQN
from stable_baselines.logger import configure
from stable_baselines.common import make_vec_env
from gym.wrappers import TimeLimit
from stable_baselines.common import make_vec_env
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
    # configure logging dir and type
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv'
    os.environ['OPENAI_LOGDIR'] = f'runs/A2C_{str(uuid.uuid4())[:4]}'
    # configure logger
    configure()

    weights = np.array([float(x) for x in sys.argv[2:]])
    arch = [20, 20, 20]

    def make_env(env_n):
        env = MinecartDeterministicEnv()
        env = MinecartObsWrapper(env)
        env = MinecartLessFuelMultiObjRewardWrapper(env, weights)
        env = TimeLimit(env, max_episode_steps=1000)
        env = CSVLogger(env, os.environ['OPENAI_LOGDIR'] + f'_{env_n}')
        # env = DummyVecEnv([lambda: env])
        return env

    n_envs = 16
    env = SubprocVecEnv([lambda i=i: make_env(i) for i in range(n_envs)])

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
                    # full_tensorboard_log=True,
                    # tensorboard_log="src/tensorboard/",
                    # verbose=1,
        )

    else:
        print("Wrong method")
        exit()

    checkpoint_callback = CheckpointCallback(
        save_freq=int(625e5), save_path='checkpoints/',
        name_prefix=str(uuid.uuid4())[:4]
    )    
    model.learn(total_timesteps=int(12e7), callback=checkpoint_callback)
    model.save(f"saved_agents/{weights[0]}_{weights[1]}_{weights[2]}")
