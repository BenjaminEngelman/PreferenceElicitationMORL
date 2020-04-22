import sys

from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.utils import MultiObjRewardWrapper, MinecartObsWrapper

from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C, DQN

import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = MinecartObsWrapper(MinecartDeterministicEnv())
    model = A2C.load("A2C_2_hidden_layers")
    obs = env.reset()
    env.render()
    dones = False
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            obs = env.reset()
        env.render()