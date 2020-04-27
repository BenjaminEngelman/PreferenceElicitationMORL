import sys

from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.utils import MultiObjRewardWrapper, MinecartObsWrapper

from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C, DQN

import matplotlib.pyplot as plt

if __name__ == "__main__":
    weights = [0.0, 0.0, 1.0]
    env = MinecartObsWrapper(MinecartDeterministicEnv())
    model = A2C.load(f"saved_agents/{weights[0]}_{weights[1]}_{weights[2]}")
    obs = env.reset()
    env.render()
    dones = False
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            print("done")
            obs = env.reset()
        env.render()