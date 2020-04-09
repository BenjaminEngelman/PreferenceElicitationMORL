import sys

from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.utils import MultiObjRewardWrapper

from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = MinecartDeterministicEnv()
    model = PPO2.load(f"{sys.argv[1]}_minecart")
    obs = env.reset()
    env.render()
    dones = False
    while not dones:
        action, _states = model.predict(obs)
        print(action)   
        obs, rewards, dones, info = env.step(action)
        env.render()
        print("CART CONTENT")
        print(env.cart.content)