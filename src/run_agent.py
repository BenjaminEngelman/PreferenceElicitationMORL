import sys

from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.utils import MultiObjRewardWrapper, MinecartObsWrapper

from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C, DQN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    weights = [0.0, 0.0, 1.0]
    env = MinecartObsWrapper(MinecartDeterministicEnv())
    agent = A2C.load(f"test_parallel.zip")
    results = []

    for _ in range(500):
        
        state = env.reset()
        env.render()

        cnt = 0
        tot_reward_mo = 0
        terminate = False

        while not terminate:
            action, _ = agent.predict(state)
            state, reward, terminate, _ = env.step(action)
            env.render()
            if cnt > 200:
                terminate = True
            tot_reward_mo = tot_reward_mo + reward * np.power(0.98, cnt)
            cnt = cnt + 1

        results.append(np.round(tot_reward_mo, 2))

    print(np.mean(np.array(results), axis=0))