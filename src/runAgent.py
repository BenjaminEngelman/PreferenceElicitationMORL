import sys

from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.utils import MultiObjRewardWrapper, MinecartObsWrapper, MinecartMultiObjRewardWrapper

from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C, DQN
import numpy as np
import matplotlib.pyplot as plt

# weights = [0.35,0.08,0.5700000000000001]
weights = np.array([0.75,0.1,0.15])
def utility(res):
    return np.dot(res, weights)


if __name__ == "__main__":
    env = MinecartObsWrapper(MinecartDeterministicEnv())
    # agent = A2C.load(f"test_parallel.zip")
    # 0.75_0.1_0.15_002
    agent = A2C.load(f"agents_last/0.75_0.1_0.15_005_ok")
    

    results = []

    for _ in range(500):
        
        state = env.reset()
        env.render()

        cnt = 0
        tot_reward_mo = 0
        terminate = False

        while not terminate:
            action, _ = agent.predict(state)
            # action = env.action_space.sample()
            state, reward, terminate, _ = env.step(action)
            env.render()
            if cnt > 1000:
                terminate = True
            # tot_reward_mo = tot_reward_mo + reward * np.power(0.98, cnt)
            cnt = cnt + 1

        # print(tot_reward_mo, utility(tot_reward_mo))

    # print(np.mean(np.array(results), axis=0))