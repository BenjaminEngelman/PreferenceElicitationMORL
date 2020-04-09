import numpy as np
import sys
from minecart.envs.minecart_env import MinecartDeterministicEnv
from src.utils import MultiObjRewardWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from src.utils import CustomPolicy

from stable_baselines import PPO2, A2C, DQN
from stable_baselines.common import make_vec_env


if __name__ == "__main__":
    weights =  np.array([0.99, 0.0, 0.01])
    env = MultiObjRewardWrapper(MinecartDeterministicEnv(), weights)

    if sys.argv[1] == "PPO":
        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="src/tensorboard/")
    elif sys.argv[1] == "A2C":
        model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="src/tensorboard/", n_steps=500)
    elif sys.argv[1] == "DQN":
        model = DQN(DQNMlpPolicy, env, verbose=1, tensorboard_log="src/tensorboard/")

    else:
        print("Wrong method")
        exit()

    model.learn(total_timesteps=50000000)
    model.save(f"{sys.argv[1]}_minecart")



