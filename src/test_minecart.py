from minecart.envs.minecart_env import MinecartDeterministicEnv
from stable_baselines.common.env_checker import check_env

import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt

env = MinecartDeterministicEnv()
# check_env(env)
state = env.reset()
terminal = False
i = 0
while not terminal:
    action = np.random.randint(0, 6)
    state, reward, terminal, info = env.step(action)
    print(state)
    i+= 1

# print("end")
