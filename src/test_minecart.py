from minecart.envs.minecart_env import MinecartDeterministicEnv
import numpy as np


env = MinecartDeterministicEnv()
state = env.reset()
terminal = False
while not terminal:
    action = np.random.randint(0, 6)
    state, reward, terminal, info = env.step(action)
    print(reward)
    env.render()
print("end")
