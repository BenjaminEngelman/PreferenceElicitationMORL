from minecart.envs.minecart_env import MinecartDeterministicEnv
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt

env = MinecartDeterministicEnv()
state = env.reset()
terminal = False
i = 0
while not terminal:
    action = np.random.randint(0, 6)
    state, reward, terminal, info = env.step(action)
    print(state.shape)
    plt.imshow(env.render())
    plt.show()
    if i == 10:
        im = Image.fromarray(state)
        im.save("test.jpeg")
        exit()
    i+= 1

print("end")
