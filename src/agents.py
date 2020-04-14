import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from src.utils import argmax
from stable_baselines import DQN
from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq.build_graph import build_train
from stable_baselines.deepq.policies import DQNPolicy, MlpPolicy, CnnPolicy
from src.constants import BST_DIRECTIONS, GAMMA_BST
import sklearn

class Agent():
    def __init__(self, env, decay, random_state):
        self.random_state = random_state
        self.env = env

        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decaying_factor = decay
        self.n_actions = env.nA
        self.epsilon = self.max_epsilon
        self.lr = 0.1
        self.gamma = GAMMA_BST

    def epsilonGreedy(self, q_values):
        a = argmax(q_values)
        if self.random_state.uniform(0, 1) < self.epsilon:
            a = self.random_state.randint(self.n_actions)
        return a
    
    def decayEpsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decaying_factor
    
    def reset(self):
        raise NotImplementedError()   



class Qlearning(Agent):
    
    def __init__(self, env, decay=0.999999, random_state=1):
        super().__init__(env, decay=decay, random_state=random_state)
        self.n_states = env.nS
        self.n_actions = env.nA

        # self.qtable = [np.array([0.]*self.n_actions) for state in range(n_states)]
        self.qtable = [np.random.uniform(-1, 175, self.n_actions) for state in range(self.n_states)]
    
    def learn(self, n_episodes):
        learning_stats = {
            "average_cumulative_reward": [],
            "cumulative_reward": [],
        }

        average_cumulative_reward = 0.0
        # Loop over episodes
        for i in range(n_episodes):
            # if i % 1000 == 0:
            #     print(f"Episode {i}")
            state = self.env.reset()
            terminate = False
            cumulative_reward = 0.0

            # Loop over time-steps
            step = 0
            while not terminate:
                # Get the action
                action = self.get_action(state)

                # Perform the action
                next_state, reward, terminate, _ = self.env.step(action)
                
                # Update agent
                self.update( state, action, reward, next_state, terminate )

                # Update statistics
                cumulative_reward += reward
                state = next_state
                step += 1

            # Per-episode statistics
            average_cumulative_reward *= 0.95
            average_cumulative_reward += 0.05 * cumulative_reward

            ep_stats = [i, cumulative_reward, average_cumulative_reward]

            learning_stats["cumulative_reward"].append(ep_stats[1])
            learning_stats["average_cumulative_reward"].append(ep_stats[2])

        return learning_stats
    
    def reset(self):
        """
        Reset the Qtable and the epsilon
        """
        self.qtable = [np.random.uniform(-1, 175, self.n_actions) for state in range(self.n_states)]
        self.epsilon = self.max_epsilon
    
    def update(self, s, a, r, n_s, d):
        """
        Q-learning update rule 
        Epsilon decay
        """
        target = r if d else r + self.gamma * (max(self.qtable[n_s]))
        self.qtable[s][a] += self.lr * (target - self.qtable[s][a])
        self.decayEpsilon()
    
    def get_action(self, state):
        """
        Epsilon-Greedily select action for a state 
        """
        q_values = self.qtable[state]
        return self.epsilonGreedy(q_values)
    
    def predict(self, state):
        """
        Greedily select action for a state
        """
        q_values = self.qtable[state]
        return argmax(q_values), '' # for compatibilty
    
    def demonstrate(self):
        """
        Demonstrate the learned policy
        """
        state = self.env.reset()
        terminate = False
        step = 0
        while not terminate or step < 500:
            plt.imshow(self.env.render())
            plt.show()
            action = self.predict(state)
            state, _, terminate, _ = self.env.step(action)
            step += 1
        plt.imshow(self.env.render())
        


    def show_qtable(self):
        table = np.chararray((12, 12))
        for i in range(self.n_states):
            if (max(self.qtable[i])) != 0:
                table[i // 12, i % 12] = BST_DIRECTIONS[argmax(self.qtable[i])]
            else:
                table[i // 12, i % 12] = "N"
        print(table)
        print(self.epsilon)


if __name__ == "__main__":
    from minecart.envs.minecart_env import MinecartDeterministicEnv

    env = MinecartDeterministicEnv()
    weights = np.array([0.99, 0.0, 0.01])
    s = env.reset()
    agent = MODQN(env)
    agent.learn(10000, weights)


    