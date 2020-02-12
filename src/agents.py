import random
import numpy as np
from utils import argmax

class Agent():
    
    def __init__(self, n_actions, epsilon):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.lr = 0.001
        self.gamma = 0.9

    def epsilonGreedy(self, q_values):
        a = argmax(q_values)
        if random.random() < self.epsilon:
            a = random.randrange(self.n_actions)
        return a
    
    def reset(self):
        raise NotImplementedError()   

class QLearningAgent(Agent):
    
    def __init__(self, n_actions, epsilon, n_states):
        super().__init__(n_actions, epsilon)
        self.n_states = n_states
        self.qtable = [np.array([0.]*self.n_actions) for state in range(n_states)]
    
    def reset(self):
        self.qtable = [np.array([0.]*self.n_actions) for state in range(self.n_states)]
    
    def update(self, s, a, r, n_s):
        self.qtable[s][a] += self.lr * (r + self.gamma * (max(self.qtable[n_s])) - self.qtable[s][a])
    
    def get_action(self, state):
        q_values = self.qtable[state]
        return self.epsilonGreedy(q_values)