import random
import numpy as np
from utils import argmax

DIRECTION = ["U", "R", "D", "L"]

class Agent():
    
    def __init__(self, n_actions, decay, random_state):
        self.random_state = random_state

        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decaying_factor = decay
        self.n_actions = n_actions
        self.epsilon = self.max_epsilon
        self.lr = 0.1
        self.gamma = 0.99

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

class QLearningAgent(Agent):
    
    def __init__(self, n_actions, n_states, decay=0.999999, random_state=1):
        super().__init__(n_actions, decay=decay, random_state=random_state)
        self.n_states = n_states
        self.qtable = [np.array([0.]*self.n_actions) for state in range(n_states)]
        # self.qtable = [np.random.uniform(0, 100, n_actions) for state in range(n_states)]
    
    def reset(self):
        self.qtable = [np.array([0.]*self.n_actions) for state in range(self.n_states)]
        self.epsilon = self.max_epsilon
    
    def update(self, s, a, r, n_s):
        self.qtable[s][a] += self.lr * (r + self.gamma * (max(self.qtable[n_s])) - self.qtable[s][a])
        self.decayEpsilon()
    
    def get_action(self, state, greedy=False):
        q_values = self.qtable[state]
        if greedy:
            return argmax(q_values)
        else:
            return self.epsilonGreedy(q_values)
    
    def show_qtable(self):
        table = np.chararray((11, 10))
        for i in range(self.n_states):
            if (max(self.qtable[i])) != 0:
                table[i // 10, i % 10] = DIRECTION[argmax(self.qtable[i])]
            else:
                table[i // 10, i % 10] = "N"
        print(table)
        print(self.epsilon)
