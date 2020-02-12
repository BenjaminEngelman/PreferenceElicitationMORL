import numpy as np
from agents import Agent

EPISODES = 50

class Solver(object):
    def __init__(self, modp, agent):
        self.env = modp
        self.agent = agent
    
    def train_agent(self, weights):
        # Loop over episodes
        for _ in range(EPISODES):
            state = self.env.reset()
            terminate = False

            # Loop over time-steps
            while not terminate:

                # Getthe action
                action = self.agent.get_action(state)

                # Perform the action
                next_state, reward, terminate = self.env.step(a)
                
                actual_r = weights.dot(reward)

                # Update agent
                self.agent.update(state, action, actual_r, next_state)

                # Update statistics
                cumulative_reward += actual_r
                state = next_state
    
    def eval_agent(self):
        state = self.env.reset()
        cnt = 0
        tot_reward_mo = 0
        terminate = False

        while not terminate:
            action = self.agent.get_action(state)
            _, reward, terminate = self.env.step(action)

            if cnt > 100:
                terminate = True
            tot_reward_mo = tot_reward_mo + reward * np.power(self.agent.gamma, cnt)
            cnt = cnt + 1

        return tot_reward_mo


    def solve(self, weights):
        self.train_agent(weights)
        V = self.eval_agent()
        self.agent.reset()

        return V
       

       