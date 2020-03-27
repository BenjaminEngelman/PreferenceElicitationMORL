import numpy as np
from numpy.random import RandomState
from agents import Agent

# EPISODES = 150000
# EPISODES = 100000
EPISODES = 110000


class Solver(object):
    def __init__(self, modp, agent):
        self.env = modp
        self.agent = agent
    
    def train_agent(self, weights):

        learning_stats = {
            "average_cumulative_reward": [],
            "cumulative_reward": [],
        }

        average_cumulative_reward = 0.0
        # Loop over episodes
        for i in range(EPISODES):
            # if i % 1000 == 0:
            #     print(f"Episode {i}")
            state = self.env.reset()
            terminate = False
            cumulative_reward = 0.0

            # Loop over time-steps
            step = 0
            while not terminate:
                # Getthe action
                action = self.agent.get_action(state)

                # Perform the action
                next_state, reward, terminate, _ = self.env.step(action)
                
                weighted_reward = weights.dot(reward)

                # Update agent
                self.agent.update( state, action, weighted_reward, next_state, terminate )

                # Update statistics
                cumulative_reward += weighted_reward
                state = next_state
                step += 1
            # if i % 1000 == 0:
            #     print(f"Num steps: {step}")
            #     print(f"Cumulative reward: {cumulative_reward}")
            #     print()

            # Per-episode statistics
            average_cumulative_reward *= 0.95
            average_cumulative_reward += 0.05 * cumulative_reward

            ep_stats = [i, cumulative_reward, average_cumulative_reward]

            learning_stats["cumulative_reward"].append(ep_stats[1])
            learning_stats["average_cumulative_reward"].append(ep_stats[2])

        return learning_stats
    
    def eval_agent(self):
        state = self.env.reset()
        # print(self.env._x, self.env._y)

        cnt = 0
        tot_reward_mo = 0
        terminate = False

        while not terminate:
            action = self.agent.get_action(state, greedy=True)
            state, reward, terminate, _ = self.env.step(action)

            if cnt > 1000:
                terminate = True
            tot_reward_mo = tot_reward_mo + reward #* np.power(self.agent.gamma, cnt)
            cnt = cnt + 1


        return tot_reward_mo

    def solve(self, weights):
        self.train_agent(weights)
        V = self.eval_agent()
        self.agent.reset()

        return V
       

if __name__ == "__main__":
    
    from env import BountyfulSeaTreasureEnv, DeepSeaTreasureEnv, OtherDeepSeaTreasure
    from agents import QLearningAgent
    import matplotlib.pyplot as plt
    
    random_state = RandomState(42)
    
    env = BountyfulSeaTreasureEnv()

    n_actions = env.nA
    n_states = env.nS
    print(n_actions, n_states)
    agent = QLearningAgent(n_actions=n_actions, n_states=n_states, decay=0.999997, random_state=random_state)
    solver = Solver(env, agent)

    learning_stats = solver.train_agent(np.array([1, 0]))
    agent.show_qtable()

    solver.eval_agent()

    plt.plot(learning_stats["average_cumulative_reward"])
    plt.show()
    plt.imshow(env.render())
    plt.show()

