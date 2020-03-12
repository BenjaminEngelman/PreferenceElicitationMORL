from gym.envs.toy_text import discrete
import gym
import numpy as np
import cv2


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class DeepSeaTreasureEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=10):

        self.shape = (width+1, width)
        self.start_state_index = 0

        nS = np.prod(self.shape)
        nA = 4

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(DeepSeaTreasureEnv, self).__init__(nS, nA, P, isd)

    def _treasures(self):

        if self.shape[1] > 10:
            raise ValueError('Default Deep Sea Treasure only supports a grid-size of max 10')

        return {(1, 0): 1,
                (2, 1): 2,
                (3, 2): 3,
                (4, 3): 5,
                (4, 4): 8,
                (4, 5): 16,
                (7, 6): 24,
                (7, 7): 50,
                (9, 8): 74,
                (10, 9): 124}

    def _unreachable_positions(self):
        u = []
        treasures = self._treasures()
        for p in treasures.keys():
            for i in range(p[0]+1, self.shape[0]):
                u.append((i, p[1]))
        return u

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):

        unreachable = self._unreachable_positions()
        treasures = self._treasures()
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_position = tuple(new_position)
        if new_position in unreachable:
            new_position = tuple(current)
        new_state = np.ravel_multi_index(new_position, self.shape)

        if new_position in treasures:
            reward = [treasures[new_position], -1]
            done = True
        else:
            reward = [0, -1]
            done = False
        return [(1., new_state, np.array(reward), done)]

    def render(self, mode='rgb_array'):
        tile_size = 30
        img = np.full((self.shape[0]*tile_size, self.shape[1]*tile_size, 3), 255, np.uint8)

        y = np.tile(np.arange(tile_size, (self.shape[0]+1)*tile_size, tile_size), self.shape[1])
        x = np.repeat(np.arange(tile_size, (self.shape[1]+1)*tile_size, tile_size), self.shape[0])
        for x_i in x:
            for y_i in y:
                cv2.circle(img, (x_i, y_i), 0, (255, 0, 0))

        for c, t in self._treasures().items():
            cv2.putText(img, str(t), (tile_size*c[1]+tile_size//2, tile_size*c[0]+tile_size//2), cv2.FONT_HERSHEY_SIMPLEX, .2, 255)
        position = np.unravel_index(self.s, self.shape)
        cv2.putText(img, 'sub', (tile_size*position[1]+tile_size//2, tile_size*position[0]+tile_size//2), cv2.FONT_HERSHEY_SIMPLEX, .2, 255)

        return img


class BountyfulSeaTreasureEnv(DeepSeaTreasureEnv):

    def __init__(self, width=10):
        # random treasure-depths for each x-pos
        depths = np.random.choice(range(4), size=width-1, p=[.3, .5, .1, .1])
        depths = [0, 0, 0, 1, 1, 1, 0, 1, 0]
        # depths = np.array([0, 0, 1, 1, 2, 0, 0, 1, 1])
        # add first treasure depth (always 1)
        depths = np.append([1], depths)
        depths = np.cumsum(depths)
        # limit to grid
        depths[depths > width] = width
        self.depths = depths
        super(BountyfulSeaTreasureEnv, self).__init__(width=width)

    def _treasures(self):

        pareto_front = lambda x: np.round(-45.64496 - (59.99308/-0.2756738)*(1 - np.exp(0.2756738*x)))
        treasures = {(d, i): pareto_front(-(i+d)) for i, d in enumerate(self.depths)}
        return treasures

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_reward = 166
    
    def reward(self, rew):
        return rew / self.max_reward


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = RewardWrapper(BountyfulSeaTreasureEnv())
    # print(len(env._treasures()))
    # plt.imshow(env.render(mode="human"))
    plt.show()