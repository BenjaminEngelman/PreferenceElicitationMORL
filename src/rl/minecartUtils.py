import os

import gym
import numpy as np
from gym import spaces
from stable_baselines.common.callbacks import BaseCallback

from src.constants import MINECART_MINES_POS
from src.utils import euclidean_distance


class MinecartObsWrapper(gym.ObservationWrapper):
    def observation(self, s):
        state = np.append(s['position'], [s['speed'], s['orientation'], *s['content']])
        return state


class MultiObjRewardWrapper(gym.RewardWrapper):
    """
    Transform a multi-ojective reward (= array)
    to a single scalar
    """

    def __init__(self, env, weights):
        super().__init__(env)
        self.weights = weights
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

    def reward(self, rew):
        return self.weights.dot(rew)


class MinecartMultiObjRewardWrapper(MultiObjRewardWrapper):

    def __init__(self, env, weights, penalty_fac):
        super().__init__(env, weights)
        self.penalty_fac = penalty_fac

    def reward(self, rew):
        cart_pos = self.cart.pos
        dist2mines = [euclidean_distance(cart_pos, mine_pos) for mine_pos in MINECART_MINES_POS]
        mean_dist = np.mean(dist2mines)
        # min_dist = np.min(dist2mines)

        if np.sum(self.cart.content) != self.capacity:
            reward = self.weights.dot(rew) - (self.penalty_fac * mean_dist)
        else:
            reward = self.weights.dot(rew)

        return reward


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model once after num_steps steps

    :param num_steps: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """

    def __init__(self, save_path: str, name_prefix='rl_model', num_steps=15_000_000, verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.num_steps = num_steps
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls == self.num_steps:
            path = os.path.join(self.save_path, '{}_checkpoint'.format(self.name_prefix))
            self.model.save(path)
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True
