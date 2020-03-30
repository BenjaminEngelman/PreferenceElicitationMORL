import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import argmax
from stable_baselines import DQN
from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq.build_graph import build_train
from stable_baselines.deepq.policies import DQNPolicy, MlpPolicy, CnnPolicy
from constants import BST_DIRECTIONS


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

class MOQlearning(Agent):
    
    def __init__(self, env, decay=0.999999, random_state=1):
        super().__init__(env, decay=decay, random_state=random_state)
        self.n_states = env.nS
        self.n_actions = env.nA

        # self.qtable = [np.array([0.]*self.n_actions) for state in range(n_states)]
        self.qtable = [np.random.uniform(-1, 175, self.n_actions) for state in range(self.n_states)]
    
    def learn(self, n_episodes, weights):
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
                # Getthe action
                action = self.get_action(state)

                # Perform the action
                next_state, reward, terminate, _ = self.env.step(action)
                
                weighted_reward = weights.dot(reward)

                # Update agent
                self.update( state, action, weighted_reward, next_state, terminate )

                # Update statistics
                cumulative_reward += weighted_reward
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
        self.qtable = [np.array([0.]*self.n_actions) for state in range(self.n_states)]
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
        table = np.chararray((11, 10))
        for i in range(self.n_states):
            if (max(self.qtable[i])) != 0:
                table[i // 10, i % 10] = BST_DIRECTIONS[argmax(self.qtable[i])]
            else:
                table[i // 10, i % 10] = "N"
        print(table)
        print(self.epsilon)

class MODQN(DQN):
    """
    Adaptation of the DQN agent from stable-baselines
    for Multiple ojectives environement
    It uses boths Double DQN and Dueling DQN
    """
    def __init__(self, env):
        super(MODQN, self).__init__(
            CnnPolicy,
            env,
            tensorboard_log="./tensorboard/",
            batch_size=64,
            gamma=0.98,
            verbose=1,
            buffer_size=100000,
            exploration_final_eps=0.01,
            prioritized_replay_eps=0.01,
            prioritized_replay_alpha=2.0
        )
    
    def demonstrate(self):
        state = self.env.reset()
        terminate = False
        step = 0
        while not terminate or step < 500:
            plt.imshow(self.env.render())
            plt.show()
            action, _ = self.predict(state)
            state, _, terminate, _ = self.env.step(action)
            step += 1
        self.env.render()
        
    
    def learn(self, total_timesteps, obj_weights, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                initial_p=self.exploration_initial_eps,
                                                final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_successes = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            reset = True
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            for timestep in range(total_timesteps):

                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)

                # IMPORTANT FOR MULTI-OBJECTCTIVES ENVIRONEMENT
                rew = obj_weights.dot(rew)

                self.num_timesteps += 1

                # Stop training if return value is False
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, rew
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs_, action, reward_, new_obs_, float(done))
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                if writer is not None:
                    ep_rew = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                        self.num_timesteps)

                episode_rewards[-1] += reward_
                if done:
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:

                    callback.on_rollout_end()
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    # pytype:disable=bad-unpacking
                    if self.prioritized_replay:
                        assert self.beta_schedule is not None, \
                                "BUG: should be LinearSchedule when self.prioritized_replay True"
                        experience = self.replay_buffer.sample(self.batch_size,
                                                                beta=self.beta_schedule.value(self.num_timesteps),
                                                                env=self._vec_normalize_env)
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size,
                                                                                                env=self._vec_normalize_env)
                        weights, batch_idxes = np.ones_like(rewards), None
                    # pytype:enable=bad-unpacking

                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + self.num_timesteps) % 100 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                    dones, weights, sess=self.sess, options=run_options,
                                                                    run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        else:
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                    dones, weights, sess=self.sess)
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                    callback.on_rollout_start()

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring",
                                            int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()

        callback.on_training_end()
        return self

if __name__ == "__main__":
    from minecart.envs.minecart_env import MinecartDeterministicEnv

    env = MinecartDeterministicEnv()
    weights = np.array([0.2, 0.6, 0.2])
    s = env.reset()
    agent = MODQN(env)
    agent.learn(10000, weights)


    