"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import numpy as np
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer, ReplayBufferNew, ReplayBufferStructure, ReplayBufferStructureLean
from result_buffer import ResultBuffer
import time
import csv
import os
import logging
import tensorflow as tf
from analysis import Qgraph

logger = logging.getLogger("logger")


class DDPG(object):
    def __init__(self, env, actor, critic, config):

        self.env = env
        self.actor = actor
        self.critic = critic
        self.sess = actor.sess
        self.config = config

        # initiate replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)

        # initiate results buffer
        self.results = ResultBuffer(config.directory, ["policy_eval", "train", "eval", "long_eval"])

        # training parameters
        self.train_episode_len = config.episode_len
        self.episode_num = config.episode_num
        self.episode_num_pol = config.episode_num_pol
        self.env_size = config.env_size
        self.offline_batch_size = config.batch_size
        self.explore_noise = config.noise_std
        self.explore_noise_dec = config.noise_dec
        self.reward_decay = config.gamma
        self.reward_averaging = config.tau

        # evaluation parameters
        self.eval_len = config.eval_len
        self.env_eval_size = config.env_eval_size
        self.last_eval_len = config.last_eval_len

    def run_episode(self, name, pol_eval=None, noise=True):
        self.results.add_episode(name,
                                 self.actor.lr,
                                 self.explore_noise,
                                 self.replay_buffer.size)

        z = self.env.reset(self.env_size)

        for j in range(self.train_episode_len):
            step_results = dict()

            # Added exploration noise
            if noise:
                u = self.noisy_action(z)
            else:
                u = self.actor.predict(z)

            z2, r, w = self.env.step(z, u)
            self.results.average_reward = self.reward_averaging * np.mean(r) + \
                                          (1 - self.reward_averaging) * self.results.average_reward

            step_results.update({'states': z,
                                 'actions': u,
                                 'online_rewards': r,
                                 'disturbance': w})

            for zz, uu, rr, zz2, ww in zip(z, u, r, z2, w):
                self.replay_buffer.add(zz, uu, rr, zz2, ww)

            # Keep adding experience to the memory until there are at least mini-batch size samples
            z_batch, u_batch, r_batch, z2_batch, w_batch = self.replay_buffer.sample_batch(self.offline_batch_size)

            y_target = self.TD_target(z_batch, u_batch, r_batch, z2_batch)


            # Update the critic given the targets
            q_loss, global_step_critic, g_norm_critic = self.critic.train(z_batch, u_batch, y_target)

            step_results.update({'q_loss': q_loss,
                                 'global_step_critic': global_step_critic,
                                 'g_norm_critic': g_norm_critic})

            if pol_eval is None:
                # Update the actor to maximize the critic
                a_outs = self.actor.predict(z_batch)
                grads = self.critic.action_gradients(z_batch, a_outs)
                global_step_actor, g_norm_actor = self.actor.train(z_batch, grads[0])

                step_results.update({'global_step_actor': global_step_actor,
                                     'g_norm_actor': g_norm_actor})

                # Update target networks
                self.actor.update_target_network()
                self.actor.lr_decay()

            self.critic.update_target_network()


            self.critic.lr_decay()


            self.results.update_episode(**step_results)

            z = z2

        self.results.finalize_episode()

    def policy_eval(self, episode_num):
        for i in range(episode_num):
            self.run_episode("policy_eval", pol_eval=True)

            if np.mod(i,5) == 0:
                self.evaluate("eval", self.eval_len)

    def evaluate(self, name, eval_len):
        self.results.add_episode(name,
                                 self.actor.lr,
                                 self.explore_noise,
                                 self.replay_buffer.size)

        z = self.env.reset(self.env_eval_size)

        for j in range(eval_len):
            step_results = dict()

            u = self.actor.predict(z)

            z2, r, w = self.env.step(z, u)
            step_results.update({'states': z,
                                 'actions': u,
                                 'online_rewards': r,
                                 'disturbance': w})

            self.results.update_episode(**step_results)

            z = z2

        self.results.finalize_episode()


    def noisy_action(self, z):
        self.explore_noise *= self.explore_noise_dec
        n = np.random.randn(z.shape[0], self.actor.a_dim) * 0.5
        u_noisy = self.actor.predict_noisy(z, n)

        return u_noisy

    def TD_target(self, z_batch, u_batch, r_batch=None, z2_batch=None, evaluate=None):
        if evaluate is None:
            critic_predict = self.critic.predict_target
        else:
            critic_predict = self.critic.predict

        # Calculate the TD target
        target_q = critic_predict(z2_batch, self.actor.predict(z2_batch))

        # produce TD error with estimated Q(s2, a(s2))
        y_target = r_batch + self.reward_decay * target_q

        return y_target

    def simplex_sampler(self, size):
        unsorted_vec = np.random.rand(size, self.env.input_cardin - 1, self.env.state_cardin)
        sorted_vec = np.sort(unsorted_vec, axis=1)
        full_sorted = np.concatenate([np.zeros([size, 1, self.env.state_cardin]),
                                      sorted_vec,
                                      np.ones([size, 1, self.env.state_cardin])], axis=1)
        diff = full_sorted[:, 1:, :] - full_sorted[:, :-1, :]
        n = diff / np.sum(diff, axis=1, keepdims=True)
        # n_uniform = np.random.rand(size, self.env.input_cardin, self.env.state_cardin)
        # n_log = -np.log(n_uniform)
        # n = n_log / np.sum(n_log, axis=1, keepdims=True)
        return n

    def train(self):
        logger.info(self.results.title())

        logger.info("Initial policy evaluation:")
        logger.info(self.results.title())
        for i in range(self.episode_num_pol):
            self.run_episode("policy_eval", pol_eval=True)

            if np.mod(i, 5) == 0:
                self.evaluate("eval", self.eval_len)

        logger.info("\n\nTraining:")
        logger.info(self.results.title())
        for i in range(self.episode_num):
            self.run_episode("train")
            if np.mod(i, 25) == 0:
                self.evaluate("eval", self.eval_len)

            if np.mod(i, 100) == 0 and i > 0:
                logger.info("############ long evaluation #################")
                self.evaluate("long_eval", self.last_eval_len)

        self.evaluate("long_eval", self.last_eval_len)


class DDPG_Infinite(DDPG):
    def __init__(self, env,actor, critic, config):
        super().__init__(env, actor, critic, config)


    def TD_target(self, z_batch, u_batch, r_batch=None, z2_batch=None, evaluate=None):
        if evaluate is None:
            critic_predict = self.critic.predict_target
        else:
            critic_predict = self.critic.predict

        # Calculate the TD target
        target_q = critic_predict(z2_batch, self.actor.predict(z2_batch))

        # produce TD error with estimated Q(s2, a(s2))
        y_target = r_batch - self.results.average_reward +  self.reward_decay * target_q


        return y_target


class DDPG_Infinite_Planning(DDPG_Infinite):
    def __init__(self, env, actor, critic, config):
        super().__init__(env, actor, critic, config)

    def TD_target(self, z_batch, u_batch, r_batch=None, z2_batch=None, evaluate=None):
        if evaluate is None:
            critic_predict = self.critic.predict_target
        else:
            critic_predict = self.critic.predict


        # Calculate the TD target
        z2, r, p_y = self.env.step(z_batch, u_batch, planning=True)

        y_target = (r - self.results.average_reward)
        P, S = [np.reshape(p_y[:,i], [-1, 1]) for i in range(z2.shape[1])], [z2[:,i,:] for i in range(z2.shape[1])]

        for p, s in zip(P, S):
            y_target += np.reshape(p, [-1, 1]) * critic_predict(s, self.actor.predict_target(s))


        return y_target


# class DDPG_StructuredReplay_Infinite(DDPG_StructuredReplay):
#     def __init__(self, env, actor, critic, config):
#         super().__init__(env, actor, critic, config)
#
#     def TD_target(self, z_batch, u_batch, r_batch=None, z2_batch=None, evaluate=None):
#         if evaluate is None:
#             critic_predict = self.critic.predict_target
#         else:
#             critic_predict = self.critic.predict
#
#         # Calculate the TD target
#         target_q = critic_predict(z2_batch, self.actor.predict(z2_batch))
#
#         # produce TD error with estimated Q(s2, a(s2))
#         y_target = r_batch - self.results.average_reward +  self.reward_decay * target_q
#
#
#         return y_target
