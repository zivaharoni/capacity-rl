import tensorflow as tf
import tflearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import logging
import numpy as np
import time
import os
import scipy.io as mat4py
import channel_envs
from utils import preprocess, save_models
from model import ActorNetwork, CriticNetwork
logger = logging.getLogger("logger")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.FATAL)



def build_env(env_name, sess):
    if env_name == "ising":
        env = channel_envs.Ising(config.env_cardin, sess)
        logger.info("Ising Channel alphabet {}\tachievable rate: {}".format(env.cardinality, env.capacity()))
    elif env_name == "trapdoor":
        env = channel_envs.Trapdoor(config.env_cardin, sess)
        logger.info("Trapdoor Channel alphabet {}\tachievable rate: {}".format(env.cardinality, env.capacity()))
    elif env_name == "bec_nc1":
        env = channel_envs.RLL_0_K(config.env_cardin, sess)
        logger.info("BEC nc1 Channel alphabet {}\tachievable rate: {}".format(env.cardinality, env.capacity()))
    else:
        raise ValueError("Invalid environment name")

    return env


def build_actor_critic(sess, env):
    w_init = tflearn.initializations.xavier_initializer()

    with tf.variable_scope("model", reuse=None, initializer=w_init):
        with tf.name_scope("actor"):
            actor = ActorNetwork(sess, env, config, is_training=True)

        with tf.name_scope("critic"):
            critic = CriticNetwork(sess, env, config, is_training=True)


    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    return actor, critic


def main():

    with tf.Session(config=sess_config) as sess:

        np.random.seed(config.seed)
        tf.set_random_seed(config.seed)

        # build environment and actor/critic
        env = build_env(config.env, sess)
        actor, critic = build_actor_critic(sess, env)

        # train actor/critic
        Alg = config.alg(env, actor, critic, config)
        Alg.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    ######## randomness #########
    parser.add_argument('--seed',           type=int, default=None,         help='random seed for repeatability')

    ######## environment ########
    parser.add_argument('--config',         type=str, default='ddpg_planning',       help='choose the from {base,latest}')
    parser.add_argument('--env',            type=str, default='ising',      help='choose the from {trapdoor,ising,bec_nc1}')
    parser.add_argument('--env_cardin',     type=int, default=None,         help='choose the from {trapdoor,ising,bec_nc1}')
    parser.add_argument('--env_size',       type=int, default=None,         help='environment size for training examples')
    parser.add_argument('--env_eval_size',  type=int, default=None,         help='environment size for evaluation')

    ######## RL #########
    parser.add_argument('--planning_unroll',type=int, default=None,         help='discount factor of future rewards')
    parser.add_argument('--gamma',          type=float, default=None,       help='discount factor of future rewards')
    parser.add_argument('--tau',            type=float, default=None,       help='moving average parameter for target network')
    parser.add_argument('--buffer_size',    type=int, default=None,         help='max size of the replay buffer')
    parser.add_argument('--batch_size',     type=int, default=None,         help='size of mini-batch for offline learning with SGD')
    parser.add_argument('--episode_num',    type=int, default=None,         help='#of episodes in training')
    parser.add_argument('--episode_num_pol',type=int, default=None,         help='#of episodes in training')
    parser.add_argument('--episode_len',    type=int, default=None,         help='length of episode')
    parser.add_argument('--eval_len',       type=int, default=None,         help='length of eval (during train) episode')
    parser.add_argument('--test_len',       type=int, default=None,         help='length of test episode')
    parser.add_argument('--noise_std',      type=float, default=None,       help='actor noise std')
    parser.add_argument('--noise_dec',      type=float, default=None,       help='actor noise decay')

    ######## optimizer #########
    parser.add_argument('--opt',            type=str, default=None,         help='optimizer from {adam}')
    parser.add_argument('--lr_decay',       type=float, default=None,       help='learning rate decay of actor/critic')

    ######## actor #########
    parser.add_argument('--actor_lr',       type=float, default=None,       help='actor network learning rate')
    parser.add_argument('--actor_hid',      type=int, default=None,         help='actor hidden size')
    parser.add_argument('--actor_layers',   type=int, default=None,         help='actor #of layers')

    ######## critic #########
    parser.add_argument('--critic_lr',      type=float, default=None,       help='critic network learning rate')
    parser.add_argument('--critic_hid',     type=int, default=None,         help='critic hidden size')
    parser.add_argument('--critic_layers',  type=int, default=None,         help='critic #of layers')

    ######## summary #########
    parser.add_argument('--name',           type=str, default=None,         help='simulation name')

    parser.add_argument('--verbose',        dest='verbose',                 action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--debug',          dest='debug',                   action='store_true')
    parser.set_defaults(debug=None)

    args = parser.parse_args()

    config, sess_config, logger = preprocess(args)

    main()