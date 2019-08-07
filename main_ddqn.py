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
from channel_envs import Trapdoor, Trapdoor3, Ising, Ising3, Bec_nc1, Bec_121, Bec_Dicode
from replay_buffer import ReplayBuffer
from utils import preprocess, save_models
from model_ddqn import ActorNetwork, CriticNetwork, OrnsteinUhlenbeckActionNoise

logger = logging.getLogger("logger")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.FATAL)


def train(sess, env, env_eval, actor, critic, actor_eval, critic_eval, saver):
    def train_ep(p_ev=None, ep_len=None):
        t = time.time()
        ep_ave_r = run_episode(env, actor, critic, replay_buffer, rho_tar_list[-1],
                                    pol_eval=p_ev, episode_len=ep_len)
        ep_time = time.time() - t
        return ep_ave_r, ep_time

    def test_ep(name, plot, save):
        eval_len = config.eval_episode_len
        t = time.time()
        r, r_tar = test_actor(env_eval, actor_eval, critic_eval, rho_list, rho_tar_list,
                                eval_len=eval_len, name=name, plot=plot, save=save)
        rho_list.append(r), rho_tar_list.append(r_tar)
        ev_time = time.time() - t

        return ev_time

    def table_title():
        logger.info('{:^5} | {:^9} | {:^9} | {:^5} | {:^5} | {:^11} | {:^9} | {:^9} |'
                    .format('Epi.', 'lr', 'Noise std', 'Epi. secs', 'Eval secs', 'Epi. Avg Ro', 'Rho', 'Rho Target'))

    def table_entry(idx):
        logger.info('{:^5d} | {:^9.4e} | {:^9.4f} | {:^9d} | {:^9d} | {:^11.05f} | {:^9.05f} | {:^10.05f} |'
                    .format(idx, actor.lr(), config.noise_std, int(ep_elapsed),
                            int(ev_elapsed), ep_ave_reward, rho_list[-1], rho_tar_list[-1]))

    # initiate replay buffer
    replay_buffer = ReplayBuffer(config.buffer_size, config.seed)

    # initiate arrays for documenting average cumulative reward of online and target networks.
    rho_list, rho_tar_list = list(), list()

    logger.info("evaluating initial average reward")
    ev_elapsed = test_ep(name='pol_eval', plot=None, save=None)

    logger.info("performing initial episode of policy evaluation")
    ep_ave_reward, ep_elapsed = train_ep(p_ev=True, ep_len=1000)

    table_title()
    table_entry(-1)

    for i in range(config.episode_num):
        logger.debug("global step actor: {}".format(actor.global_step()))
        logger.debug("global step critic: {}".format(critic.global_step()))

        plot = None
        save = True if i % 50 == 0 else None

        ep_ave_reward, ep_elapsed = train_ep()
        ev_elapsed = test_ep(name=i, plot=plot, save=save)
        table_entry(i)

        if save:
            save_models(saver, sess, config.directory)

        if i % 25 == 0:
            critic.update_target_network()
            actor.update_target_network()
        # if i % 100 == 0:
        #     test_actor(env_eval, actor_eval, critic_eval, rho_list, rho_tar_list,
        #                eval_len=10000, name="final", plot=True, verbose=True)

    return rho_list, rho_tar_list


def test_actor(env, actor, critic, ro_train, ro_train_tar, eval_len=None, name=None, plot=None, verbose=None, save=None):
    def save2mat_file():
        mat4py.savemat(os.path.join(config.directory, "states", 'states_evol.mat'), {'online_s': s_evol, 'target_s': s_evol_tar})
        mat4py.savemat(os.path.join(config.directory, "states", 'actions_evol.mat'), {'online_a': a_evol, 'target_a': a_evol_tar})
        mat4py.savemat(os.path.join(config.directory, "states", 'outputs_evol.mat'), {'online_y': y_evol, 'target_y': y_evol_tar})


    eval_len = config.test_episode_len if eval_len is None else eval_len

    if verbose is not None:
        logger.info("evaluating policy for {} steps...".format(eval_len))

    ro, s_evol, a_evol, y_evol = evaluate(env, actor.predict, eval_len)
    ro_tar, s_evol_tar, a_evol_tar, y_evol_tar = evaluate(env, actor.predict_target, eval_len)
    actor.ro_summary(ro, ro_tar)

    if verbose is not None:
        logger.info('Ro: {:.07f} Ro target: {:.07f}'.format(ro, ro_tar))

    if plot is not None:
        env.plot(actor, critic, config.plot_bins, ro_train, ro_train_tar,
                 np.reshape(s_evol, [-1, env.state_cardin-1]), ro, np.reshape(s_evol_tar, [-1, env.state_cardin-1]),
                 ro_tar, config.directory, name)

    if save:
        save2mat_file()

    return ro, ro_tar


def evaluate(env, predict, T):
    s_list, a_list, w_list = list(), list(), list()
    s = env.reset()
    ep_reward = 0

    for j in range(T):
        a = predict(np.reshape(s, (env.size, env.state_cardin-1)))
        a = np.minimum(np.maximum(a, 0.0), 1.0)
        s2, r = env.step(a)

        if j >= config.eval_transient:
            s_list.append(s)
            a_list.append(a)
            w_list.append(env.w)
            ep_reward += np.mean(r)

        s = env.z

    ep_ave_reward = ep_reward / (T-config.eval_transient)

    s_evol = np.swapaxes(np.reshape(np.array(s_list), [T-config.eval_transient, -1, env.state_cardin-1]), 0, 1)
    a_evol = np.swapaxes(np.reshape(np.array(a_list), [T-config.eval_transient, -1, env.state_cardin * env.input_cardin]), 0, 1)
    y_evol = np.swapaxes(np.reshape(np.array(w_list), [T-config.eval_transient, -1, 1]), 0, 1)

    return ep_ave_reward, s_evol, a_evol, y_evol


def run_episode(env, actor, critic, replay_buffer, rho_avg, pol_eval=None, episode_len=None):
    if episode_len is None:
        episode_len = config.episode_len
    transient = 20
    batch_size = config.batch_size

    target = 0
    ep_reward = 0
    avg_actor_grad_norm = 0
    avg_critic_grad_norm = 0
    s = env.reset()

    for j in range(episode_len):

        if np.random.rand() <= 1.0:
            predict_target = critic.predict_target
            actor_predict_target = actor.predict_target
            actor_predict = actor.predict
            actor_train_f = actor.train
            train_f = critic.train
            action_gradient = critic.action_gradients
        else:
            predict_target = critic.predict
            actor_predict_target = actor.predict
            actor_predict = actor.predict_target
            actor_train_f = actor.train_target
            train_f = critic.train_target
            action_gradient = critic.action_gradients_target

        # Added exploration noise
        a = actor_predict(np.reshape(s, (env.size, env.state_cardin-1)))

        n = np.random.exponential(scale=1.0, size=[config.env_size, env.input_cardin, env.state_cardin])
        n /= np.tile(np.sum(n, axis=1, keepdims=True), [1, env.input_cardin, 1])

        a = np.reshape(a, [-1, env.input_cardin, env.state_cardin])
        a = a*(1-config.noise_std)+ n * config.noise_std
        a = np.reshape(a, [-1, actor.a_dim])

        a = np.maximum(np.minimum(a, 1), 0)

        config.noise_std *= config.noise_dec

        s2, r = env.step(a)
        p_y = env.p_y

        for k, (ss, aa, rr, ss2, pp_y) in enumerate(zip(s, a, r, s2, p_y)):
            replay_buffer.add(np.reshape(ss, (env.state_cardin-1,)), np.reshape(aa, (actor.a_dim,)), np.reshape(rr, (1,)),
                              np.reshape(ss2, (env.state_cardin-1, env.state_cardin)), np.reshape(pp_y, (env.output_cardin,)))

        # Keep adding experience to the memory until there are at least minibatch size samples
        if replay_buffer.size() > batch_size:
            s_batch, a_batch, r_batch, s2_batch, p_y_batch = \
                replay_buffer.sample_batch(config.batch_size)
        else:
            logger.debug("Replay Buffer is filling up...")
            continue



        # Calculate the TD target
        if config.planning:
            target_q = 0
            for i in range(env.output_cardin):
                target_q += np.reshape(p_y_batch[:, i], [-1, 1])*predict_target(
                    s2_batch[:, :, i], actor_predict(s2_batch[:, :, i]))
        else:
            s2_batch = env.sample_next_states(p_y_batch, s2_batch)

            target_q = predict_target(s2_batch, actor_predict(s2_batch))

        target += np.mean(target_q)

        # produce TD error with estimated Q(s2, a(s2))
        y_i = r_batch - rho_avg + critic.gamma * target_q

        # Update the critic given the targets
        critic_grad_norm  = train_f(s_batch, a_batch, np.reshape(y_i, (batch_size, 1)))
        # avg_critic_grad_norm += critic_grad_norm

        if pol_eval is None:
            a_outs = actor_predict(s_batch)
            grads = action_gradient(s_batch, a_outs)
            actor_grad_norm = actor_train_f(s_batch, grads[0])
            # avg_actor_grad_norm += actor_grad_norm
            # Update target networks
            # actor.update_target_network()

        critic.lr_decay()
        actor.lr_decay()

        if j >= transient:
            ep_reward += np.mean(r)

        s = env.z

    print(target/episode_len)
    ep_ave_reward = ep_reward / (episode_len - transient)
    # avg_actor_grad_norm /= episode_len
    # avg_critic_grad_norm /= episode_len

    return ep_ave_reward


def build_env(env_name):
    if env_name == "trapdoor":
        env = Trapdoor(config.env_size)
        env_eval = Trapdoor(config.env_eval_size)
    elif env_name == "trapdoor3":
        env = Trapdoor3(config.env_size)
        env_eval = Trapdoor3(config.env_eval_size)
    elif env_name == "ising":
        env = Ising(config.env_size)
        env_eval = Ising(config.env_eval_size)
    elif env_name == "ising3":
        env = Ising3(config.env_size, config.env_cardin)
        env_eval = Ising3(config.env_eval_size, config.env_cardin)
    elif env_name == "bec_nc1":
        env = Bec_nc1(config.env_size, 0.5)
        env_eval = Bec_nc1(config.env_eval_size, 0.5)
    elif env_name == "bec_121":
        env = Bec_121(config.env_size, 0.5)
        env_eval = Bec_121(config.env_eval_size, 0.5)
    elif env_name == "bec_dicode":
        env = Bec_Dicode(config.env_size, 0.5)
        env_eval = Bec_Dicode(config.env_eval_size, 0.5)
    else:
        raise ValueError("Invalid environment name")

    return env, env_eval


def build_actor_critic(sess, env, env_eval):
    w_init = tflearn.initializations.xavier_initializer()

    with tf.variable_scope("model", reuse=None, initializer=w_init):
        with tf.name_scope("actor"):
            actor = ActorNetwork(sess, env, config, is_training=True)

        with tf.name_scope("critic"):
            critic = CriticNetwork(sess, env, config, is_training=True)

        # if config.noise_std:
        #     actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.input_cardin * env.state_cardin),
        #                                                sigma=config.noise_std,
        #                                                sigma_dec=config.noise_dec)
        # else:
        #     actor_noise = None

    with tf.variable_scope("model", reuse=True):
        with tf.name_scope("actor"):
            actor_eval = ActorNetwork(sess, env_eval, config, is_training=False)

        with tf.name_scope("critic"):
            critic_eval = CriticNetwork(sess, env_eval, config, is_training=False)

    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    return actor, critic, actor_eval, critic_eval #, actor_noise


def main():

    with tf.Session(config=sess_config) as sess:

        np.random.seed(config.seed)
        tf.set_random_seed(config.seed)

        # build environment and actor/critic
        env, env_eval = build_env(config.env)
        actor, critic, actor_eval, critic_eval = build_actor_critic(sess, env, env_eval)
        saver = tf.train.Saver(tf.global_variables())

        # train actor/critic
        rho_list, rho_tar_list = train(sess, env, env_eval, actor, critic, actor_eval, critic_eval, saver)

        # evaluate final policy
        test_actor(env_eval, actor_eval, critic_eval, rho_list, rho_tar_list,
                   eval_len=config.test_episode_len, name="final", plot=True, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    ######## randomness #########
    parser.add_argument('--seed',           type=int, default=None,         help='random seed for repeatability')

    ######## environment ########
    parser.add_argument('--config',         type=str, default='base',       help='choose the from {base,latest}')
    parser.add_argument('--env',            type=str, default='trapdoor',   help='choose the from {trapdoor,ising,bec_nc1}')
    parser.add_argument('--env_cardin',     type=int, default=None,         help='choose the from {trapdoor,ising,bec_nc1}')
    parser.add_argument('--env_size',       type=int, default=None,         help='environment size for training examples')
    parser.add_argument('--env_eval_size',  type=int, default=None,         help='environment size for evaluation')
    parser.add_argument('--eval_tr',        type=int, default=None,         help='evaluation transient - #of examples to '
                                                                                 'dispose from cumulative reward calculation')

    ######## RL #########
    parser.add_argument('--planning',       dest='planning',                action='store_true')
    parser.set_defaults(planning=False)
    parser.add_argument('--gamma',          type=float, default=None,       help='discount factor of future rewards')
    parser.add_argument('--tau',            type=float, default=None,       help='moving average parameter for target network')
    parser.add_argument('--buffer_size',    type=int, default=None,         help='max size of the replay buffer')
    parser.add_argument('--batch_size',     type=int, default=None,         help='size of mini-batch for offline learning with SGD')
    parser.add_argument('--episode_num',    type=int, default=None,         help='#of episodes in training')
    parser.add_argument('--episode_len',    type=int, default=None,         help='length of episode')
    parser.add_argument('--eval_len',       type=int, default=None,         help='length of eval (during train) episode')
    parser.add_argument('--test_len',       type=int, default=None,         help='length of test episode')
    parser.add_argument('--noise_std',      type=float, default=None,       help='actor noise std')
    parser.add_argument('--noise_dec',      type=float, default=None,       help='actor noise decay')

    ######## optimizer #########
    parser.add_argument('--opt',            type=str, default=None,         help='optimizer from {adam}')
    parser.add_argument('--lr_decay',       type=float, default=None,       help='learning rate decay of actor/critic')
    parser.add_argument('--clip',           type=float, default=None,       help='update clipping norm')

    ######## actor #########
    parser.add_argument('--actor_lr',       type=float, default=None,       help='actor network learning rate')
    parser.add_argument('--actor_drop',     type=float, default=None,       help='actor network dropout rate')
    parser.add_argument('--actor_hid',      type=int, default=None,         help='actor hidden size')
    parser.add_argument('--actor_layers',   type=int, default=None,         help='actor #of layers')

    ######## critic #########
    parser.add_argument('--critic_lr',      type=float, default=None,       help='critic network learning rate')
    parser.add_argument('--critic_drop',    type=float, default=None,       help='critic network dropout rate')
    parser.add_argument('--critic_hid',     type=int, default=None,         help='critic hidden size')
    parser.add_argument('--critic_layers',  type=int, default=None,         help='critic #of layers')

    ######## summary #########
    parser.add_argument('--plot_bins',      type=int, default=None,         help='size of plot s vec')
    parser.add_argument('--plot_rate',      type=int, default=None,         help='generate plot figure every plot_rate episodes')
    parser.add_argument('--name',           type=str, default=None,         help='simulation name')

    parser.add_argument('--verbose',        dest='verbose',                 action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--debug',          dest='debug',                   action='store_true')
    parser.set_defaults(debug=None)

    args = parser.parse_args()

    config, sess_config, logger = preprocess(args)

    main()