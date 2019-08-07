import tensorflow as tf
import tflearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import logging
import numpy as np
import time
import os
import h5py
import scipy.io as mat4py
from channel_envs import Trapdoor, Trapdoor3, Ising, Ising3, Bec_nc1, Bec_121, Bec_Dicode
from utils import preprocess, load_models
from model import ActorNetwork, CriticNetwork, OrnsteinUhlenbeckActionNoise

logger = logging.getLogger("logger")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)


def evaluate(env, predict, T):
    s_list, w_list, a_list = list(), list(), list()

    s = env.reset()
    ep_reward = 0

    for j in range(T):
        a = predict(np.reshape(s, (env.size, env.state_cardin-1)))
        # a = np.minimum(np.maximum(a, 0.0), 1.)
        s2, r = env.step(a)

        if j >= config.eval_transient:
            s_list.append(s)
            a_list.append(a)
            w_list.append(env.w)
            ep_reward += np.mean(r)

        s = s2

    ep_ave_reward = ep_reward / (T-config.eval_transient)

    return ep_ave_reward, np.array(s_list), np.array(a_list), np.array(w_list)


def test_actor(env, actor, eval_len=None, verbose=None):

    def plot(x,y, name):
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        plt.subplot(1, 2, 1)

        H, xedges, yedges = np.histogram2d(x, y, bins=(100, 100))
        H_normalized = H / float(x.shape[0])

        # plt.hist2d(x, y, bins=(100, 100), normed=True, cmap=mpl.cm.cool)
        plt.imshow(H_normalized, cmap=mpl.cm.cool,)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.scatter(x, y)

        plt.savefig(os.path.join(config.directory, name))
        plt.close()

    def transition_matrix(evol, name):
        N = 100 # quantization rate
        P = np.zeros([N**2, N**2])
        evol = np.digitize(evol, np.linspace(0, 1, N))
        c_prime, d_prime = evol[0]
        for c, d in evol[1:]:
            P[c * N + d, c_prime * N + d_prime] += 1
            c_prime, d_prime = c, d

        P /= np.tile(np.sum(P, axis=0), [N**2, 1]) + 1e-10


        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)

        plt.imshow(P, cmap=mpl.cm.cool)
        plt.colorbar()

        plt.savefig(os.path.join(config.directory, "transition matrix"))
        plt.close()

        save("transition_matrix_{}".format(name), P)

    def save(name, var):
        # h5f = h5py.File(os.path.join(config.directory, '{}.mat'.format(name)), 'w')
        # h5f.create_dataset(name, data=var)
        # h5f.close()
        mat4py.savemat(os.path.join(config.directory, '{}.mat'.format(name)), {name: var})

    eval_len = config.test_episode_len if eval_len is None else eval_len

    if verbose is not None:
        logger.info("evaluating policy for {} steps...".format(eval_len))

    logger.info("Online Network:")

    t = time.time()
    ro, s_vec, a_vec, w_vec = evaluate(env, actor.predict, eval_len)
    save("online_states", s_vec), save("online_a", a_vec), save("online_w", w_vec)
    logger.info("evaluation elapsed time: {:2.2f}".format(time.time() - t))
    #
    # t = time.time()∑
    # plot(s_vec[:, 0], s_vec[:, 1], "online")
    # logger.info("plotting elapsed time: {:2.2f}".format(time.time() - t))
    #
    # t = time.time()
    # save("online_states", s_vec)
    # logger.info("saving states evolution elapsed time: {:2.2f}".format(time.time() - t))
    #
    # t = time.time()
    # transition_matrix(s_vec, "online")
    # logger.info("transition matrix elapsed time: {:2.2f}".format(time.time() - t))

    logger.info("Target Network:")

    t = time.time()
    ro_tar, s_vec_tar, a_vec_tar, w_vec_tar = evaluate(env, actor.predict_target, eval_len)
    save("target_states", s_vec_tar), save("target_a", a_vec_tar), save("target_w", w_vec_tar)
    logger.info("evaluation elapsed time: {:2.2f}".format(time.time() - t))

    # t = time.time()
    # plot(s_vec_tar[:, 0], s_vec_tar[:, 1], "target")
    # logger.info("plotting elapsed time: {:2.2f}".format(time.time() - t))
    #
    # t = time.time()
    # save("target_states", s_vec_tar)
    # logger.info("saving states evolution elapsed time: {:2.2f}".format(time.time() - t))
    #
    # t = time.time()
    # transition_matrix(s_vec, "target")
    # logger.info("transition matrix elapsed time: {:2.2f}".format(time.time() - t))

    if verbose is not None:
        logger.info('Rho: {:.07f} Rho target: {:.07f}'.format(ro, ro_tar))

    return ro, ro_tar


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
        env = Ising3(config.env_size)
        env_eval = Ising3(config.env_eval_size)
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


def build_actor_critic(sess, env):
    with tf.variable_scope("model", reuse=None):
        with tf.name_scope("actor"):
            actor = ActorNetwork(sess, env, config, is_training=False)

        with tf.name_scope("critic"):
            critic = CriticNetwork(sess, env, config, is_training=False)

    sess.run(tf.global_variables_initializer())

    return actor, critic


def main():

    with tf.Session(config=sess_config) as sess:

        # build environment
        _, env = build_env(config.env)

        # load actor & critic
        actor, critic = build_actor_critic(sess, env)
        load_models(sess, config.model_path)

        # evaluate final policy
        test_actor(env, actor, eval_len=config.test_episode_len // 100, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    ######## randomness #########
    parser.add_argument('--seed',           type=int, default=None,         help='random seed for repeatability')

    ######## saved model path  #########
    parser.add_argument('--model_path',     type=str, default=None,         help='path for saved model')

    ######## environment ########
    parser.add_argument('--config',         type=str, default='base',       help='choose the from {base,latest}')
    parser.add_argument('--env',            type=str, default='trapdoor',   help='choose the from {trapdoor,ising,bec_nc1}')
    parser.add_argument('--env_size',       type=int, default=None,         help='environment size for training examples')
    parser.add_argument('--env_eval_size',  type=int, default=None,         help='environment size for evaluation')
    parser.add_argument('--eval_tr',        type=int, default=None,         help='evaluation transient - #of examples to '
                                                                                 'dispose from cumulative reward calculation')

    ######## RL #########
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
    # parser.add_argument('--nonmono',        type=int, default=None,         help='non monotonic trigger to re-average target network')
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