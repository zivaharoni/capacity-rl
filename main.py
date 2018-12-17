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
from channel_envs import Trapdoor, Ising, Bec_nc1
from replay_buffer import ReplayBuffer
from utils import preprocess
from model import ActorNetwork, CriticNetwork, OrnsteinUhlenbeckActionNoise

logger = logging.getLogger("logger")

# ===========================
#   Actor and Critic DNNs
# ===========================

def plot_actor_critic(env, actor, critic, N, ro_train, ro_train_tar,
                      z, ro, z_tar, ro_tar, name=None, save=None):
    def save2mat_file(s_vec, a_vec, h_vec, z):
        mat4py.savemat(os.path.join(config.directory, 'state.mat'), {'state': s_vec})
        mat4py.savemat(os.path.join(config.directory, 'action.mat'), {'action': a_vec})
        mat4py.savemat(os.path.join(config.directory, 'state-value.mat'), {'value': h_vec})
        mat4py.savemat(os.path.join(config.directory, 'state-hist.mat'), {'state_hist': z})

    # grid the state space
    s_vec = np.reshape(np.linspace(0, 1, N), (N, actor.s_dim))

    # compute policy and state-value function
    a_vec = actor.predict(s_vec)
    h_vec = critic.predict(s_vec, a_vec)

    # compute policy and state-value function of target network
    a_vec_tar = actor.predict_target(s_vec)
    h_vec_tar = critic.predict_target(s_vec, a_vec_tar)

    if save is not None:
        save2mat_file(s_vec, a_vec, h_vec, z)

    # calculate gamma and delta of networks and their ground-truth
    if actor.a_dim == 2:
        # graphs of the function approximation
        a_vec[:,1] = 1 - a_vec[:,1]
        a_vec_plot = a_vec * np.concatenate([s_vec, 1-s_vec], axis=1)

        # graphs of the target networks
        a_vec_tar[:,1] = 1 - a_vec_tar[:,1]
        a_vec_tar_plot = a_vec_tar * np.concatenate([s_vec, 1-s_vec], axis=1)

    elif actor.a_dim == 1:
        a_vec_plot = (1-a_vec) * s_vec
        a_vec_tar_plot = (1-a_vec) * s_vec
    else:
        raise ValueError("plot_actor_critic: Unexpected actor dimension")

    # graphs of ground-truth functions
    h_ast, a_ast = env.optimal_bellman(N)

    fig = plt.figure(1)
    fig.set_size_inches(18.5, 10.5)

    # plot the policy function
    labels = list()
    for k in range(actor.a_dim):
        plt.subplot(2,3,k+1)
        plt.plot(s_vec, a_vec_plot[:,k], 'b-')
        labels.append("policy")

        plt.plot(s_vec, a_vec_tar_plot[:,k], 'r-')
        labels.append("target policy")

        plt.plot(s_vec, a_ast[k, :], 'c-')
        labels.append("true policy")

        plt.legend(labels)
        plt.title("policy function {}".format(k))

    # plot the state-value function
    plt.subplot(2,3,3)
    labels = list()

    bias = np.mean(h_ast-h_vec_tar)
    plt.plot(s_vec, h_vec + bias, 'b--')
    labels.append("state-value")

    plt.plot(s_vec, h_vec_tar + bias, 'r--')
    labels.append("target state-value")

    plt.plot(s_vec, h_ast, 'c--')
    labels.append("true state-value")

    plt.legend(labels)
    plt.title("state-value function")

    # plot the state histogram
    plt.subplot(2,3,4)
    weights = np.ones_like(z) / len(z)
    n, bins, _ = plt.hist(z, N, range=(0.,1.), weights=weights, facecolor='green')
    weights = np.ones_like(z_tar) / len(z_tar)
    n_tar, bins_tar, _ = plt.hist(z_tar, N, range=(0.,1.), weights=weights, facecolor='blue')

    plt.legend(["regular","target"])
    plt.title("state histogram")

    plt.subplot(2,3,5)
    plt.semilogy(np.array(ro_train), 'r--')
    plt.semilogy(np.array(ro_train_tar), 'c--')
    plt.legend(["regular","target"])
    plt.title("average reward vs. episodes; average ro: {:2.6f} average ro target: {:2.6f}".format(ro, ro_tar))

    logger.info("dominant values:")
    for a,b in sorted(zip(n,bins), key=lambda x: x[0])[-4:]:
        logger.info("{:2.2f}-{:2.2f}:\t{:2.5f}".format(b, b + 1/N, a))

    logger.info("dominant values target:")
    for a,b in sorted(zip(n_tar,bins_tar), key=lambda x: x[0])[-4:]:
        logger.info("{:2.2f}-{:2.2f}:\t{:2.5f}".format(b, b + 1/N, a))

    plot_name = 'plot.png' if name is None else 'plot-{}.png'.format(name)
    plt.savefig(os.path.join(config.directory, "plots", plot_name))
    plt.close()


def run_episode(env, actor, actor_noise, critic, replay_buffer, ro_avg, pol_eval=None, episode_len=None):
    if episode_len is None:
        episode_len = config.episode_len
    transient = 20
    batch_size = config.batch_size

    ep_reward = 0
    avg_actor_grad_norm = 0
    avg_critic_grad_norm = 0
    s = env.reset()

    for j in range(episode_len):

        a = actor.predict(np.reshape(s, (config.env_size, actor.s_dim)))
        # Added exploration noise
        if actor_noise:
            # n = actor_noise()
            # actor_noise.sigma_dec()
            # a = np.minimum(np.maximum(a + n, 0.), 1.)
            a = a if np.random.rand() > actor_noise.sigma else np.random.rand(a.shape[0], a.shape[1])
            actor_noise.sigma_dec()

        s2, r = env.step(a)

        for k, (ss, aa, rr, ss2) in enumerate(zip(s, a, r, s2)):
            replay_buffer.add(np.reshape(ss, (actor.s_dim,)), np.reshape(aa, (actor.a_dim,)), np.reshape(rr, (1,)),
                              np.reshape(ss2, (actor.s_dim,)))

        # Keep adding experience to the memory until there are at least minibatch size samples
        if replay_buffer.size() > batch_size:
            s_batch, a_batch, r_batch, s2_batch = \
                replay_buffer.sample_batch(config.batch_size)
        else:
            logger.debug("Replay Buffer is filling up...")
            continue

        # Update the actor policy using the sampled gradient after policy evaluation
        target_q = critic.predict_target(
            s2_batch, actor.predict_target(s2_batch))

        # produce TD error with estimated Q(s2, a(s2))
        y_i = r_batch - ro_avg + critic.gamma * target_q

        # Update the critic given the targets
        critic_grad_norm  = critic.train(s_batch, a_batch, np.reshape(y_i, (batch_size, 1)))
        avg_critic_grad_norm += critic_grad_norm
        critic.update_target_network()

        if pol_eval is None:
            if j % 3 == 0:
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor_grad_norm = actor.train(s_batch, grads[0])
                avg_actor_grad_norm += actor_grad_norm
                # Update target networks
                actor.update_target_network()

        if j >= transient:
            ep_reward += np.mean(r)

        s = s2

    ep_ave_reward = ep_reward / (episode_len - transient)
    avg_actor_grad_norm /= episode_len / 3
    avg_critic_grad_norm /= episode_len

    return ep_ave_reward


def train(sess, env, env_eval, actor, critic, actor_noise, actor_eval, critic_eval, saver):

    replay_buffer = ReplayBuffer(config.buffer_size, config.seed)

    ro_list, ro_tar_list = list(), list()

    logger.info("evaluating initial average reward")
    t = time.time()
    ro, ro_tar = test_actor(env_eval, actor_eval, critic_eval, ro_list, ro_tar_list,
                            eval_len=config.eval_episode_len, name=0)
    ro_list.append(ro), ro_tar_list.append(ro_tar)
    ev_elapsed = time.time() - t

    # print result table title
    logger.info('{:^5} | {:^7} | {:^5} | {:^5} | {:^11} | {:^9} | {:^9} |'
                .format('Epi.', 'Noise std', 'Epi. secs', 'Eval secs', 'Epi. Avg Ro', 'Ro', 'Ro Target'))

    logger.info("performing initial episode of policy evaluation")
    t = time.time()
    ep_ave_reward = run_episode(env, actor, actor_noise, critic, replay_buffer, ro_list[-1],
                                pol_eval=True, episode_len=1000)
    ep_elapsed = time.time() - t

    logger.info('{:^5d} | {:^10.4f} | {:^9d} | {:^9d} | {:^11.05f} | {:^9.05f} | {:^9.05f} |'
                .format(-1, actor_noise.sigma, int(ep_elapsed),
                        int(ev_elapsed), ep_ave_reward, ro, ro_tar))

    for i in range(config.episode_num):
        logger.debug("global step actor: {}".format(actor.global_step()))
        logger.debug("global step critic: {}".format(critic.global_step()))

        t = time.time()
        ep_ave_reward = run_episode(env, actor, actor_noise, critic, replay_buffer, ro_list[-1])
        ep_elapsed = time.time() - t

        if (i+1) % 200 == 0:
            logger.info('{:^5} | {:^7} | {:^5} | {:^5} | {:^11} | {:^9} | {:^9} |'
                        .format('Epi.', 'Noise std', 'Epi. secs', 'Eval secs', 'Epi. Avg Ro', 'Ro', 'Ro Target'))


        plot = True if i % 25 == 0 else None

        t = time.time()
        ro, ro_tar = test_actor(env_eval, actor_eval, critic, ro_list, ro_tar_list,
                                eval_len=config.eval_episode_len, name=i, plot=plot)
        ro_list.append(ro), ro_tar_list.append(ro_tar)
        ev_elapsed = time.time() - t

        logger.info('{:^5d} | {:^10.4f} | {:^9d} | {:^9d} | {:^11.05f} | {:^9.05f} | {:^9.05f} |'
                    .format(i,  actor_noise.sigma , int(ep_elapsed),
                            int(ev_elapsed), ep_ave_reward, ro, ro_tar))


    logger.info("saving actor critic .... ")
    saver.save(sess, os.path.join(config.directory,"actor_critic"))

    return ro_list, ro_tar_list


def evaluate(env, predict, T):
    s_list = list()
    s = env.reset()
    ep_reward = 0

    for j in range(T):
        a = predict(np.reshape(s, (env.size, env.state_dim)))
        a = np.minimum(np.maximum(a, 0.0), 1.)
        s2, r = env.step(a)
        s = s2
        if j >= config.eval_transient:
            s_list.append(s)
            ep_reward += np.mean(r)

    ep_ave_reward = ep_reward / (T-config.eval_transient)

    return ep_ave_reward, np.array(s_list).flatten()


def test_actor(env, actor, critic, ro_train, ro_train_tar, eval_len=None, name=None, plot=None, verbose=None):
    eval_len = config.test_episode_len if eval_len is None else eval_len

    if verbose is not None:
        logger.info("evaluating policy for {} steps...".format(eval_len))

    ro, s_vec = evaluate(env, actor.predict, eval_len)
    ro_tar, s_vec_tar = evaluate(env, actor.predict_target, eval_len)
    actor.ro_summary(ro, ro_tar)

    if verbose is not None:
        logger.info('Ro: {:.07f} Ro target: {:.07f}'.format(ro, ro_tar))

    if plot is not None:
        plot_actor_critic(env, actor, critic, config.plot_bins, ro_train, ro_train_tar, s_vec, ro, s_vec_tar, ro_tar, name)

    return ro, ro_tar


def build_env(env_name):
    if env_name == "trapdoor":
        env = Trapdoor(config.env_size)
        env_eval = Trapdoor(config.env_eval_size)
    elif env_name == "ising":
        env = Ising(config.env_size)
        env_eval = Ising(config.env_eval_size)
    elif env_name == "bec_nc1":
        env = Bec_nc1(config.env_size, 0.2)
        env_eval = Bec_nc1(config.env_eval_size, 0.2)
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

        if config.noise_std:
            actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim),
                                                       sigma=config.noise_std,
                                                       sigma_dec=config.noise_dec)
        else:
            actor_noise = None

    with tf.variable_scope("model", reuse=True):
        with tf.name_scope("actor"):
            actor_eval = ActorNetwork(sess, env_eval, config, is_training=False)

        with tf.name_scope("critic"):
            critic_eval = CriticNetwork(sess, env_eval, config, is_training=False)

    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    return actor, critic, actor_eval, critic_eval, actor_noise


def main():

    with tf.Session(config=sess_config) as sess:

        np.random.seed(config.seed)
        tf.set_random_seed(config.seed)

        # build environment and actor/critic
        env, env_eval = build_env(config.env)
        actor, critic, actor_eval, critic_eval, actor_noise = build_actor_critic(sess, env, env_eval)
        saver = tf.train.Saver(tf.global_variables())

        # train actor and critic
        ro_list, ro_tar_list = train(sess, env, env_eval, actor, critic, actor_noise, actor_eval, critic_eval, saver)

        # evaluate final policy
        test_actor(env_eval, actor_eval, critic, ro_list, ro_tar_list,
                   eval_len=config.test_episode_len, name="final", plot=True, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    ######## randomness #########
    parser.add_argument('--seed',           type=int, default=None,         help='random seed for repeatability')

    ######## environment ########
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
    parser.add_argument('--opt',            type=str, default=None,         help='optimezr from {adam}')
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
    parser.add_argument('--name',           type=str, default=None,         help='simulation name')

    parser.add_argument('--verbose',        dest='verbose',                 action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--debug',          dest='debug',                   action='store_true')
    parser.set_defaults(debug=None)

    args = parser.parse_args()

    config, sess_config, logger = preprocess(args)

    main()