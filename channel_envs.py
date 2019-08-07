import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import os
logger = logging.getLogger("logger")


def H_b(z):
    h_b = -(np.multiply(z, np.log2(z + 1e-20)) + np.multiply(1 - z, np.log2(1 - z + 1e-20)))

    return h_b

# Parent Class for a general unifilar finite state channel.
# That is p(y,s'|s,x) = p(y|s,x)1[s'=f(x,s,y)]
class Unifilar(object):
    def __init__(self, size, P_out, P_state):
        self._size = size
        self._capacity = None
        # self._state_dim = state_dim
        # self._action_dim = action_dim
        self.input_cardin, self.state_cardin, self.output_cardin = P_out.shape
        self._P_out = P_out
        self._P_state = P_state
        self._eps = 1e-10
        self.z = None

    def _expand_u(self, a):
        u = np.reshape(a, [self._size, self.input_cardin, self.state_cardin])
        return u

    def _expand_z(self):
        p_s_feedback = np.concatenate([self.z, 1 - np.sum(self.z, axis=2, keepdims=True)], axis=2)
        return np.tile(p_s_feedback, [1, self.input_cardin, 1])

    def _reduce_z(self, z):
        return z[:, :, :self.state_cardin-1]

    def step(self, a):
        if np.max(a) > 1 or np.min(a) < 0:
            raise ValueError("Environment got invalid action")
        if self.z is None:
            raise ValueError("DP state is not initialized")

        u = self._expand_u(a)
        z = self._expand_z()

        r = self._reward(z,u)
        z_prime = self._next_state(z, u)

        self.z = z_prime

        return self.z, r

    def _reward(self,z,u):
        y_cardin = self.output_cardin
        x_cardin = self.input_cardin
        s_cardin = self.state_cardin

        p_xsy = np.zeros([self._size,x_cardin,s_cardin,y_cardin])

        for i in range(y_cardin):
            p_xsy[:, :, :, i] = z * u * np.expand_dims(self._P_out[:, :, i], axis=0)

        self._p_y = p_y = np.sum(p_xsy, axis=(1,2)) / np.reshape(np.sum(p_xsy + self._eps, axis=(1,2,3)), [self._size,1])

        if np.sum(np.logical_and(self._p_y >= 0.0, self._p_y <= 1.0 )*1) != self._p_y.size:
            print("p_y err: min {:2.2f} max {:2.2f}".format(np.min(self._p_y), np.max(self._p_y)))
        if np.mean(np.logical_or(self._p_y >= 0.0, self._p_y <= 1.0 )*1) != 1:
            print("invalid diturbance distribution")
            print(z)
            exit(1)

        assert np.mean(np.logical_or(p_xsy >= 0.0, p_xsy <= 1.0 )*1) == 1, "invalid joint distribution"

        # r = np.log2(p_y + self._eps)
        r = np.sum(-p_y * np.log2(p_y + self._eps), axis=1) - \
            np.sum(-p_xsy * np.log2(self._P_out + self._eps), axis=(1,2,3))


        return np.reshape(r, [self._size, 1])

    def _next_state(self, z,u):
        y_cardin = self._P_out.shape[2]
        x_cradin = self._P_out.shape[0]
        s_cardin = self._P_out.shape[1]

        def f(z,u,p_o,p_s):
            p = np.zeros([self._size,x_cradin,s_cardin,y_cardin])
            for k in range(p_s.shape[-1]):
                p[:, :, :, k] = z * u * np.expand_dims(p_o, axis=0) * np.expand_dims(p_s[:, :, k], axis=0)
            return np.sum(p, axis=(1,2)) / np.reshape(np.sum(p, axis=(1,2,3)) + self._eps, [self._size,1])

        def choose_w():
            w = self.w
            z_prime = np.zeros_like(w)
            for a in np.unique(w):
                z_prime = np.where(w==a,f(z,u,self._P_out[:, :, a] ,self._P_state[:, :, a, :]), z_prime)
            return np.expand_dims(z_prime, axis=1)

        def rand_w():
            p_cum = np.cumsum(self._p_y, axis=1)
            noise = np.tile(np.random.rand(self._size, 1), [1, y_cardin])
            noise2 = (noise < p_cum) * 1
            w = np.reshape(np.argmax(noise2, axis=1), [self._size, 1])
            return w

        self.w = rand_w()
        z_prime = choose_w()

        return self._reduce_z(z_prime)

    def sample_next_states(self, p_y_batch, s2_batch):
        p_cum = np.cumsum(p_y_batch, axis=1)
        noise = np.tile(np.random.rand(p_y_batch.shape[0], 1), [1, p_y_batch.shape[1]])
        noise2 = (noise < p_cum) * 1
        w = np.reshape(np.argmax(noise2, axis=1), [p_y_batch.shape[0], 1])

        s2_batch_ = np.zeros([s2_batch.shape[0], s2_batch.shape[1]])
        for y in range(self.output_cardin):
            s2_batch_ = np.where(w == y, s2_batch[:, :, y], s2_batch_)
        return s2_batch_

    def reset(self):
        self.z = np.random.rand(self._size, 1, self.state_cardin)
        self.z /= np.sum(self.z, axis=2, keepdims=True)
        self.z = self._reduce_z(self.z)
        return self.z

    def plot(self, actor, critic, N, ro_train, ro_train_tar, z, ro, z_tar, ro_tar, directory, name=None):
        def extract_a(a):
            a_vec = np.reshape(a, [-1, self.input_cardin, self.state_cardin])
            a_vec = a_vec[:, 0, :]
            a_vec[:, 1] = 1 - a_vec[:, 1]
            a_vec = a_vec * np.concatenate([s_vec, 1 - s_vec], axis=1)
            return a_vec

        def plot_action():
            labels = list()
            for k in range(a_vec.shape[-1]):
                plt.subplot(2, 3, k + 1)
                plt.plot(s_vec, a_vec[:, k], 'b-')
                labels.append("policy")

                plt.plot(s_vec, a_vec_tar[:, k], 'r-')
                labels.append("target policy")

                plt.plot(s_vec, a_ast[k, :], 'c-')
                labels.append("true policy")

                plt.legend(labels)
                plt.title("policy function {}".format(k))

        def plot_value_function():
            plt.subplot(2, 3, 3)
            labels = list()

            if h_ast is not None and a_ast is not None:
                bias = np.mean(h_ast - h_vec_tar)
            else:
                bias = 0

            plt.plot(s_vec, h_vec + bias, 'b--')
            labels.append("state-value")

            plt.plot(s_vec, h_vec_tar + bias, 'r--')
            labels.append("target state-value")

            plt.plot(s_vec, h_ast, 'c--')
            labels.append("true state-value")

            plt.legend(labels)
            plt.title("state-value function")

        def plot_state_histogram():
            plt.subplot(2, 3, 4)
            weights = np.ones_like(z) / len(z)
            n, bins, _ = plt.hist(z, N, range=(0., 1.), weights=weights, facecolor='green')
            weights = np.ones_like(z_tar) / len(z_tar)
            n_tar, bins_tar, _ = plt.hist(z_tar, N, range=(0., 1.), weights=weights, facecolor='blue')

            plt.legend(["regular", "target"])
            plt.title("state histogram")

            # logger.info("dominant values:")
            # for a, b in sorted(zip(n, bins), key=lambda x: x[0])[-4:]:
            #     logger.info("{:2.2f}-{:2.2f}:\t{:2.5f}".format(b, b + 1 / N, a))
            #
            # logger.info("dominant values target:")
            # for a, b in sorted(zip(n_tar, bins_tar), key=lambda x: x[0])[-4:]:
            #     logger.info("{:2.2f}-{:2.2f}:\t{:2.5f}".format(b, b + 1 / N, a))

        def plot_average_reward():
            plt.subplot(2, 3, 5)
            plt.semilogy(np.array(ro_train), 'r--')
            plt.semilogy(np.array(ro_train_tar), 'c--')
            plt.legend(["regular", "target"])
            plt.title("average reward vs. episodes; average ro: {:2.6f} average ro target: {:2.6f}".format(ro, ro_tar))

        def discretize_states():
            pass

        def estimate_transition_mat():
            pass

        def save_figure():
            plot_name = 'plot.png' if name is None else 'plot{}.png'.format(name)
            plt.savefig(os.path.join(directory, "plots", plot_name))
            plt.close()

        # grid the state space
        s_vec = np.reshape(np.linspace(0, 1, N), (N, actor.s_dim))

        # compute policy and state-value function
        a_vec_raw = actor.predict(s_vec)
        h_vec = critic.predict(s_vec, a_vec_raw)
        a_vec = extract_a(a_vec_raw)

        # compute policy and state-value function of target network
        a_vec_tar_raw = actor.predict_target(s_vec)
        h_vec_tar = critic.predict_target(s_vec, a_vec_tar_raw)
        a_vec_tar = extract_a(a_vec_tar_raw)

        # graphs of ground-truth functions
        h_ast, a_ast = self.optimal_bellman(N)

        fig = plt.figure(1)
        fig.set_size_inches(18.5, 10.5)

        # plot the policy function
        plot_action()

        # plot the state-value function
        plot_value_function()

        # plot the state histogram
        plot_state_histogram()

        # plot average reward vs episodes
        plot_average_reward()

        save_figure()

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._capacity

    @property
    def p_y(self):
        return self._p_y

    def optimal_bellman(self, N):
        return None, None


class Unifilar2(object):
    def __init__(self, size, P_out, P_state):
        self._size = size
        self._capacity = None
        # self._state_dim = state_dim
        # self._action_dim = action_dim
        self.input_cardin, self.state_cardin, self.output_cardin = P_out.shape
        self._P_out = P_out
        self._P_state = P_state
        self._eps = 1e-10
        self.z = None

    def _expand_u(self, a):
        u = np.reshape(a, [self._size, self.input_cardin, self.state_cardin])
        return u

    def _expand_z(self):
        z = np.expand_dims(self.z, axis=1)
        z_cat = np.concatenate([z, 1 - np.sum(z, axis=2, keepdims=True)], axis=2)
        z_tile = np.tile(z_cat, [1, self.input_cardin, 1])
        # p_s_feedback = np.concatenate([np.expand_dims(self.z, axis=1), 1 - np.sum(self.z, axis=1, keepdims=True)], axis=2)
        # return np.tile(p_s_feedback, [1, self.input_cardin, 1])
        return z_tile

    def _reduce_z(self, z):
        return z[:, :self.state_cardin-1, :]

    def step(self, a):
        if np.max(a) > 1 or np.min(a) < 0:
            raise ValueError("Environment got invalid action")
        if self.z is None:
            raise ValueError("DP state is not initialized")

        u = self._expand_u(a)
        z = self._expand_z()

        r = self._reward(z,u)
        z_prime = self._next_state(z, u)

        # self.z = z_prime[:,:,self.w]
        self.z = np.zeros_like(self.z)
        for y in range(self.output_cardin):
            self.z = np.where(self.w == y, z_prime[:, :, y], self.z)
        # self.z = np.where(self.w == 0, z_prime[:, :, 0], z_prime[:, :, 1])

        return z_prime, r

    def _reward(self,z,u):
        y_cardin = self.output_cardin
        x_cardin = self.input_cardin
        s_cardin = self.state_cardin

        p_xsy = np.zeros([self._size,x_cardin,s_cardin,y_cardin])

        for i in range(y_cardin):
            p_xsy[:, :, :, i] = z * u * np.expand_dims(self._P_out[:, :, i], axis=0)

        self._p_y = p_y = np.sum(p_xsy, axis=(1,2)) / np.reshape(np.sum(p_xsy + self._eps, axis=(1,2,3)), [self._size,1])

        if np.sum(np.logical_and(self._p_y >= 0.0, self._p_y <= 1.0 )*1) != self._p_y.size:
            print("p_y err: min {:2.2f} max {:2.2f}".format(np.min(self._p_y), np.max(self._p_y)))
        if np.mean(np.logical_or(self._p_y >= 0.0, self._p_y <= 1.0 )*1) != 1:
            print("invalid diturbance distribution")
            print(z)
            exit(1)

        assert np.mean(np.logical_or(p_xsy >= 0.0, p_xsy <= 1.0 )*1) == 1, "invalid joint distribution"

        # r = np.log2(p_y + self._eps)
        r = np.sum(-p_y * np.log2(p_y + self._eps), axis=1) - \
            np.sum(-p_xsy * np.log2(self._P_out + self._eps), axis=(1,2,3))


        return np.reshape(r, [self._size, 1])

    def _next_state(self, z,u):
        y_cardin = self._P_out.shape[2]
        x_cradin = self._P_out.shape[0]
        s_cardin = self._P_out.shape[1]

        def f(z,u,p_o,p_s):
            p = np.zeros([self._size,x_cradin,s_cardin,y_cardin])
            for k in range(p_s.shape[-1]):
                p[:, :, :, k] = z * u * np.expand_dims(p_o, axis=0) * np.expand_dims(p_s[:, :, k], axis=0)
            return np.sum(p, axis=(1,2)) / np.reshape(np.sum(p, axis=(1,2,3)) + self._eps, [self._size,1])

        def choose_w():
            # w = self.w
            z_prime = np.zeros([z.shape[0],z.shape[1], y_cardin])
            for y in range(y_cardin):
                z_prime[:,:,y] = f(z,u,self._P_out[:, :, y] ,self._P_state[:, :, y, :])
            return z_prime

        def rand_w():
            p_cum = np.cumsum(self._p_y, axis=1)
            noise = np.tile(np.random.rand(self._size, 1), [1, y_cardin])
            noise2 = (noise < p_cum) * 1
            w = np.reshape(np.argmax(noise2, axis=1), [self._size, 1])
            return w

        self.w = rand_w()
        z_prime = choose_w()

        return self._reduce_z(z_prime)

    def sample_next_states(self, p_y_batch, s2_batch):
        p_cum = np.cumsum(p_y_batch, axis=1)
        noise = np.tile(np.random.rand(p_y_batch.shape[0], 1), [1, p_y_batch.shape[1]])
        noise2 = (noise < p_cum) * 1
        w = np.reshape(np.argmax(noise2, axis=1), [p_y_batch.shape[0], 1])

        s2_batch_ = np.zeros([s2_batch.shape[0], s2_batch.shape[1]])
        for y in range(self.output_cardin):
            s2_batch_ = np.where(w == y, s2_batch[:, :, y], s2_batch_)
        return s2_batch_

    def reset(self):
        self.z = np.random.rand(self._size, self.state_cardin)
        self.z /= np.sum(self.z, axis=1, keepdims=True)
        self.z =  self.z[:, :self.state_cardin-1]
        return self.z

    def plot(self, actor, critic, N, ro_train, ro_train_tar, z, ro, z_tar, ro_tar, directory, name=None):
        def extract_a(a):
            a_vec = np.reshape(a, [-1, self.input_cardin, self.state_cardin])
            a_vec = a_vec[:, 0, :]
            a_vec[:, 1] = 1 - a_vec[:, 1]
            a_vec = a_vec * np.concatenate([s_vec, 1 - s_vec], axis=1)
            return a_vec

        def plot_action():
            labels = list()
            for k in range(a_vec.shape[-1]):
                plt.subplot(2, 3, k + 1)
                plt.plot(s_vec, a_vec[:, k], 'b-')
                labels.append("policy")

                plt.plot(s_vec, a_vec_tar[:, k], 'r-')
                labels.append("target policy")

                plt.plot(s_vec, a_ast[k, :], 'c-')
                labels.append("true policy")

                plt.legend(labels)
                plt.title("policy function {}".format(k))

        def plot_value_function():
            plt.subplot(2, 3, 3)
            labels = list()

            if h_ast is not None and a_ast is not None:
                bias = np.mean(h_ast - h_vec_tar)
            else:
                bias = 0

            plt.plot(s_vec, h_vec + bias, 'b--')
            labels.append("state-value")

            plt.plot(s_vec, h_vec_tar + bias, 'r--')
            labels.append("target state-value")

            plt.plot(s_vec, h_ast, 'c--')
            labels.append("true state-value")

            plt.legend(labels)
            plt.title("state-value function")

        def plot_state_histogram():
            plt.subplot(2, 3, 4)
            weights = np.ones_like(z) / len(z)
            n, bins, _ = plt.hist(z, N, range=(0., 1.), weights=weights, facecolor='green')
            weights = np.ones_like(z_tar) / len(z_tar)
            n_tar, bins_tar, _ = plt.hist(z_tar, N, range=(0., 1.), weights=weights, facecolor='blue')

            plt.legend(["regular", "target"])
            plt.title("state histogram")

            # logger.info("dominant values:")
            # for a, b in sorted(zip(n, bins), key=lambda x: x[0])[-4:]:
            #     logger.info("{:2.2f}-{:2.2f}:\t{:2.5f}".format(b, b + 1 / N, a))
            #
            # logger.info("dominant values target:")
            # for a, b in sorted(zip(n_tar, bins_tar), key=lambda x: x[0])[-4:]:
            #     logger.info("{:2.2f}-{:2.2f}:\t{:2.5f}".format(b, b + 1 / N, a))

        def plot_average_reward():
            plt.subplot(2, 3, 5)
            plt.semilogy(np.array(ro_train), 'r--')
            plt.semilogy(np.array(ro_train_tar), 'c--')
            plt.legend(["regular", "target"])
            plt.title("average reward vs. episodes; average ro: {:2.6f} average ro target: {:2.6f}".format(ro, ro_tar))

        def discretize_states():
            pass

        def estimate_transition_mat():
            pass

        def save_figure():
            plot_name = 'plot.png' if name is None else 'plot{}.png'.format(name)
            plt.savefig(os.path.join(directory, "plots", plot_name))
            plt.close()

        # grid the state space
        s_vec = np.reshape(np.linspace(0, 1, N), (N, actor.s_dim))

        # compute policy and state-value function
        a_vec_raw = actor.predict(s_vec)
        h_vec = critic.predict(s_vec, a_vec_raw)
        a_vec = extract_a(a_vec_raw)

        # compute policy and state-value function of target network
        a_vec_tar_raw = actor.predict_target(s_vec)
        h_vec_tar = critic.predict_target(s_vec, a_vec_tar_raw)
        a_vec_tar = extract_a(a_vec_tar_raw)

        # graphs of ground-truth functions
        h_ast, a_ast = self.optimal_bellman(N)

        fig = plt.figure(1)
        fig.set_size_inches(18.5, 10.5)

        # plot the policy function
        plot_action()

        # plot the state-value function
        plot_value_function()

        # plot the state histogram
        plot_state_histogram()

        # plot average reward vs episodes
        plot_average_reward()

        save_figure()

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._capacity

    @property
    def p_y(self):
        return self._p_y

    def optimal_bellman(self, N):
        return None, None


# Child classes that inherit the unifilar structure
# and modify the channel equations.
class Trapdoor(Unifilar):

    def __init__(self, size):
        def p_s(x, s, y, s_prime):
            next_s =  np.logical_xor(x, np.logical_xor(s, y)) * 1
            return (s_prime == next_s) * 1

        def p_y(x,s,y):
            if x == s:
                return (x == y)*1
            else:
                return 0.5

        self.state_cardin = 2
        self.input_cardin = 2
        self.output_cardin = 2

        P_out = np.array([[[p_y(x,s,y)       for y in range(self.output_cardin)]
                                             for s in range(self.state_cardin)]
                                             for x in range(self.input_cardin)])

        P_state = np.array([[[[p_s(x,s,y,s_prime)    for s_prime in range(self.state_cardin)]
                                                     for y in range(self.output_cardin)]
                                                     for s in range(self.state_cardin)]
                                                     for x in range(self.input_cardin)])

        super().__init__(size, P_out, P_state)

    @property
    def capacity(self):
        return np.log2((1+np.sqrt(5))/2)

    def optimal_bellman(self, N):
        s_vec = np.linspace(0, 1, N)
        z_0 = 0.0
        z_1 = 0.382
        z_2 = 0.613
        z_3 = 1.0
        ro_ast = self.capacity

        # optimal state-value function
        h_ast = np.zeros(N)
        h_ast[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = \
            H_b(s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]) - \
            ro_ast * s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)] + np.log2(np.sqrt(5)-1)
        h_ast[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = 1.0
        h_ast[np.logical_and(s_vec >= z_2, s_vec <= z_3)] =  \
            H_b(s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)]) + \
            ro_ast * s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)] + np.log2(3-np.sqrt(5))

        # optimal policy
        a_ast_0 = np.zeros(N)
        a_ast_0[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]
        a_ast_0[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = 0.5 * (3 - np.sqrt(5))
        a_ast_0[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = 0.5 * (np.sqrt(5) - 1) * \
                                                                s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)]

        a_ast_1 = np.zeros(N)
        a_ast_1[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = 0.5 * (np.sqrt(5) - 1) * \
                                                                (1 - s_vec[
                                                                    np.logical_and(s_vec >= z_0, s_vec <= z_1)])
        a_ast_1[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = 0.5 * (3 - np.sqrt(5))
        a_ast_1[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = (
                    1 - s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)])

        a_ast = np.stack((a_ast_0, a_ast_1), axis=0)

        return h_ast, a_ast


class Trapdoor3(Unifilar2):

    def __init__(self, size):
        def p_s(x, s, y, s_prime):
            # if x == y:
            #     next_s = s
            #     p_s = (s_prime == next_s) * 1
            # elif s == y:
            #     next_s = x
            #     p_s = (s_prime == next_s) * 1
            # else:
            #     p_s = 0
            p_s = (((x+s) % self.state_cardin) == s_prime)*1
            return p_s

        def p_y(x,s,y):
            if x == s:
                return (x == y)*1
            else:
                return 0.5* np.logical_or(x==y, s==y)

        self.state_cardin = 3
        self.input_cardin = 3
        self.output_cardin = 3

        P_out = np.array([[[p_y(x,s,y)       for y in range(self.output_cardin)]
                                             for s in range(self.state_cardin)]
                                             for x in range(self.input_cardin)])

        P_state = np.array([[[[p_s(x,s,y,s_prime)    for s_prime in range(self.state_cardin)]
                                                     for y in range(self.output_cardin)]
                                                     for s in range(self.state_cardin)]
                                                     for x in range(self.input_cardin)])
        super().__init__(size, P_out, P_state)

    @property
    def capacity(self):
        return np.log2((1+np.sqrt(5))/2)

    def plot(self, actor, critic, N, ro_train, ro_train_tar, z, ro, z_tar, ro_tar, directory, name=None):

        def plot_state_histogram():
            ax = fig.add_subplot(121, projection='3d')
            x, y = z[:,0], z[:,1]
            hist, xedges, yedges = np.histogram2d(x, y, bins=100, range=[[0, 1.0], [0, 1.0]])

            # Construct arrays for the anchor positions of the 16 bars.
            xpos, ypos = np.meshgrid(xedges[:-1] + 0.005, yedges[:-1] + 0.005, indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = np.zeros_like(xpos)

            # Construct arrays with the dimensions for the 16 bars.
            dx = dy = 0.01 * np.ones_like(zpos)
            dz = hist.ravel()

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')


        def plot_average_reward():
            plt.subplot(1, 2, 2)
            plt.semilogy(np.array(ro_train), 'r--')
            plt.semilogy(np.array(ro_train_tar), 'c--')
            plt.legend(["regular", "target"])
            plt.title("average reward vs. episodes; average ro: {:2.6f} average ro target: {:2.6f}".format(ro, ro_tar))


        def save_figure():
            plot_name = 'plot.png' if name is None else 'plot{}.png'.format(name)
            plt.savefig(os.path.join(directory, "plots", plot_name))
            plt.close()

        fig = plt.figure(1)
        fig.set_size_inches(18.5, 10.5)


        # plot the state histogram
        plot_state_histogram()

        # plot average reward vs episodes
        plot_average_reward()

        save_figure()


class Ising(Unifilar):
    def __init__(self, size):
        def p_s(x, s, y, s_prime):
            return (s_prime == x) * 1

        def p_y(x,s,y):
            if x == s:
                return (x == y)*1
            else:
                return 0.5

        self.state_cardin = 2
        self.input_cardin = 2
        self.output_cardin = 2

        P_out = np.array([[[p_y(x,s,y)       for y in range(self.output_cardin)]
                                             for s in range(self.state_cardin)]
                                             for x in range(self.input_cardin)])
        P_state = np.array([[[[ p_s(x, s, y, s_prime)   for s_prime in range(self.state_cardin)]
                                                        for y in range(self.output_cardin)]
                                                        for s in range(self.state_cardin)]
                                                        for x in range(self.input_cardin)])


        super().__init__(size, P_out, P_state)

    @property
    def capacity(self):
        return 0.575522


    def optimal_bellman(self, N):
        s_vec = np.linspace(0, 1, N)
        c = 0.4503
        z_0 = 0.0
        z_1 = (1-c)/(1+c)
        z_2 = (2*c)/(1+c)
        z_3 = 1.0
        ro_ast = self.capacity

        # optimal state-value function
        h_ast = np.zeros(N)
        s = s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)]
        h_ast[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = \
            1/(1-c) * H_b((2*c+(1-c)*s)/2) - \
            s + (c*s-4*c-s)/(2-2*c) * ro_ast + \
            (2*c + (1-c)*s) / (2 - 2*c) * H_b((2*c)/(c*(2-s)+s))

        h_ast[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = \
            H_b(s_vec[np.logical_and(s_vec >= z_1, s_vec <= z_2)])

        h_ast[np.logical_and(s_vec >= z_0, s_vec <= z_1)] =  h_ast[np.logical_and(s_vec >= z_2, s_vec <= z_3)][::-1]

        # optimal policy
        a_ast_0 = np.zeros(N)
        a_ast_0[np.logical_and(s_vec >=z_0, s_vec <= z_2)] = \
            s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_2)]
        a_ast_0[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = \
            c * (2 - s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)])

        a_ast_1 = np.zeros(N)
        a_ast_1[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = \
            c * (1 + s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)])
        a_ast_1[np.logical_and(s_vec >= z_1, s_vec <= z_3)] = \
            1 - s_vec[np.logical_and(s_vec >= z_1, s_vec <= z_3)]
        a_ast = np.stack((a_ast_0, a_ast_1), axis=0)

        return h_ast, a_ast


class Ising3(Unifilar2):
    def __init__(self, size, cardinality=3):
        def f_s(x, s, y, s_prime):
            if x == s_prime:
                p_s_prime = 1.0
            else:
                p_s_prime = 0.0
            return p_s_prime

        def f_y(x,s,y):
                if x == s:
                    p_y = (x == y)*1
                elif x == y:
                    p_y = 0.5
                elif s == y:
                    p_y = 0.5
                else:
                    p_y = 0.
                    # p_y = 0.5 * np.logical_or(x == y, s == y) #np.logical_or(multivar_ind(x, y), multivar_ind(s, y)) #if multivar_ind(x, s) != 1 else multivar_ind(x, y)
                return p_y

        self.cardinality = cardinality
        self.state_cardin = cardinality
        self.input_cardin = cardinality
        self.output_cardin = cardinality

        P_out = np.array([[[f_y(x,s,y)   for y in range(self.output_cardin)]
                                         for s in range(self.state_cardin)]
                                         for x in range(self.input_cardin)])

        P_state = np.array([[[[ f_s(x, s, y, s_prime)   for s_prime in range(self.state_cardin)]
                                                                    for y in range(self.output_cardin)]
                                                                    for s in range(self.state_cardin)]
                                                                    for x in range(self.input_cardin)])

        super().__init__(size, P_out, P_state)

        logger.info("maximal achievable rate is {:2.6f}".format(self.capacity))


    def plot(self, actor, critic, N, ro_train, ro_train_tar, z, ro, z_tar, ro_tar, directory, name=None):

        def plot_state_histogram_1d():
            plt.subplot(1, 2, 1)
            weights = np.ones_like(z) / len(z)
            n, bins, _ = plt.hist(z, N, range=(0., 1.), weights=weights, facecolor='green')
            weights = np.ones_like(z_tar) / len(z_tar)
            n_tar, bins_tar, _ = plt.hist(z_tar, N, range=(0., 1.), weights=weights, facecolor='blue')

            plt.legend(["regular", "target"])
            plt.title("state histogram")

        def plot_state_histogram_2d():
            ax = fig.add_subplot(121, projection='3d')
            x, y = z[:,0], z[:,1]
            hist, xedges, yedges = np.histogram2d(x, y, bins=100, range=[[0, 1.0], [0, 1.0]])

            # Construct arrays for the anchor positions of the 16 bars.
            xpos, ypos = np.meshgrid(xedges[:-1] + 0.005, yedges[:-1] + 0.005, indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = np.zeros_like(xpos)

            # Construct arrays with the dimensions for the 16 bars.
            dx = dy = 0.01 * np.ones_like(zpos)
            dz = hist.ravel()

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')

        def plot_average_reward():
            plt.subplot(1, 2, 2)
            plt.semilogy(np.array(ro_train), 'r--')
            plt.semilogy(np.array(ro_train_tar), 'c--')
            plt.legend(["regular", "target"])
            plt.title("average reward vs. episodes; average ro: {:2.6f} average ro target: {:2.6f}".format(ro, ro_tar))

        def save_figure():
            plot_name = 'plot.png' if name is None else 'plot{}.png'.format(name)
            plt.savefig(os.path.join(directory, "plots", plot_name))
            plt.close()

        fig = plt.figure(1)
        fig.set_size_inches(18.5, 10.5)

        if self.cardinality == 2:
            plot_state_histogram_1d()
        elif self.cardinality == 3:
            # plot the state histogram
            plot_state_histogram_2d()
        else:
            pass

        # plot average reward vs episodes
        plot_average_reward()

        save_figure()

    @property
    def capacity(self):
        p = np.linspace(0.0, 1.0, 10000)
        eps = 1e-10
        h_b = -2*(p * np.log2(p+eps) + (1-p) * np.log2((1-p)/(self.cardinality-1)+eps))
        c = np.max(h_b / (p + 3))

        return c

# Need to generalize for the latest formulation
class Bec_nc1(Unifilar):
    def __init__(self, size, e):
        def f(x, s, y):
            return x
        P_out = np.array([[[1-e, e, 0], [1-e, e, 0]], [[0, e, 1-e], [0, e, 1-e]]])
        P_state = np.array([[[[(s_prime == f(x,s,y)) * 1    for s_prime in range(2)]
                                                            for y in range(3)]
                                                            for s in range(2)]
                                                            for x in range(2)])
        super().__init__(size, P_out, P_state)
        self.e = e

    # def _next_state(self, z,u):
    #     y_cardinal = self._P_out.shape[-1]
    #
    #     def f(z,u,p_o,p_s):
    #         p = np.zeros([self._size, 2, 2, 2])
    #         p[:, :, :, 0] = z * u * np.expand_dims(p_o, axis=0) * np.expand_dims(p_s[:, :, 0], axis=0)
    #         p[:, :, :, 1] = z * u * np.expand_dims(p_o, axis=0) * np.expand_dims(p_s[:, :, 1], axis=0)
    #         return np.sum(p, axis=(1,2)) / np.reshape(np.sum(p, axis=(1,2,3)) + self._eps, [self._size,1])
    #
    #     assert np.mean(np.logical_or(self._p_y >= 0.0, self._p_y <= 1.0 )*1) == 1, "invalid diturbance distribution"
    #     # w = np.reshape(np.array([np.random.choice(y_cardinal, 1, p=self._p_y[i,:]) for i in range(self._size)]), [self._size, 1])
    #
    #     p_cum = np.cumsum(self._p_y, axis=1)
    #     noise = np.tile(np.random.rand(self._size, 1), [1, y_cardinal])
    #     noise2 = (noise < p_cum)*1
    #     w = np.reshape(np.argmax(noise2, axis=1), [self._size, 1])
    #
    #     z_prime = np.where(w == 0, f(z,u,self._P_out[:, :, 0],self._P_state[:, :, 0, :]),
    #                        np.where(w == 1, f(z, u, self._P_out[:, :, 1], self._P_state[:, :, 1, :]),
    #                                         f(z, u, self._P_out[:, :, 2], self._P_state[:, :, 2, :])))
    #     return self._reduce_z(z_prime)

    def _expand_u(self, a):
        # u = np.reshape(a, [self._size, 1, self._action_dim])
        # u = np.concatenate([u, 1-u], axis=1)
        u_1 = np.concatenate([np.ones([self._size, 1]), np.zeros([self._size, 1])], axis=1)
        # return np.concatenate([u, u_1], axis=2)
        u = np.reshape(a, [-1, self.input_cardin, self.state_cardin])
        u[:, :, 1] = u_1
        return u


    @property
    def capacity(self):
        eps = self.e
        p = np.linspace(0.0001, 0.5, 10000)

        h_b = -(p * np.log2(p) + (1-p) * np.log2(1-p))
        ro = np.max(h_b / (p + 1/(1-eps)))
        k = np.argmax(h_b / (p + 1/(1-eps)))
        p_e = p[k]
        return ro, p_e

    def optimal_bellman(self, N):
        s_vec = np.linspace(0, 1, N)
        e = self.e
        ro_ast, p_ast = self.capacity

        z_0 = 0.0
        z_1 = p_ast
        z_2 = 1.0

        # optimal state-value function
        h_ast = np.zeros(N)
        h_ast[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = ro_ast
        z = s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]
        h_ast[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = \
            (1-e) * H_b(z) - z * (1-e) * ro_ast

        # optimal policy
        z = s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]

        a_ast = np.zeros(N)
        a_ast[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = z
        a_ast[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = p_ast
        a_ast = np.reshape(a_ast, [1, N])
        return h_ast, a_ast

class Bec_121(Unifilar):
    state_dim = 3
    action_dim = 4

    state_cardin = 4
    input_cardin = 2
    output_cardin = 6

    def __init__(self, size, e):
        def multivar_ind(x, y):
            return (np.mean((x ==  y) * 1) == 1) * 1

        def f_s(x, s, y, s_prime):
            s = np.unpackbits(np.array([s], dtype=np.uint8))[-2:]
            p_s_prime = multivar_ind(s_prime, 2*x+s[0])
            return p_s_prime

        def f_y(x,s,y):
            if y == 5:
                return e
            else:
                s = np.unpackbits(np.array([s], dtype=np.uint8))[-2:]
                p_y = multivar_ind(y, x + 2*s[0] + s[1]) * (1-e)
                return p_y

        P_out = np.array([[[f_y(x,s,y)   for y in range(self.output_cardin)]
                                         for s in range(self.state_cardin)]
                                         for x in range(self.input_cardin)])

        P_state = np.array([[[[ f_s(x, s, y, s_prime)   for s_prime in range(self.state_cardin)]
                                                                    for y in range(self.output_cardin)]
                                                                    for s in range(self.state_cardin)]
                                                                    for x in range(self.input_cardin)])

        super().__init__(size, P_out, P_state)
        self.e = e

        print("channels capacity lower-bound is {:2.6f}".format(self.capacity))

    @property
    def capacity(self):
        eps = self.e
        ro = 1 - (2*eps ** 3)*(1+eps)/(eps ** 3 + eps ** 2 + 2)
        return ro

class Bec_Dicode(Unifilar):

    state_cardin = 2
    input_cardin = 2
    output_cardin = 4

    def __init__(self, size, e):
        def multivar_ind(x, y):
            return (np.mean((x ==  y) * 1) == 1) * 1

        def f_s(x, s, y):
            return x

        def f_y(x,s,y):
            if y == 3:
                return e
            else:
                return multivar_ind(y, x -s + 1) * (1-e)

        P_out = np.array([[[f_y(x,s,y)   for y in range(self.output_cardin)]
                                         for s in range(self.state_cardin)]
                                         for x in range(self.input_cardin)])

        P_state = np.array([[[[multivar_ind(s_prime, f_s(x,s,y))    for s_prime in range(self.state_cardin)]
                                                                    for y in range(self.output_cardin)]
                                                                    for s in range(self.state_cardin)]
                                                                    for x in range(self.input_cardin)])

        super().__init__(size, P_out, P_state)
        self.e = e
        print("channels capacity lower-bound is {:2.6f}".format(self.capacity))

    @property
    def capacity(self):
        eps = self.e
        ro = 1 - (2*eps ** 2)/(1+eps)
        return ro


#########################

