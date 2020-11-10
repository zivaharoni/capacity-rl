import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
import logging
import os
from time import time
logger = logging.getLogger("logger")


class UnifilarTF(object):
    def __init__(self, sess, P_out, P_state):
        self.sess = sess
        self._capacity = None
        self.input_cardin, self.state_cardin, self.output_cardin = P_out.shape
        self._P_out = tf.constant(P_out, dtype=tf.float32)
        self._P_state = tf.constant(P_state, dtype=tf.float32)
        self._create_env()


    def _create_env(self):
        self.z = tf.placeholder(tf.float32, shape=[None, self.state_cardin-1])
        self.u = tf.placeholder(tf.float32, shape=[None, self.input_cardin*self.state_cardin])
        self._size = tf.cast(tf.shape(self.z), dtype=tf.int64)[0]

        self._joint =  self._compute_joint()

        self._p_y = tf.reduce_sum(self._joint, axis=(1,2,4))

        self._compute_reward()

        self._randomize_symbols()

        next_states = tf.reduce_sum(self._joint, axis=(1,2),keepdims=True) / tf.reduce_sum(self._joint, axis=(1,2,4), keepdims=True)
        next_states = tf.reshape(next_states, shape=[-1, self.output_cardin, self.state_cardin])
        next_states = tf.clip_by_value(next_states, 0.0, 1.0)


        self._next_states = next_states = next_states[:,:,:-1]
        next_state_indices = tf.stack([tf.range(self._size), self._symbols], axis=1)
        self._z_prime = tf.gather_nd(next_states, next_state_indices)


    def _compute_joint(self):
        z = tf.concat([self.z, 1 - tf.reduce_sum(self.z, axis=1, keepdims=True)], axis=1)
        z = tf.expand_dims(z, axis=1)
        z = tf.reshape(z, [-1, 1, z.shape[2], 1, 1])

        u = tf.reshape(self.u, [-1, self.input_cardin, self.state_cardin, 1, 1])

        p_o = tf.reshape(self._P_out, [1, self._P_out.shape[0], self._P_out.shape[1], self._P_out.shape[2], 1])
        p_s = tf.expand_dims(self._P_state, axis=0)

        joint = z * u * p_o * p_s
        return joint

    def _compute_reward(self):
        p_xsy = tf.reduce_sum(self._joint, axis=4)

        py_arg = -self._p_y * tf.log(self._p_y)/tf.log(2.)
        py_arg = tf.where(tf.is_nan(py_arg), tf.zeros_like(py_arg), py_arg)

        pxsy_arg = -p_xsy * tf.log(self._P_out )/tf.log(2.)
        pxsy_arg = tf.where(tf.is_nan(pxsy_arg), tf.zeros_like(pxsy_arg), pxsy_arg)
        reward = tf.reduce_sum(py_arg, axis=1) - tf.reduce_sum(pxsy_arg, axis=(1,2,3))
        # reward = py_arg - tf.reduce_sum(pxsy_arg, axis=(1,2))

        self._reward = tf.reshape(reward, shape=[-1, 1])

    def _randomize_symbols(self):
        p_cum = tf.cumsum(self._p_y, axis=1)
        noise = tf.random_uniform(shape=[self._size, 1], minval=0.0, maxval=1.0, dtype=tf.float32)
        noise = tf.tile(noise, [1, self.output_cardin])

        chosen_symbol_raw = tf.where(tf.less_equal(noise,p_cum), tf.ones_like(noise), tf.zeros_like(noise))
        self._symbols = tf.argmax(chosen_symbol_raw, axis=1)

    def reset(self, size):
        # z = -np.log(np.random.rand(size, self.state_cardin))
        # z /= np.sum(z, axis=1, keepdims=True)
        # z =  z[:, :self.state_cardin-1]
        z = np.zeros([size, self.state_cardin-1])
        return z

    def constrained_u(self,u):
        return u

    def reward(self, z, u):
        u = self.constrained_u(u)
        return self.sess.run(self._reward, feed_dict={self.z: z, self.u: u})

    def symbols(self, z, u):
        u = self.constrained_u(u)
        return self.sess.run(self._symbols, feed_dict={self.z: z, self.u: u})

    def next_state(self, z, u):
        u = self.constrained_u(u)
        return self.sess.run(self._z_prime, feed_dict={self.z: z, self.u: u})

    def p_y(self, z, u):
        u = self.constrained_u(u)
        return self.sess.run(self._p_y, feed_dict={self.z: z, self.u: u})

    def step(self, z, u, planning=None):
        u = self.constrained_u(u)
        if planning is None:
            return self.sess.run([self._z_prime, self._reward, self._symbols], feed_dict={self.z: z, self.u: u})
        else:
            return self.sess.run([self._next_states, self._reward, self._p_y], feed_dict={self.z: z, self.u: u})

    @property
    def capacity(self):
        return self._capacity
    @property
    def P_out(self):
        return self._P_out

class Ising(UnifilarTF):
    def __init__(self, cardinality, sess):
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



        super().__init__(sess, P_out, P_state)

    def capacity(self):
        p = np.linspace(0.0, 1.0, 10000)
        eps = 1e-10
        h_b = -2*(p * np.log2(p+eps) + (1-p) * np.log2((1-p)/(self.cardinality-1)+eps))
        c = np.max(h_b / (p + 3))
        return c


class Trapdoor(UnifilarTF):
    def __init__(self, cardinality, sess):
        def f_s(x, s, y, s_prime):
            if x == y:
                if s == s_prime:
                    p_s_prime = 1.0
                else:
                    p_s_prime = 0.0
            elif s == y:
                if x == s_prime:
                    p_s_prime = 1.0
                else:
                    p_s_prime = 0.0
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



        super().__init__(sess, P_out, P_state)

    def capacity(self):
        if self.cardinality == 2:
            c = np.log2((1+np.sqrt(5))/2)
        else:
            c = "UNKNOWN"
        return c


class RLL_0_K(UnifilarTF):
    def __init__(self, cardinality, sess, eps=0.0):
        def f_s(x, s, y, s_prime):
            if s == self.cardinality-1 or x == 1:
                if s_prime == 0:
                    p_s_prime = 1.0
                else:
                    p_s_prime = 0.0
            else:
                if s+1 == s_prime:
                    p_s_prime = 1.0
                else:
                    p_s_prime = 0.0

            return p_s_prime

        def f_y(x,s,y):
                if y == 2:
                    p_y = self.eps
                elif x == y:
                    p_y = 1-self.eps
                else:
                    p_y = 0.0
                return p_y

        self.cardinality = cardinality+1
        self.state_cardin = cardinality+1
        self.input_cardin = 2
        self.output_cardin = 3
        self.eps = eps
        P_out = np.array([[[f_y(x,s,y)   for y in range(self.output_cardin)]
                                         for s in range(self.state_cardin)]
                                         for x in range(self.input_cardin)])

        P_state = np.array([[[[ f_s(x, s, y, s_prime)   for s_prime in range(self.state_cardin)]
                                                                    for y in range(self.output_cardin)]
                                                                    for s in range(self.state_cardin)]
                                                                    for x in range(self.input_cardin)])



        super().__init__(sess, P_out, P_state)

    def constrained_u(self,u):
        u = np.reshape(u,[-1, self.input_cardin, self.state_cardin])
        u[:,0,-1] = 0.
        u[:, 1, -1] = 1.
        u = np.reshape(u,[-1, self.input_cardin * self.state_cardin])
        return u


    def capacity(self):
        if self.eps == 0:
            def adjacency(k):
                A = np.zeros([k, k])
                A[:,0] = 1
                for m in range(k-1):
                    A[m,m+1] = 1
                return A

            A = adjacency(self.cardinality)
            c = np.log2(np.max(np.abs(np.linalg.eig(A)[0])))

        else:
            if self.cardinality == 2:
                c = "UNKNOWN"
            else:
                c = "UNKNOWN"
        return c
