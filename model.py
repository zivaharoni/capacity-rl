from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import tensorflow as tf
import tflearn
import numpy as np
import os
logger = logging.getLogger("logger")

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, env, config, is_training=None):
        self._set_hyper_parameters(sess, env, config, is_training)

        # Actor Network
        with tf.name_scope("online"):
            self._create_actor_network()

        if self.is_training:
            self._create_optimizer()

        # Target Network
        with tf.name_scope("target"):
            self._create_target_network()

    ######################## init methods ########################
    def _set_hyper_parameters(self, sess, env, config, is_training):
        self.sess = sess
        self.s_dim = env.state_cardin-1
        self.a_dim = env.input_cardin * env.state_cardin
        self.env = env

        self.learning_rate = tf.Variable(config.actor_lr, trainable=False, name="lr")
        self.learning_rate_decay = config.lr_decay
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.hidden_size = config.actor_hid
        self.layers = config.actor_layers
        self.is_training = is_training
        self.opt = config.opt

    def _create_actor_network(self):
        last_trainable_var = len(tf.trainable_variables())
        self.inputs, self.out = self._create_network()
        self.network_params = tf.trainable_variables()[last_trainable_var:]

    def _create_target_network(self):
        last_trainable_var = len(tf.trainable_variables())
        self.target_inputs, self.target_out = self._create_network(target=True)
        self.target_network_params = tf.trainable_variables()[last_trainable_var:]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.target_network_params[i], 1 - self.tau) +
                tf.multiply(self.network_params[i], self.tau))
            for i in range(len(self.target_network_params))]

        self.copy_target_network_params = \
            [self.target_network_params[i].assign(self.network_params[i])
            for i in range(len(self.target_network_params))]

    def _create_network(self, target=None):
        bias_init = tflearn.initializations.uniform(shape=None, minval=0.1, maxval=0.2,dtype=tf.float32)
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = inputs
        for i in range(self.layers):
            net = tflearn.fully_connected(net, self.hidden_size, name="fc-{}".format(i+1), bias_init=bias_init)
            net = tflearn.layers.normalization.batch_normalization(net, name="bn-{}".format(i+1))

            net = tflearn.activations.relu(net)

            # if i < self.layers-1:
            #     net = tflearn.activations.relu(net)
            # else:
            #     net = tflearn.activations.tanh(net)

        bias_init = tflearn.initializations.uniform(shape=None, minval=-0.04, maxval=0.04,dtype=tf.float32)
        out = tflearn.fully_connected(
            net, self.a_dim, bias_init=bias_init)

        if target is None:
            self.last_hidden = out = tflearn.layers.normalization.batch_normalization(out, name="bn-out")
            self.noise =  tf.zeros_like(out, name="noise")
            out += self.noise

        out = tf.reshape(out , [-1, self.env.input_cardin, self.env.state_cardin])
        out = tf.reshape(tf.nn.softmax(out, axis=1), [-1, self.a_dim])

        return inputs, out

    def _create_optimizer(self):
        # global counter
        self._global_step = tf.Variable(0.0, trainable=False, name="global_step")

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, tf.cast(tf.shape(self.out)[0],tf.float32)), self.actor_gradients))

        _, self.grad_norm = tf.clip_by_global_norm(self.actor_gradients, 0.5)
        # self.actor_gradients, self.grad_norm = tf.clip_by_global_norm(self.actor_gradients, 0.5)

        # Optimization Op
        if self.opt == "adam":
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.opt == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise ValueError("{} is not a valid optimizer".format(self.opt))

        self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params),
                                                       global_step=self._global_step)

        # learning rate decay
        self._lr_decay = tf.assign(self.learning_rate, self.learning_rate * self.learning_rate_decay)

    ######################## computational methods ########################
    def train(self, inputs, a_gradient):
        _, global_step, g_norm = self.sess.run([self.optimize, self._global_step, self.grad_norm], feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
        return global_step, g_norm

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_noisy(self, inputs, noise):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.noise: noise
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def copy_target_network(self):
        self.sess.run(self.copy_target_network_params)

    def lr_decay(self):
        self.sess.run(self._lr_decay)

    @property
    def lr(self):
        return self.sess.run(self.learning_rate)

    @property
    def global_step(self):
        return self.sess.run(self._global_step)


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, env, config, is_training=None):
        self._set_hyper_parameters(sess, env, config, is_training)

        # Critic network
        self._create_critic_network()

        if self.is_training:
            self._create_optimizer()

        # Target Network
        self._create_target_network()


    ######################## init methods ########################
    def _set_hyper_parameters(self, sess, env, config, is_training):
        self.sess = sess
        self.s_dim = env.state_cardin-1
        self.a_dim = env.input_cardin * env.state_cardin
        self.env = env

        self.learning_rate = tf.Variable(config.critic_lr, trainable=False)
        self.learning_rate_decay = config.lr_decay
        self.tau = config.tau
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.hidden_size = config.critic_hid
        self.layers = config.critic_layers
        self.is_training = is_training
        self.opt = config.opt

    def _create_critic_network(self):
        last_trainable_var = len(tf.trainable_variables())
        self.inputs, self.action, self.out = self._create_network()
        self.network_params = tf.trainable_variables()[last_trainable_var:]

    def _create_target_network(self):
        last_trainable_var = len(tf.trainable_variables())
        self.target_inputs, self.target_action, self.target_out = self._create_network()
        self.target_network_params = tf.trainable_variables()[last_trainable_var:]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.target_network_params[i], 1 - self.tau) +
                tf.multiply(self.network_params[i], self.tau))
            for i in range(len(self.target_network_params))]

        self.copy_target_network_params = \
            [self.target_network_params[i].assign(self.network_params[i])
            for i in range(len(self.target_network_params))]

    def _create_network(self):
        bias_init = tflearn.initializations.uniform(shape=None, minval=0.0, maxval=0.1,dtype=tf.float32)

        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])

        net_s = inputs
        for i in range(self.layers):
            net_s = tflearn.fully_connected(net_s, self.hidden_size, name="fc-{}-s".format(i+1), bias=False, bias_init=bias_init)
            net_s = tflearn.layers.normalization.batch_normalization(net_s, name="bn-{}-s".format(i+1))
            net_s = tflearn.activations.relu(net_s)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        net_a = action
        for i in range(self.layers-1):
            net_a = tflearn.fully_connected(net_a, self.hidden_size, name="fc-{}-a".format(i+1), bias=False, bias_init=bias_init)
            net_a = tflearn.layers.normalization.batch_normalization(net_a, name="bn-{}-a".format(i+1))
            net_a = tflearn.activations.relu(net_a)


        t1 = tflearn.fully_connected(net_s, self.hidden_size, name="fc-comb-s", bias=False, bias_init=bias_init)
        t2 = tflearn.fully_connected(net_a, self.hidden_size, bias=False, name="fc-comb-s")
        net = tflearn.layers.normalization.batch_normalization(t1+t2, name="bn-out-comb")
        net = tflearn.activation(net, activation='relu')

        out = tflearn.fully_connected(net, 1, name="fc-last", bias=False)
        return inputs, action, out

    def _create_optimizer(self):
        # global counter
        self._global_step = tf.Variable(0.0,trainable=False, name="global_step")

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)

        # define optimization Op
        if self.opt == "adam":
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.opt == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise ValueError("{} is not a valid optimizer".format(self.opt))

        grad_and_vars = self.optimizer.compute_gradients(self.loss, self.network_params)

        grads = [g for g,_ in grad_and_vars]
        grads, self.grad_norm = tf.clip_by_global_norm(grads, 0.5)
        # grad_and_vars = [(g,v) for g,v in zip(grads, self.network_params)]
        # grad_and_vars = [(g,v) for g,v in zip(grads, self.network_params)]
        self.optimize = self.optimizer.apply_gradients(grad_and_vars,
                                                       global_step=self._global_step)

        self.action_grads = tf.gradients(self.out , self.action)

        self._lr_decay = tf.assign(self.learning_rate, self.learning_rate * self.learning_rate_decay)

    ######################## computational methods ########################
    def train(self, inputs, action, predicted_q_value):
        _,loss, global_step, g_norm = self.sess.run([self.optimize, self.loss,self._global_step, self.grad_norm], feed_dict={
                            self.inputs: inputs,
                            self.action: action,
                            self.predicted_q_value: predicted_q_value
                        })
        return loss, global_step, g_norm

    def compute_loss(self, inputs, action, predicted_q_value):
        loss = self.sess.run(self.loss, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })
        return loss

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def copy_target_network(self):
        self.sess.run(self.copy_target_network_params)

    def lr_decay(self):
        self.sess.run(self._lr_decay)

    @property
    def lr(self):
        return self.sess.run(self.learning_rate)

    @property
    def global_step(self):
        return self.sess.run(self._global_step)



