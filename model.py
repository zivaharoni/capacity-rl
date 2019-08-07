from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import tensorflow as tf
import tflearn
import numpy as np
import os
logger = logging.getLogger("logger")

def act(z, name=None):
    b1 = tf.get_variable(name + 'b1',shape=z.shape[-1])
    # b2 = tf.get_variable(name + 'b2',shape=z.shape[-1])
    y = tf.nn.relu(z + b1) #* tf.sigmoid(-z + b1 - b2)
    return y

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

        self._create_summary_ops()

    ######################## init methods ########################
    def _set_hyper_parameters(self, sess, env, config, is_training):
        self.summary_dir = os.path.join(config.directory, "summaries")
        self.train_summary_op_list = list()
        self.sess = sess
        self.s_dim = env.state_cardin-1
        self.a_dim = env.input_cardin * env.state_cardin
        self.env = env

        self.learning_rate = tf.Variable(config.actor_lr, trainable=False)
        self.learning_rate_decay = config.lr_decay
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.hidden_size = config.actor_hid
        self.layers = config.actor_layers
        self.is_training = is_training
        self.opt = config.opt
        self.drop = config.actor_drop
        if self.drop > 0.0:
            raise Warning("Actor Network: dropout is not implemented")

    def _create_actor_network(self):
        last_trainable_var = len(tf.trainable_variables())
        self.inputs, self.out = self._create_network("online")
        self.network_params = tf.trainable_variables()[last_trainable_var:]

    def _create_target_network(self):
        last_trainable_var = len(tf.trainable_variables())
        self.target_inputs, self.target_out = self._create_network("target")
        self.target_network_params = tf.trainable_variables()[last_trainable_var:]

        # Op for periodically updating target network with online network
        # weights
        # self.update_target_network_params = \
        #     [self.target_network_params[i].assign(
        #         tf.multiply(self.target_network_params[i], tf.div(self._global_step, tf.add(self._global_step, 1.))) +
        #         tf.div(self.network_params[i], tf.add(self._global_step, 1.)))
        #     for i in range(len(self.target_network_params))]
        # self.update_target_network_params = \
        #     [self.target_network_params[i].assign(self.network_params[i])
        #     for i in range(len(self.target_network_params))]
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.target_network_params[i], 1 - self.tau) +
                tf.multiply(self.network_params[i], self.tau))
            for i in range(len(self.target_network_params))]

    def _create_network(self,name):
        bias_init = tflearn.initializations.uniform(shape=None, minval=0.1, maxval=0.2,dtype=tf.float32)
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = inputs
        for i in range(self.layers):
            net = tflearn.fully_connected(net, self.hidden_size, name="fc-{}".format(i+1), bias=True, bias_init=bias_init)
            net = tflearn.layers.normalization.batch_normalization(net, name="bn-{}".format(i+1))
            # net = tflearn.activations.relu(net)
            net = act(net, name="{}fc-{}".format(name, i+1) )
            # if self.is_training:
            #     net = tf.nn.dropout(net, 0.5, name="dropout-{}".format(i+1))

        bias_init = tflearn.initializations.uniform(shape=None, minval=-0.04, maxval=0.04,dtype=tf.float32)
        out = tflearn.fully_connected(
            net, self.a_dim, bias_init=bias_init)
        out = tflearn.layers.normalization.batch_normalization(out, name="bn-out")
        out = tf.reshape(out , [-1, self.env.input_cardin, self.env.state_cardin])
        out = tf.reshape(tf.nn.softmax(out, axis=1), [-1, self.a_dim])

        return inputs, out

    def _create_optimizer(self):
        # global counter
        self._global_step = tf.Variable(0.0, trainable=False, name="global_step")
        self._global_inc = self._global_step.assign_add(1.0)

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient)
        # self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.actor_gradients))

        # policy_entropy = -tf.constant(,uu,r gk vtjrubvfor g1,g2 in zip(self.actor_gradients,entropy_gradients)]


        # Optimization Op
        if self.opt == "adam":
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.opt == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise ValueError("{} is not a valid optimizer".format(self.opt))

        self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))

        # learning rate decay
        self._lr_decay = tf.assign(self.learning_rate, self.learning_rate * self.learning_rate_decay)

        grad = tf.concat([tf.reshape(g, [-1]) for g in self.actor_gradients], axis=0)
        self.grad_norm = tf.log(tf.norm(grad))
        self.train_summary_op_list.append(tf.summary.scalar("grad_norm", self.grad_norm))

    def _create_summary_ops(self):

        self.summary_writer = tf.summary.FileWriter(self.summary_dir)
        if self.is_training:
            self.train_summary = tf.summary.merge(self.train_summary_op_list)

        self.avg_reward = tf.placeholder(tf.float32, shape=(), name="reward")
        self.avg_reward_target = tf.placeholder(tf.float32, shape=(), name="reward_target")

        ro = tf.summary.scalar("avg_reward", self.avg_reward)
        ro_target = tf.summary.scalar("avg_reward_target", self.avg_reward_target)

        self._ro_summary = tf.summary.merge([ro, ro_target])
        self._ro_summary_step = tf.Variable(0.0, trainable=False, name="ro_summary_step")
        self._ro_summary_inc = self._ro_summary_step.assign_add(1.0)

    ######################## computational methods ########################
    def train(self, inputs, a_gradient):
        _,i,s, g = self.sess.run([self.optimize, self._global_step, self.train_summary, self.grad_norm], feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
        self.summary_writer.add_summary(s, i)
        return g

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run([self.update_target_network_params, self._global_inc])

    def lr(self):
        return self.sess.run(self.learning_rate)

    def lr_decay(self):
        self.sess.run(self._lr_decay)

    def global_step(self):
        return self.sess.run(self._global_step)

    def set_global_step(self, val):
        return self.sess.run(self._global_step.assign(val))

    def ro_summary(self, reward, reward_target):
        s, i = self.sess.run([self._ro_summary, self._ro_summary_inc], feed_dict={self.avg_reward: reward,
                                                                                  self.avg_reward_target: reward_target})
        self.summary_writer.add_summary(s, i)


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

        self._create_summary_ops()


    ######################## init methods ########################
    def _set_hyper_parameters(self, sess, env, config, is_training):
        self.summary_dir = os.path.join(config.directory, "summaries")
        self.train_summary_op_list = list()
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
        self.drop = config.critic_drop
        if self.drop > 0.0:
            raise Warning("Critic Network: dropout is not implemented")

    def _create_critic_network(self):
        last_trainable_var = len(tf.trainable_variables())
        self.inputs, self.action, self.out = self._create_network("online")
        self.network_params = tf.trainable_variables()[last_trainable_var:]

    def _create_target_network(self):
        last_trainable_var = len(tf.trainable_variables())
        self.target_inputs, self.target_action, self.target_out = self._create_network("target")
        self.target_network_params = tf.trainable_variables()[last_trainable_var:]

        # self.update_target_network_params = \
        #     [self.target_network_params[i].assign(
        #         tf.multiply(self.target_network_params[i], tf.div(self._global_step, tf.add(self._global_step, 1.))) +
        #         tf.div(self.network_params[i], tf.add(self._global_step, 1.)))
        #     for i in range(len(self.target_network_params))]
        # self.update_target_network_params = \
        #     [self.target_network_params[i].assign(self.network_params[i])
        #     for i in range(len(self.target_network_params))]
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.target_network_params[i], 1 - self.tau) +
                tf.multiply(self.network_params[i], self.tau))
            for i in range(len(self.target_network_params))]

    def _create_network(self,name):
        bias_init = tflearn.initializations.uniform(shape=None, minval=0.0, maxval=0.1,dtype=tf.float32)

        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])

        net_s = inputs
        for i in range(self.layers):
            net_s = tflearn.fully_connected(net_s, self.hidden_size, name="fc-{}-s".format(i+1), bias=False, bias_init=bias_init)
            net_s = tflearn.layers.normalization.batch_normalization(net_s, name="bn-{}-s".format(i+1))
            # net_s = tflearn.activations.relu(net_s)
            net_s = act(net_s, name="{}fc-{}-s".format(name,i+1))
            # if self.is_training:
            #     net_s = tf.nn.dropout(net_s, 0.5, name="dropout-{}".format(i+1))

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases

        net_a = action
        for i in range(self.layers-1):
            net_a = tflearn.fully_connected(net_a, self.hidden_size, name="fc-{}-a".format(i+1), bias=False, bias_init=bias_init)
            net_a = tflearn.layers.normalization.batch_normalization(net_a, name="bn-{}-a".format(i+1))
            # net_a = tflearn.activations.relu(net_a)
            net_a = act(net_a, name="{}fc-{}-a".format(name,i+1))
            # if self.is_training:
            #     net_a = tf.nn.dropout(net_a, 0.5, name="dropout-{}".format(i+1))


        t1 = tflearn.fully_connected(net_s, self.hidden_size, name="fc-comb-s", bias=False, bias_init=bias_init)
        t2 = tflearn.fully_connected(net_a, self.hidden_size, bias=False, name="fc-comb-s")
        net = tflearn.layers.normalization.batch_normalization(t1+t2, name="bn-out-comb")
        # net = tflearn.activation(t1+t2, activation='relu')
        net = act(net, name="{}fc-out-comb".format(name))

        out = tflearn.fully_connected(net, 1, name="fc-last", bias=False)
        out = out / tf.norm(out.W)
        return inputs, action, out

    def _create_optimizer(self):
        # global counter
        self._global_step = tf.Variable(0.0,trainable=False, name="global_step")
        self._global_inc = self._global_step.assign_add(1.0)

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


        self.optimize = self.optimizer.apply_gradients(grad_and_vars)

        self.action_grads = tf.gradients(self.out , self.action)

        self._lr_decay = tf.assign(self.learning_rate, self.learning_rate * self.learning_rate_decay)


        grad_vec = tf.concat([tf.reshape(g, [-1]) for (g,v)  in grad_and_vars], axis=0)
        self.grad_norm = tf.log(tf.norm(grad_vec))
        self.train_summary_op_list.append(tf.summary.scalar("grad_norm", self.grad_norm))

    def _create_summary_ops(self):
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)
        if self.is_training:
            self.train_summary = tf.summary.merge(self.train_summary_op_list)

    ######################## computational methods ########################
    def train(self, inputs, action, predicted_q_value):
        _, i, s, g = self.sess.run([self.optimize, self._global_step, self.train_summary, self.grad_norm], feed_dict={
                            self.inputs: inputs,
                            self.action: action,
                            self.predicted_q_value: predicted_q_value
                        })
        self.summary_writer.add_summary(s, i)
        return g

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
        self.sess.run([self.update_target_network_params, self._global_inc])

    def lr(self):
        return self.sess.run(self.learning_rate)

    def lr_decay(self):
        self.sess.run(self._lr_decay)

    def global_step(self):
        return self.sess.run(self._global_step)

    def set_global_step(self, val):
        return self.sess.run(self._global_step.assign(val))


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, sigma_dec=1., theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self._sigma_dec = sigma_dec
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def sigma_dec(self):
        self.sigma *= self._sigma_dec

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise'