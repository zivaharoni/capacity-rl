from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import logging
import re
import numpy as np

logger = logging.getLogger("logger")

class SGD(object):
    def __init__(self, grads, tvars, lr, clip=None):

        self._updates = grads
        self._optimizer = tf.train.GradientDescentOptimizer(lr)

        if clip is not None:
            self._updates, _ = tf.clip_by_global_norm(self._updates, clip)

        self._train_op = self._optimizer.apply_gradients(
                zip(self._updates, tvars))


    @property
    def train_op(self):
        return self._train_op

    @property
    def updates(self):
        return self._updates

    @property
    def optimizer(self):
        return self._optimizer


class ASGD(object):

    def __init__(self, grads, tvars, lr, scope=""):

        sgd = SGD(grads, tvars, lr)

        self._updates = sgd.updates

        self._define_aux_vars(scope)

        self._train_op = list([sgd.train_op])

        with tf.name_scope("averaging"):
            self._accu_vars, self._save_vars, self._load_vars, self._final_assign_op = (list() for _ in range(4))

            for var in tvars:
                self._accu_vars.append(tf.get_variable(var.op.name + "_final",
                                                        initializer=tf.zeros_like(var, dtype=tf.float32), trainable=False))

                self._final_assign_op.append(tf.assign(var, self._accu_vars[-1] / self._global_step))

                tmp_var = (tf.get_variable(var.op.name + "tmp",
                                           initializer=tf.zeros_like(var, dtype=tf.float32), trainable=False))
                self._save_vars.append(tf.assign(tmp_var, var))
                self._load_vars.append(tf.assign(var, tmp_var))

        with tf.name_scope("trigger_mux"):
            def trigger_on():
                with tf.name_scope("trigger_is_on"):
                    op = list()
                    op.append(tf.identity(self._trigger))
                    op.append(tf.identity(self._T))
                    for i, var in enumerate(tvars):
                        op.append(tf.assign_add(self._accu_vars[i], var))
                    op.append(tf.assign_add(self._global_step, 1.))
                return op

            def trigger_off():
                with tf.name_scope("trigger_is_off"):
                    op = list()
                    op.append(tf.identity(self._trigger))
                    op.append(tf.identity(self._T))
                    for i, var in enumerate(tvars):
                        op.append(tf.identity(self._accu_vars[i]))
                    op.append(tf.identity(self._global_step))
                return op

            with tf.control_dependencies(self._train_op):
                self._train_op.append(tf.cond(self._trigger, lambda: trigger_on(), lambda: trigger_off()))

    def _define_aux_vars(self, scope):
        ######## trigger #########
        self._trigger = tf.get_variable("ASGD_trigger_" + scope, initializer=tf.constant(False, dtype=tf.bool), trainable=False)
        self._set_trigger = tf.assign(self._trigger, True)

        ######## T #########
        self._T = tf.get_variable("T_" + scope, initializer=tf.constant(0, dtype=tf.int32), trainable=False)
        self._new_T = tf.placeholder(tf.int32, shape=[], name="new_T_" + scope)
        self._set_T = tf.assign(self._T, self._new_T)

        ######## global counter #########
        self._global_step = tf.get_variable(name="global_step_" + scope, initializer=0.0, trainable=False)

    @property
    def global_step(self):
        return self._global_step

    @property
    def train_op(self):
        return self._train_op

    @property
    def trigger(self):
        return self._trigger

    @property
    def final_assign_op(self):
        return self._final_assign_op

    @property
    def save_vars(self):
        return self._save_vars

    @property
    def load_vars(self):
        return self._load_vars

    @property
    def set_trigger(self):
        return self._set_trigger

    @property
    def T(self):
        return self._T

    @property
    def set_T(self):
        return self._set_T

    @property
    def new_T(self):
        return self._new_T