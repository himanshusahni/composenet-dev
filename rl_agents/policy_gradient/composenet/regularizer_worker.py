"""
@author: himanshusahni
Modified from a3c code by dennybritz

Worker thread for a3c training.
"""

import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import random
from collections import deque
import logging
from copy import copy

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

from utils import make_copy_params_op

def key_func(var):
  """
  key function for sorting in the graph structures that ensures weights are
  copied and gradients are applied correctly.
  here, only regularizer scopes are ignored
  Args:
     var: tensorflow variable
  """
  if 'regularizer' in var.name.split('/')[0]:
   return '/'.join(var.name.split('/')[1:])
  else:
   return var.name

class RegularizerPolicy(object):
  """
  Converts an embedding into a policy, i.e. a distribution over actions
  Without the hooks for optimization
  Args:
    name_scope: Prepended to scope of loss related variables and policy layer
      for worker threads
    num_outputs: number of actions
    embedder: embedding object for task
  """

  def __init__(self, name_scope, num_outputs, embedder):
    # assuming batch size is same for everything
    self.batch_size = embedder.batch_size
    # collect all the state input placeholders
    self.states = embedder.states
    self.num_outputs = num_outputs

    with tf.variable_scope(name_scope, tf.AUTO_REUSE):
      with tf.variable_scope("policy_net", reuse=tf.AUTO_REUSE):
        self.logits = tf.contrib.layers.fully_connected(
          embedder.embedding, num_outputs, activation_fn=None)
        self.probs = tf.nn.softmax(self.logits) + 1e-8
        self.predictions = {
          "logits": self.logits,
          "probs": self.probs
        }

class RegularizerWorker(object):
  """
  A regularizer worker. Samples a random environment, runs an episode on it to collect
  experience and runs regularizer on embeddings.
  Args:
    name: A unique name for this worker.
    envs: All the environments.
    policy_nets: All the globally shared policy networks.
    value_nets: All the globally shared value networks.
  """
  def __init__(
      self, name, envs, policy_nets, value_nets, global_counter, logfile, global_scopes):
    self.name = name
    self.discount_factor = 0.99
    self.global_step = tf.contrib.framework.get_global_step()
    self.global_counter = global_counter
    self.envs = envs
    self.num_skills = len(envs)
    self.global_scopes = global_scopes
    self.crashes = 0
    self.state_buffer = deque(maxlen=10000)
    # logging
    self.logger = logging.getLogger('regularizer')
    hdlr = logging.FileHandler(logfile)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    hdlr.setFormatter(formatter)
    self.logger.addHandler(hdlr)
    self.logger.setLevel(logging.INFO)

    self.state = None

  def create_copy_ops(self):
    '''
    hook op global/local copy
    '''
    global_variables = []
    for gs in self.global_scopes:
      global_variables += tf.contrib.slim.get_variables(
        scope=gs,
        collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    local_variables = tf.contrib.slim.get_variables(
      scope=self.name+'/',
      collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    v1_list = list(sorted(global_variables, key=key_func))
    v2_list = list(sorted(local_variables, key=key_func))
    self.copy_params_op = []
    for v1, v2 in zip(v1_list, v2_list):
      op = v2.assign(v1)
      self.copy_params_op.append(op)

  def create_train_ops(self):
    '''
    hook up the local/global gradients
    '''
    # calculate loss on embeddings w.r.t all other embeddings
    self.loss = 0
    for i in range(self.num_skills):
      for j in range(i+1, self.num_skills):
        self.loss += -tf.losses.cosine_distance(
          tf.nn.l2_normalize(self.skill_embedder[i].embedding, 0),
          tf.nn.l2_normalize(self.skill_embedder[j].embedding, 0), axis=0)
    self.loss /= (self.num_skills*self.num_skills)

    self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
    self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
    # only take gradients for the trainable layers
    self.grads_and_vars = [[grad, var] \
      for grad, var in self.grads_and_vars \
      if grad is not None]

    local_grads, local_vars = zip(*list(sorted(
      self.grads_and_vars,
      key=lambda grad_and_var: key_func(grad_and_var[1]))))
    # Clip gradients
    local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
    global_vars = []
    for i in range(self.num_skills):
      global_vars += tf.contrib.slim.get_variables(
        scope="global_{}/shared/".format(i),
        collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    global_vars = list(sorted(global_vars, key=key_func))
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    self.train_ops = self.optimizer.apply_gradients(local_global_grads_and_vars,
      global_step=self.global_step)

  def run(self, sess, coord, t_max):
    with sess.as_default(), sess.graph.as_default():
      # pick an initial environment
      self.env_num = random.randint(0, self.num_skills-1)
      self.env = self.envs[self.env_num]
      # Initial state
      self.state = self.env.reset()
      self.ep_r = 0
      self.ep_s = 0
      self.last_logged = 0
      try:
        while not coord.should_stop():
          # Copy Parameters from the global networks
          sess.run(self.copy_params_op)

          # Collect some experience
          self.run_n_steps(t_max, sess)

          # Update the global networks
          if len(self.state_buffer) > 100:
            loss = self.update(sess)
            gc = copy(self.global_counter)
            gc = next(gc)
            if gc > self.last_logged + 500:
                self.last_logged = gc
                self.logger.info('Global Step : {}, environment : regularizer, loss = {}'\
                  .format(gc, loss))

      except tf.errors.CancelledError:
        return

  def _policy_net_predict(self, state, sess, n):
    # feed in the state to all embedders
    feed_dict = {}
    for i in range(len(self.policy_nets[n].states)):
      feed_dict[self.policy_nets[n].states[i]] = [state]
    preds = sess.run(self.policy_nets[n].predictions, feed_dict)
    return preds["probs"][0]

  def run_n_steps(self, n, sess):
    for _ in range(n):
      # Store state
      self.state_buffer.append(self.state)
      # Take a step
      action_probs = self._policy_net_predict(self.state, sess, self.env_num)
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      try:
        next_state, reward, done = self.env.step(action)
      except ValueError as ex:
        self.crashes += 1
        print "thread {} crashed, total {} times, on step {} with message {}"\
          .format(
          self.name,
          self.crashes,
          self.env.steps,
          str(ex))
        # the solution in this case is to just start a new episode
        self.env_num = random.randint(0, self.num_skills-1)
        self.env = self.envs[self.env_num]
        self.state = self.env.reset()
        self.ep_r = 0
        self.ep_s = 0
        break

      # Increase local and global counters
      self.ep_r += reward
      self.ep_s += 1

      if done:
        # sys.stderr.write("Thread {}, task {}, Task steps {}, Global steps {}, episode reward {}, episode steps {}\n".format(self.name, self.task_id, task_t, global_t, self.ep_r, self.ep_s))
        self.ep_r = 0
        self.ep_s = 0
        self.env_num = random.randint(0, self.num_skills-1)
        self.env = self.envs[self.env_num]
        self.state = self.env.reset()
        break
      else:
        self.state = next_state

  def update(self, sess):
    """
    Updates global policy and value networks based on collected experience
    """

    # Accumulate minibatch exmaples
    states = random.sample(self.state_buffer, 32)

    feed_dict = {}
    for embedder in self.skill_embedder:
      for i in range(len(embedder.states)):
        feed_dict[embedder.states[i]] = np.array(states)

    # Train the global estimators using local gradients
    regularization_loss, _ = sess.run(
      [self.loss, self.train_ops],
      feed_dict)

    return regularization_loss
