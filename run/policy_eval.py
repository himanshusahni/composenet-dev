"""
@author: dennybritz

Modified for the ComposeNet project by Saurabh Kumar.
"""

import sys
import os
import shutil
import itertools
import numpy as np
import tensorflow as tf
import time
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from estimators import ValueEstimator, PolicyEstimator
from worker import make_copy_params_op


def log_results(eval_logger, iteration, rewards, steps):
  """Function that logs the reward statistics obtained by the agent.

  Args:
    logfile: File to log reward statistics.
    iteration: The current iteration.
    rewards: Array of rewards obtained in the current iteration.
  """
  eval_logger.info('Iteration : {}, mean reward = {}, mean step = {}, \
    all rewards = {}, all steps = {}'.format(iteration, np.mean(rewards),
    np.mean(steps), ','.join([str(r) for r in rewards]),
    ','.join([str(s) for s in steps])))


class PolicyEval(object):
  """
  Helps evaluating a policy by running a fixed number of episodes in an environment,
  and logging summary statistics to a text file.
  Args:
    env: environment to run in
    policy_net: A policy estimator
  """
  def __init__(self, env, policy_net, saver=None, n_eval=50, logfile=None, checkpoint_path=None):

    self.env = env
    self.global_policy_net = policy_net
    self.saver = saver
    self.n_eval = n_eval
    self.checkpoint_path = checkpoint_path
    self.logger = logging.getLogger('eval runs')
    hdlr = logging.FileHandler(logfile)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    hdlr.setFormatter(formatter)
    self.logger.addHandler(hdlr)
    self.logger.setLevel(logging.INFO)

    self.crashes = 0

    # Local policy net
    with tf.variable_scope("policy_eval"):
      self.policy_net = PolicyEstimator(
        policy_net.num_outputs,
        state_dims=self.env.get_state_size(),
        channels=self.env.get_num_channels())

    # Op to copy params from global policy/value net parameters
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

  def _policy_net_predict(self, state, sess):
    feed_dict = { self.policy_net.states: [state] }
    preds = sess.run(self.policy_net.predictions, feed_dict)
    return preds["probs"][0]

  def eval(self, sess, n_eval):
    with sess.as_default(), sess.graph.as_default():
      # Copy params to local model
      global_step, _ = sess.run([tf.contrib.framework.get_global_step(), self.copy_params_op])

      eval_rewards = []
      episode_lengths = []

      i = 0
      while i < n_eval:
        # Run an episode
        done = False
        state = self.env.reset()
        total_reward = 0.0
        episode_length = 0
        # if os.path.exists('experiment_logs/eval_runs/minecraft/apples/{}/{}'.format(global_step,i)):
          # shutil.rmtree('experiment_logs/eval_runs/minecraft/apples/{}/{}'.format(global_step,i))
        # os.makedirs('experiment_logs/eval_runs/minecraft/apples/{}/{}'.format(global_step,i))
        # all_states = [state]
        # all_rewards = [None]
        # all_actions = [None]
        while not done:
          action_probs = self._policy_net_predict(state, sess)
          action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
          try:
            next_state, reward, done = self.env.step(action)
          except ValueError as ex:
            self.crashes += 1
            print "thread policy_eval crashed, total {} times, on step {} with message {}"\
              .format(
              self.crashes,
              self.env.steps,
              str(ex))
            # the solution in this case is to just start a new episode
            break
          total_reward += reward
          episode_length += 1
          state = next_state
          # all_states.append(next_state)
          # all_rewards.append(reward)
          # all_actions.append(action)

        # only add episode if successfully finished
        if done:
          eval_rewards.append(total_reward)
          episode_lengths.append(episode_length)
          i += 1

        # for s, state in enumerate(all_states):
          # fig, axarr = plt.subplots(1,3)
          # for j in range(state.shape[-1]):
              # axarr[j].imshow(state[:,:,j])
          # plt.suptitle('Action: {}, Reward: {}'.format(all_actions[s], all_rewards[s]))
          # fig.savefig('experiment_logs/eval_runs/minecraft/apples/{}/{}/{}.png'.format(global_step,i,s))
          # plt.close('all')

      log_results(self.logger, global_step, eval_rewards, episode_lengths)

      # if self.saver is not None:
        # tf.add_to_collection('policy_train_op', self.policy_net.train_op)
        # tf.add_to_collection('action_probs', self.policy_net.probs)
        # tf.add_to_collection('state', self.policy_net.states)

        # self.saver.save(sess, self.checkpoint_path, global_step=global_step)
      return global_step, eval_rewards, episode_lengths

  def continuous_eval(self, eval_every, sess, coord):
    """
    Continuously evaluates the policy every [eval_every] seconds.
    """
    c = 0
    try:
      while not coord.should_stop():
        global_step, rewards, episode_lengths = self.eval(sess, self.n_eval)
        # Sleep until next evaluation cycle
        time.sleep(eval_every)
    except tf.errors.CancelledError:
      return
