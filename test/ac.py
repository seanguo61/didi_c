import numpy as np
import tensorflow as tf
import random, os
from copy import deepcopy

from tensorflow.contrib.learn.python.learn import trainable


class Estimator:
    """ build value network
    """

    def __init__(self,
                 sess,
                 action_dim,
                 state_dim,
                 env,
                 scope="estimator",
                 summaries_dir=None):
        self.sess = sess
        self.n_valid_grid = env.n_valid_grids
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.M = env.M
        self.N = env.N
        self.scope = scope
        self.T = 288
        self.env = env

        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):

            # Build the value function graph
            # with tf.variable_scope("value"):
            value_loss = self._build_value_model()

            with tf.variable_scope("policy"):
                actor_loss, entropy = self._build_mlp_policy()

            self.loss = actor_loss + .5 * value_loss - 10 * entropy

            # self.loss_gradients = tf.gradients(self.value_loss, tf.trainable_variables(scope=scope))
            # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("value_loss", self.value_loss),
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),
            # tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))
        ])

        self.policy_summaries = tf.summary.merge([
            tf.summary.scalar("policy_loss", self.policy_loss),
            tf.summary.scalar("adv", tf.reduce_mean(self.tfadv)),
            tf.summary.scalar("entropy", self.entropy),
            # tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))
        ])

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_value_model(self):

        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        # The TD target value
        self.y_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")

        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")

        # 3 layers feed forward network.

        l1 = tf.layers.dense(X, 16, tf.nn.relu, trainable=trainable)
        l2 = tf.layers.dense(l1, 8, tf.nn.relu, trainable=trainable)

        self.value_output = tf.layers.dense(l2, 1, tf.nn.relu, trainable=trainable)

        # self.losses = tf.square(self.y_pl - self.value_output)
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.value_output))

        self.value_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.value_loss)

        return self.value_loss

    def _build_mlp_policy(self):

        self.policy_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="P")
        self.ACTION = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="action")
        self.tfadv = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantage')

        l1 = tf.layers.dense(self.policy_state, 16, tf.nn.relu, trainable=trainable)
        l2 = tf.layers.dense(l1, 8, tf.nn.relu, trainable=trainable)

        self.logits = logits = tf.layers.dense(l2, "logits", self.action_dim, act=tf.nn.relu) + 1  # avoid valid_logits are all zeros
        self.valid_logits = logits * self.neighbor_mask

        self.softmaxprob = tf.nn.softmax(tf.log(self.valid_logits + 1e-8))
        self.logsoftmaxprob = tf.nn.log_softmax(self.softmaxprob)

        self.neglogprob = - self.logsoftmaxprob * self.ACTION
        self.actor_loss = tf.reduce_mean(tf.reduce_sum(self.neglogprob * self.tfadv, axis=1))
        self.entropy = - tf.reduce_mean(self.softmaxprob * self.logsoftmaxprob)

        self.policy_loss = self.actor_loss - 0.01 * self.entropy

        self.policy_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.policy_loss)
        return self.actor_loss, self.entropy

    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, gamma):
        """for policy network"""
        advantage = []
        node_reward = node_reward.flatten()
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()

        for idx, next_state_id in enumerate(next_state_ids):
            next_state_id = int(next_state_id)
            temp_adv = node_reward[next_state_id] + gamma * qvalue_next[next_state_id] - curr_state_value[idx]
            advantage.append(temp_adv)
        return advantage

    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        targets = []
        node_reward = node_reward.flatten()
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()

        for idx in np.arange(self.n_valid_grid):
            grid_prob = valid_prob[idx][self.valid_action_mask[idx] > 0]
            neighbor_grid_ids = self.neighbors_list[idx]
            curr_grid_target = np.sum(
                grid_prob * (node_reward[neighbor_grid_ids] + gamma * qvalue_next[neighbor_grid_ids]))
            # assert np.sum(grid_prob) == 1 numerical issue.
            targets.append(curr_grid_target)

        return np.array(targets).reshape([-1, 1])

    def update_policy(self, policy_state, advantage, action_choosen_mat, curr_neighbor_mask, learning_rate,
                      global_step):
        sess = self.sess
        feed_dict = {self.policy_state: policy_state,
                     self.tfadv: advantage,
                     self.ACTION: action_choosen_mat,
                     self.neighbor_mask: curr_neighbor_mask,
                     self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.policy_summaries, self.policy_train_op, self.policy_loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss

    def update_value(self, s, y, learning_rate, global_step):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, state_dim]
          a: Chosen actions of shape [batch_size, action_dim], 0, 1 mask
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.summaries, self.value_train_op, self.value_loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss


class policyReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        # self.next_states = []
        self.neighbor_mask = []
        self.actions = []
        self.rewards = []  # advantages

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r, mask):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            # random.seed(0)
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.neighbor_mask[index:(index + new_sample_lens)] = mask

    def sample(self):

        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask]
        # random.seed(0)
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.neighbor_mask[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.neighbor_mask = []
        self.curr_lens = 0


class ReplayMemory:
    """ collect the experience and sample a batch for training networks.
        without time ordering
    """

    def __init__(self, memory_size, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0  # current memory lens

    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            # random.seed(0)
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s

    def sample(self):

        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states]
        # random.seed(0)
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.next_states[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.curr_lens = 0


class ModelParametersCopier:
    """
    Copy model parameters of one estimator to another.
    """

    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)
