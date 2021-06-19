import gym
import numpy as np
import tensorflow as tf
# from agent2.memory2 import *
import math
from memory import *


initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.3),
    'bias_initializer': tf.constant_initializer(0.1)
}


class DQN2(object):
    def __init__(self,gra, s_dim, a_dim, batch_size, gamma, lr, epsilon, replace_target_iter):
        with gra.as_default():
            self.s_dim = s_dim  # 状态维度
            self.a_dim = a_dim  # one hot行为维度
            self.gamma = gamma
            self.lr = lr  # learning rate
            self.epsilon = epsilon  # epsilon-greedy
            self.replace_target_iter = replace_target_iter  # 经历C步后更新target参数

            self.memory = Memory(batch_size, 10000)
            self._learn_step_counter = 0
            self._generate_model()

            self.saver = tf.train.Saver()

            self.sess = tf.Session()
            tf.global_variables_initializer().run(session=self.sess)
        assert self.sess.graph is gra

    def saver_net(self):
        # saver = tf.train.Saver()
        self.saver.save(self.sess, "data/agent2")
        # print('already')

    def restore_net(self):
        # saver = tf.train.Saver()
        self.saver.restore(self.sess, 'agent2/data/agent2')

    def choose_action(self, s):
        self.epsilon = self.epsilon * 0.98

        if np.random.rand() < self.epsilon:
            return [np.random.randint(self.a_dim),np.random.randint(self.a_dim)]
        else:
            q_eval_z_a1 = self.sess.run(self.q_eval_z_a1, feed_dict={
                self.s: s[np.newaxis, :]
            })
            q_eval_z_a2 = self.sess.run(self.q_eval_z_a2, feed_dict={
                self.s: s[np.newaxis, :]
            })
            # print(q_eval_z_a1.squeeze().argmax(), q_eval_z_a2.squeeze().argmax())
            return [q_eval_z_a1.squeeze().argmax(), q_eval_z_a2.squeeze().argmax()]

    def real_choose_action(self, s):
        q_eval_z_a1 = self.sess.run(self.q_eval_z_a1, feed_dict={
            self.s: s[np.newaxis, :]
        })
        q_eval_z_a2 = self.sess.run(self.q_eval_z_a2, feed_dict={
            self.s: s[np.newaxis, :]
        })
        return q_eval_z_a1.squeeze().argmax(), q_eval_z_a2.squeeze().argmax()

    def turn_p(self, a, b):
        sump = tf.reduce_sum(a)

        sumq = tf.reduce_sum(b)

        for i in range(len(p)):
            p[i] = p[i] / sump
        for i in range(len(q)):
            q[i] = q[i] / sumq

        return p,q

    def calculate_v(self, a, b):
        c = a[0]
        d = b[0]
        v = 0
        p,q = self.turn_p(a,b)
        for i in range(len(p)):
            for j in range(len(q)):
                v += p[i]*q[j] *(c[i] + d[j])
        return v

    def _generate_model(self):
        self.s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s')
        self.a1 =
        self.a2 = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a2')

        self.r = tf.placeholder(tf.float32, shape=(None, 1), name='r')
        self.s_ = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_')
        self.done = tf.placeholder(tf.float32, shape=(None, 1), name='done')

        self.q_eval_z_a1, self.q_eval_z_a2 = self._build_net(self.s, 'eval_net', True)
        self.q_target_z_a1, self.q_target_z_a2 = self._build_net(self.s_, 'target_net', False)

        self.q_eval_z_v = self.calculate_v(self.q_eval_z_a1,self.q_eval_z_a2)
        self.q_target_z_v = self.calculate_v(self.q_target_z_a1, self.q_target_z_a2)

        # y = r + gamma * max(q^)
        q_target = self.r + self.gamma \
            * self.q_eval_z_v * (1 - self.done)

        q_eval = self.q_target_z_v

        self.loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        param_target = tf.global_variables(scope='target_net')
        param_eval = tf.global_variables(scope='eval_net')

        # 将eval网络参数复制给target网络
        self.target_replace_ops = [tf.assign(t, e) for t, e in zip(param_target, param_eval)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s, 20, activation=tf.nn.relu, trainable=trainable, ** initializer_helper)
            a1 = tf.layers.dense(l, self.a_dim, trainable=trainable, **initializer_helper)
            a2 = tf.layers.dense(l, self.a_dim, trainable=trainable, **initializer_helper)

        return a1,a2

    def store_transition_and_learn(self, s, a1, a2, r, s_, done):
        if self._learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_ops)

        # 将行为转换为one hot形式
        one_hot_action1 = np.zeros(self.a_dim)
        one_hot_action2 = np.zeros(self.a_dim)
        one_hot_action1[a1] = 1
        one_hot_action2[a2] = 1

        self.memory.store_transition(s, one_hot_action1, one_hot_action2, [r], s_, [done])
        self._learn()
        self._learn_step_counter += 1

    def _learn(self):
        s, a1, a2, r, s_, done = self.memory.get_mini_batches()

        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={
            self.s: s,
            self.a1: a1,
            self.a2: a2,
            self.r: r,
            self.s_: s_,
            self.done: done
        })

if __name__ == "__main__":
    g = tf.Graph()
    n = 2
    road_range = 35
    action_section = 1
    agent = DQN2(
            gra=g,
            s_dim=2 * n,
            a_dim=int(math.ceil(road_range/action_section)),
            batch_size=128,
            gamma=0.99,
            lr=0.01,
            epsilon=0.1,
            replace_target_iter=300
        )
    agent._generate_model()