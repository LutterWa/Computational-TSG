import tensorflow.compat.v1 as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
A_UPDATE_STEPS = 10  # actor update steps
C_UPDATE_STEPS = 10  # critic update steps
METHOD = [dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
          dict(name='clip', epsilon=0.2)  # Clipped surrogate objective
          ][1]


class PPO(object):
    def __init__(self, state_dim, action_dim, action_bound, load_path=None):
        self.load_flag = load_path
        graph = tf.Graph()
        with graph.as_default():  # 构建计算图
            self.state = tf.placeholder(tf.float32, [None, state_dim], 'state')
            # critic
            with tf.name_scope('critic'):
                l1 = tf.layers.dense(self.state, 64, tf.nn.relu)
                l1 = tf.layers.dense(l1, 64, tf.nn.relu)
                self.v = tf.layers.dense(l1, 1)
                with tf.name_scope('train_critic'):
                    # Update actor network
                    self.discounted_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')  # 折扣回报函数
                    self.advantage = self.discounted_r - self.v
                    self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
                    self.critic_train_op = tf.train.AdamOptimizer(C_LR).minimize(self.critic_loss)

            # actor
            with tf.name_scope('actor'):
                def _build_anet(state, name, trainable):
                    with tf.variable_scope(name):
                        l1 = tf.layers.dense(state, 64, tf.nn.relu, trainable=trainable)
                        l1 = tf.layers.dense(l1, 64, tf.nn.relu, trainable=trainable)
                        mean = action_bound * tf.layers.dense(l1, action_dim, tf.nn.tanh, trainable=trainable)
                        logstd = tf.Variable(np.zeros(action_dim, dtype=np.float32), trainable=trainable)
                        norm_dist = tf.distributions.Normal(loc=mean, scale=tf.exp(logstd))  # 正态分布
                        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)  # 提取图中的元素
                    return norm_dist, params

                pi, pi_params = _build_anet(self.state, 'pi', trainable=True)  # 新策略
                old_pi, old_pi_params = _build_anet(self.state, 'oldpi', trainable=False)  # 旧策略

                # 预先定义两个操作
                with tf.name_scope('update_oldpi'):  # 将pi网络参数赋给oldpi网络
                    self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, old_pi_params)]
                with tf.name_scope('action'):  # 执行动作
                    self.act_op = pi.loc[0]  # 直接选择均值作为动作
                    self.sample_op = tf.squeeze(pi.sample(1), axis=0)[0]  # 从正态分布中随机选择一个动作

                with tf.name_scope('train_critic'):
                    self.action = tf.placeholder(tf.float32, [None, action_dim], 'action')  # 创建action占位符
                    self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')  # 创建A(s,a)占位符
                    ratio = tf.exp(pi.log_prob(self.action) - old_pi.log_prob(self.action))  # 比率函数r(θ)
                    surr = ratio * self.adv  # 目标函数的第一项

                    if METHOD['name'] == 'kl_pen':  # 第一类目标函数(KL散度型)
                        self.tflam = tf.placeholder(tf.float32, None,
                                                    'lambda')  # 新加入的参数，Adaptive KL Penalty Coefficient
                        kl = tf.distributions.kl_divergence(old_pi, pi)  # 计算KL散度，衡量两个分布的差异
                        self.kl_mean = tf.reduce_mean(kl)  # d参数,KL散度的均值
                        self.action_loss = -tf.reduce_mean(surr - self.tflam * kl)  # 目标函数

                    else:  # 第二类目标函数(CLIP型)
                        self.action_loss = -tf.reduce_mean(
                            tf.minimum(surr,
                                       tf.clip_by_value(ratio, 1. - METHOD['epsilon'],
                                                        1. + METHOD['epsilon']) * self.adv)
                        )  # 目标函数
                    self.action_train_op = tf.train.AdamOptimizer(A_LR).minimize(self.action_loss)  # 使用Adam优化目标函数

        self.sess = tf.Session(graph=graph)

        # 加载模型
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if load_path is None:
                    self.sess.run(tf.global_variables_initializer())  # 初始化计算图
                else:
                    saver = tf.train.Saver()
                    saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir=load_path))

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        self.action_bound = action_bound

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state, done, GAMMA=0.9):
        """
        Calculate cumulative reward
        :param next_state:
        :param done:
        :param GAMMA:
        :return: None
        """
        if done:
            v_s_ = 0
        else:
            v_s_ = self.get_v(np.array(next_state, np.float32))
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()

    def update(self):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.sess.run(self.update_oldpi_op)  # oldpi=pi
                adv = self.sess.run(self.advantage, {self.state: s, self.discounted_r: r})  # 通过critic得到advantage value

                # update actor
                if METHOD['name'] == 'kl_pen':  # KL散度型
                    for _ in range(A_UPDATE_STEPS):  #
                        _, kl = self.sess.run([self.action_train_op, self.kl_mean],
                                              {self.state: s, self.action: a, self.adv: adv, self.tflam: METHOD['lam']})
                        if kl > 4 * METHOD['kl_target']:  # 根据d和d目标的均值动态调整Adaptive KL Penalty Coefficient
                            break
                        elif kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                            METHOD['lam'] /= 2  # 放松KL约束限制
                        elif kl > METHOD['kl_target'] * 1.5:
                            METHOD['lam'] *= 2  # 增强KL约束限制
                        METHOD['lam'] = np.clip(METHOD['lam'], 1e-4,
                                                10)  # sometimes explode, this clipping is my solution
                else:  # clipping method, find this is better (OpenAI's paper)
                    [self.sess.run(self.action_train_op, {self.state: s, self.action: a, self.adv: adv}) for _ in
                     range(A_UPDATE_STEPS)]

                # update critic
                [self.sess.run(self.critic_train_op, {self.state: s, self.discounted_r: r}) for _ in
                 range(C_UPDATE_STEPS)]

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def get_action(self, s, greedy=False):
        s = s[np.newaxis, :]
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if greedy:
                    action = self.sess.run(self.act_op, {self.state: s})
                else:
                    action = self.sess.run(self.sample_op, {self.state: s})
        return np.clip(action, -self.action_bound, self.action_bound)

    def get_v(self, s):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if s.ndim < 2:
                    s = s[np.newaxis, :]
                return self.sess.run(self.v, {self.state: s})[0, 0]

    def save_model(self, save_path):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.load_flag is None:  # 未加载模型
                    saver = tf.train.Saver()
                    saver.save(self.sess, save_path=save_path)
                    # print("the model in {} save success".format(save_path))
