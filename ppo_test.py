import random
import math
import numpy as np
import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from scipy.io import savemat
from missile_3d import MISSILE
from ppo_model import PPO
from dnn_test import DeepNeuralNetwork

BATCH_SIZE = 32  # update batch size
GAMMA = 0.99
TEST_EPISODES = 100
REWARD_SAVE_CASE = 0
dnny = DeepNeuralNetwork()  # 纵向通道
dnnz = keras.models.load_model("./dnn_model/flight.h5")  # 侧向通道
pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度


class OMissile(MISSILE):
    def __init__(self):
        super().__init__()
        v, theta, psi, r, q_theta, q_psi, x, y, z, t = self.collect()
        self.nav_theta = 0.
        self.nav_psiv = 0.
        tmp_y = np.array([[v * math.cos(self.nav_psiv) / 1e2, theta / -1, r / 1e4, q_theta / -1, x / 1e4, y / 1e4]])
        tmp_z = np.array([[v * math.cos(self.nav_theta) / 1e2, psi / -1, r / 1e4, -q_psi / -1, x / 1e4, z / 1e4]])
        self.iay = float(dnny.predict(tmp_y))
        self.iaz = float(dnnz.predict(tmp_z))
        self.tgo = r / v

    def get_ia(self):
        v, theta, psi, r, q_theta, q_psi, x, y, z, t = self.collect()
        # tmp_y = np.array([[v / 1e2, theta / -1, r / 1e4, q_theta / -1, x / 1e4, y / 1e4]])
        # tmp_z = np.array([[v / 1e2, psi / -1, r / 1e4, -q_psi / -1, x / 1e4, z / 1e4]])
        tmp_y = np.array([[v * math.cos(self.nav_psiv) / 1e2, theta / -1, r / 1e4, q_theta / -1, x / 1e4, y / 1e4]])
        tmp_z = np.array([[v * math.cos(self.nav_theta) / 1e2, psi / -1, r / 1e4, -q_psi / -1, x / 1e4, z / 1e4]])
        self.iay = float(dnny.predict(tmp_y))
        self.iaz = float(dnnz.predict(tmp_z))
        return self.iay, self.iaz

    def get_tgo(self):
        v, theta, psi, r, q_theta, q_psi, x, y, z, t = self.collect()
        self.tgo = r / v
        return self.tgo

    def get_state(self, y_target, z_target):
        iay, iaz = self.get_ia()
        tgo = self.get_tgo()
        state_y = np.array([(y_target - iay) / tgo])
        state_z = np.array([(z_target - iaz) / tgo])
        return state_y, state_z


if __name__ == '__main__':
    env = OMissile()
    # set the init parameter
    state_dim = 1
    action_dim = 1
    action_bound = 2 * 9.81  # action limitation

    # test
    agent_y = PPO(state_dim, action_dim, action_bound, r'./ppo_model')
    agent_z = PPO(state_dim, action_dim, action_bound, r'./ppo_model')
    flight_data = {}
    for episode in range(TEST_EPISODES):
        desired_iay = []  # 期望的ia
        actual_iay = []  # 实际的ia
        impact_angle_errory = []  # ia的误差
        desired_iaz = []  # 期望的ia
        actual_iaz = []  # 实际的ia
        impact_angle_errorz = []  # ia的误差

        env.modify([200., 0. / RAD, 0. / RAD, -20000., 10000., 5000.])  # [200., 0. / RAD, 0. / RAD, -20000., 10000., 5000.]
        tdy = random.uniform(-60., -20.)
        tdz = random.uniform(-20., 20.)

        # expre = 1
        # tdy = [-20., -40., -60.][expre]
        # tdz = [20., 0., -20.][expre]

        t = []
        state_y, state_z = env.get_state(tdy, tdz)
        done = False
        TSG = False
        while done is False:
            if TSG:
                done, env.nav_theta, env.nav_psiv = env.step(tdy / RAD, tdz / RAD, h=0.01)  # TSG
            else:
                action_y = agent_y.get_action(state_y, greedy=True)  # use the mean of distribution as action
                action_z = agent_z.get_action(state_z, greedy=True)  # use the mean of distribution as action
                done, env.nav_theta, env.nav_psiv = env.step(action_y, -action_z, h=0.01)
            state_y, state_z = env.get_state(tdy, tdz)

            desired_iay.append(tdy)
            actual_iay.append(env.iay)
            impact_angle_errory.append(tdy - env.iay)

            desired_iaz.append(tdz)
            actual_iaz.append(env.iaz)
            impact_angle_errorz.append(tdz - env.iaz)

        # tmpenergy = 0
        # for e in range(len(env.reac)):
        #     tmpenergy = tmpenergy + env.reac[e] ** 2 * 0.1
        # print(tmpenergy)
        # env.plot_data()
        # plt.ioff()
        # plt.show()

        flight_data['sim_{}'.format(episode)] = env.save_data()
        flight_data['time{}'.format(episode)] = {'desired_iay': np.array(desired_iay),
                                                 'actual_iay': np.array(actual_iay),
                                                 'impact_time_errory': np.array(impact_angle_errory),
                                                 'desired_iaz': np.array(desired_iaz),
                                                 'actual_iaz': np.array(actual_iaz),
                                                 'impact_time_errorz': np.array(impact_angle_errorz)}

        # print the result
        print('Testing | Episode: {}/{} | Target: {:.2f}/{:.2f} | Actual: {:.2f}/{:.2f} | Error: {:.2f}/{:.2f}'
              .format(episode + 1, TEST_EPISODES,
                      tdy, tdz,
                      env.Y[2] * RAD, env.Y[3] * RAD,
                      tdy - env.Y[2] * RAD, tdz - env.Y[3] * RAD))
        # plt.figure(3)
        # plt.ion()
        # plt.clf()
        # plt.subplots_adjust(hspace=0.6)
        # plt.subplot(2, 1, 1)
        # plt.plot(np.array(env.reY)[:, 0], np.array(desired_iay)[:-1], 'k--', label='desired ia')
        # plt.plot(np.array(env.reY)[:, 0], np.array(actual_iay)[:-1], 'k-', label='predict ia')
        # plt.plot(np.array(env.reY)[:, 0], np.array(env.reY)[:, 2] * RAD, 'r-', label='actual theta')
        # plt.xlabel('Time (s)')
        # plt.ylabel('impact angle of xoy')
        # plt.legend()
        # plt.grid()
        #
        # plt.subplot(2, 1, 2)
        # plt.plot()
        # plt.plot(np.array(env.reY)[:, 0], np.array(desired_iaz)[:-1], 'k--', label='desired ia')
        # plt.plot(np.array(env.reY)[:, 0], np.array(actual_iaz)[:-1], 'k-', label='predict ia')
        # plt.plot(np.array(env.reY)[:, 0], np.array(env.reY)[:, 3] * RAD, 'r-', label='actual psi')
        # plt.xlabel('Time (s)')
        # plt.ylabel('impact angle of xoz')
        # plt.grid()
        #
        # plt.pause(0.1)
    savemat('./mats/flight_sim_ppo_monte_3d.mat', flight_data)
    # savemat('./mats/flight_sim_tsg_{}.mat'.format(3), flight_data)
