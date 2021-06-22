import random
import time
import math
import os
import numpy as np
import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from scipy.io import savemat
from OAgent_sim import OAgentSim
from ppo_model import PPO
from dnn_test import DeepNeuralNetwork

BATCH_SIZE = 32  # update batch size
TRAIN_EPISODES = 1000  # total number of episodes for training
TEST_EPISODES = 100  # total number of episodes for testing
GAMMA = 0.99
REWARD_SAVE_CASE = 0
# dnn = DeepNeuralNetwork()
dnn = keras.models.load_model("./dnn_model/flight.h5")
# tgo = DeepNeuralNetwork(4, 'tgo_model')
pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度


class OMissile(OAgentSim):
    def __init__(self, no_=0, missile=None, target=None):
        super().__init__(no_=0, missile=None, target=None)
        v, theta, r, q, x, y, t = self.collect()
        self.ia = float(dnn.predict(np.array([[v / 1e2, theta / -1, r / 1e4, q / -1, x / 1e4, y / 1e4]])))
        self.tgo = r / v  # float(tgo.predict([[v / 315, theta / -0.6, x / -1e4, y / 1.5e4]]))

    def get_ia(self):
        v, theta, r, q, x, y, t = self.collect()
        dnn_state = np.array([[v / 1e2, theta / -1, r / 1e4, q / -1, x / 1e4, y / 1e4]])
        self.ia = float(dnn.predict(dnn_state))
        return self.ia

    def get_tgo(self, dnn_state=None):
        v, theta, r, q, x, y, t = self.collect()
        # dnn_state = [v / 315, theta / -0.6, x / -1e4, y / 1.5e4]
        # self.tgo = float(tgo.predict([dnn_state]))
        self.tgo = r / v
        return self.tgo

    def get_state(self, a_target):
        ia = self.get_ia()
        tgo = self.get_tgo()
        state_local = [(a_target - ia) / tgo]
        return np.array(state_local)

    def get_reward(self, t_target):
        v, theta, r, q, x, y, t = self.collect()
        tgo = self.get_tgo()
        e_local = (t_target - self.ia) / tgo
        # vy = v * math.sin(theta)  # y向速度
        # zem = y / tgo + vy
        k1 = 0.8
        k2 = 0.2
        reward_local = k1 * math.exp(-(e_local / 1e0) ** 2) + \
                       k2 * math.exp(-(self.ac / 1e1) ** 2)
        return np.array(reward_local)


if __name__ == '__main__':
    env = OMissile()

    # set the init parameter
    state_dim = 1
    action_dim = 1
    action_bound = 2 * 9.81  # action limitation
    t0 = time.time()
    model_num = 0

    train = False  # choose train or test
    if train:
        agent = PPO(state_dim, action_dim, action_bound)
        dict_episode_reward = {'all_episode_reward': [], 'episode_reward': [], 'target_angle': [], 'actual_angle': []}
        dict_episode_angle = {'desired ia': [], 'actual ia': [], 'impact angle error': []}
        all_episode_reward = []
        for episode in range(int(TRAIN_EPISODES)):
            env.modify()  # 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量
            if episode % 10 == 0:
                td = random.choice([-10., -20., -30., -40., -50., -60., -70., -80., -90.]) + 50.
            desired_ia = []  # 期望的ia
            actual_ia = []  # 实际的ia
            impact_angle_error = []  # ia的误差
            state = env.get_state(td)
            episode_reward = 0
            done = False
            while done is False:
                # collect state, action and reward
                action = agent.get_action(state)  # get new action with old state
                done = env.step(action=float(action), h=0.1)
                state_ = env.get_state(td)  # get new state with new action
                reward = env.get_reward(td)  # get new reward with new action
                agent.store_transition(state, action, reward)  # train with old state
                state = state_  # update state
                episode_reward += reward

                desired_ia.append(td)
                actual_ia.append(env.ia)
                impact_angle_error.append(td - env.ia)

                # update ppo
                if len(agent.state_buffer) >= BATCH_SIZE:
                    agent.finish_path(state_, done, GAMMA=GAMMA)
                    agent.update()
            # end of one episode
            env.plot_data(figure_num=0)

            # use the terminal data to update once
            if len(agent.reward_buffer) != 0:
                agent.reward_buffer[-1] -= env.R + (td - env.Y[2] * RAD) ** 2
                agent.finish_path(state, done, GAMMA=GAMMA)
                agent.update()
            episode_reward -= env.R + (td - env.Y[2] * RAD) ** 2

            # print the result
            episode_reward = episode_reward / env.Y[0]  # calculate the average episode reward
            print('Training | Episode: {}/{} | Average Episode Reward: {:.2f} | Running Time: {:.2f} | '
                  'Target Angle: {:.2f} | Actual Angle: {:.2f} | Error Angle: {:.2f}'
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0,
                          td, env.Y[2] * RAD, td - env.Y[2] * RAD))

            #     plt.figure(1)
            #     plt.ion()
            #     plt.clf()
            #     plt.subplots_adjust(hspace=0.6)
            #     plt.subplot(2, 1, 1)
            #     plt.plot(np.array(env.reY)[:, 0], np.array(desired_ia)[:-1], 'k--', label='desired ia')
            #     plt.plot(np.array(env.reY)[:, 0], np.array(actual_ia)[:-1], 'k-', label='actual ia')
            #     plt.xlabel('Time (s)')
            #     plt.ylabel('angle(s)')
            #     plt.legend()
            #     plt.grid()
            #
            #     plt.subplot(2, 1, 2)
            #     plt.plot()
            #     plt.plot(np.array(env.reY)[:, 0], np.array(impact_angle_error)[:-1], 'k-')
            #     plt.xlabel('Time (s)')
            #     plt.ylabel('error(s)')
            #     plt.grid()
            #
            #     plt.pause(0.1)
            #
            # calculate the discounted episode reward
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * .9 + episode_reward * .1)

            # save the episode data
            dict_episode_reward['episode_reward'].append(episode_reward)
            dict_episode_reward['all_episode_reward'] = all_episode_reward
            dict_episode_reward['target_angle'].append(td)
            dict_episode_reward['actual_angle'].append(env.Y[2] * RAD)

            # save model and data
            if episode_reward > REWARD_SAVE_CASE:
                REWARD_SAVE_CASE = episode_reward
                # if abs(td - env.Y[0]) < 0.5:
                agent.save_model('./ppo_model/agent{}'.format(model_num))
                savemat('./ppo_reward.mat', dict_episode_reward)
                model_num = (model_num + 1) % 20

        agent.save_model('./ppo_model/agent_end')
        savemat('./ppo_reward.mat', dict_episode_reward)

        plt.figure(2)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join(['PPO', time.strftime("%Y_%m%d_%H%M")])))
        plt.ioff()
        plt.show()
    else:
        # test
        agent = PPO(state_dim, action_dim, action_bound, r'./ppo_model')
        flight_data = {}
        for episode in range(100):  # TEST_EPISODES
            env.modify()  # [0., 200., 0 / RAD, -20000., 20000., 200.]
            # td = env.get_ia() * random.uniform(0.5, 1.5)
            td = random.uniform(-10., -90.) + 50.
            # td = -60.0
            desired_ia = []  # 期望的ia
            actual_ia = []  # 实际的ia
            impact_angle_error = []  # ia的误差
            state = env.get_state(td)
            # action = 0
            episode_reward = 0
            done = False
            t = []
            PPO_flag = True
            while done is False:
                action = agent.get_action(state, greedy=True)  # use the mean of distribution as action
                if PPO_flag:
                    done = env.step(action=action, h=0.1)
                else:
                    done = env.step(action=td / RAD, h=0.1)
                state = env.get_state(td)
                reward = env.get_reward(td)
                episode_reward += reward

                desired_ia.append(td)
                actual_ia.append(env.ia)
                impact_angle_error.append(td - env.ia)

            # tmpenergy = 0
            # for e in range(len(env.reac)):
            #     tmpenergy = tmpenergy + env.reac[e] ** 2 * 0.1
            # print(tmpenergy)
            # env.plot_data(figure_num=0)
            # plt.ioff()
            # plt.show()

            flight_data['sim_{}'.format(episode)] = env.save_data()
            flight_data['time{}'.format(episode)] = {'desired_ia': np.array(desired_ia),
                                                     'actual_ia': np.array(actual_ia),
                                                     'impact_time_error': np.array(impact_angle_error)}

            # print the result
            episode_reward = episode_reward / env.Y[0]  # calculate the average episode reward
            print('Testing | Episode: {}/{} | Average Episode Reward: {:.2f} | Running Time: {:.2f} | '
                  'Target: {:.2f} | Actual: {:.2f} | Error: {:.2f}'
                  .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0,
                          td, env.Y[2] * RAD, td - env.Y[2] * RAD))  # actual_ia[-1]))
            plt.figure(1)
            plt.ion()
            plt.clf()
            plt.subplots_adjust(hspace=0.6)
            plt.subplot(2, 1, 1)
            plt.plot(np.array(env.reY)[:, 0], np.array(desired_ia)[:-1], 'k--', label='desired ia')
            plt.plot(np.array(env.reY)[:, 0], np.array(actual_ia)[:-1], 'k-', label='actual ia')
            plt.xlabel('Time (s)')
            plt.ylabel('$impact angle(s)$')
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot()
            plt.plot(np.array(env.reY)[:, 0], np.array(impact_angle_error)[:-1], 'k-')
            plt.xlabel('Time (s)')
            plt.ylabel('impact angle error(s)')
            plt.grid()

            plt.pause(0.1)

        savemat('./flight_sim_tsg_monte.mat', flight_data)
