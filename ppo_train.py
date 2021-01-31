import random
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from OAgent_sim import OAgentSim
from ppo_model import PPO
from dnn_test import DeepNeuralNetwork

BATCH_SIZE = 32  # update batch size
TRAIN_EPISODES = 500  # total number of episodes for training
TEST_EPISODES = 100  # total number of episodes for testing
GAMMA = 0.95
REWARD_SAVE_CASE = 0
dnn = DeepNeuralNetwork()
pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度


class OMissile(OAgentSim):
    def __init__(self, no_=0, missile=None, target=None):
        super().__init__(no_=0, missile=None, target=None)
        v, theta, r, q, x, y, t = self.collect()
        self.ia = float(dnn.predict([[v / 1e2, theta / -1, r / 1e4, q / -1, x / 1e4, y / 1e4]]))
        self.tgo = float(tgo.predict([[v / 315, theta / -0.6, x / -1e4, y / 1.5e4]]))

    def get_ia(self):
        v, theta, r, q, x, y, t = self.collect()
        dnn_state = [v / 1e2, theta / -1, r / 1e4, q / -1, x / 1e4, y / 1e4]
        self.ia = float(dnn.predict([dnn_state]))
        return self.ia

    def get_state(self, a_target):
        v, theta, r, q, x, y, t = self.collect()
        ia = self.get_ia()
        tgo = r / v
        state_local = [(a_target - ia) / tgo]
        return np.array(state_local)

    def get_reward(self, t_target):
        v, theta, r, q, x, y, t = self.collect()
        tgo = r / v
        e_local = (t_target - self.ia) / tgo
        vy = v * math.sin(theta)  # y向速度
        zem = y + vy * tgo
        k1 = 0.6
        k2 = 0.2
        k3 = 0.2
        reward_local = k1 * math.exp(-(e_local / 1e1) ** 2) + \
                       k2 * math.exp(-(zem / 1e4) ** 2) + \
                       k3 * math.exp(-self.ac ** 2)
        return np.array(reward_local)


if __name__ == '__main__':
    env = OMissile()

    # set the init parameter
    state_dim = 1
    action_dim = 1
    action_bound = 3 * 9.81  # action limitation
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
            td = random.uniform(-30, -150)
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
            # env.plot_data(figure_num=0)

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

            # plt.figure()
            # plt.subplots_adjust(hspace=0.6)
            # plt.subplot(2, 1, 1)
            # plt.plot(np.array(env.reY)[:, 0], np.array(desired_ia)[:-1], 'k--', label='desired ia')
            # plt.plot(np.array(env.reY)[:, 0], np.array(actual_ia)[:-1], 'k-', label='actual ia')
            # plt.xlabel('Time (s)')
            # plt.ylabel('t_go(s)')
            # plt.legend()
            # plt.grid()
            #
            # plt.subplot(2, 1, 2)
            # plt.plot()
            # plt.plot(np.array(env.reY)[:, 0], np.array(impact_angle_error)[:-1], 'k-')
            # plt.xlabel('Time (s)')
            # plt.ylabel('impact time error(s)')
            # plt.grid()
            # plt.show()

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

        plt.figure(1)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join(['PPO', time.strftime("%Y_%m%d_%H%M")])))
        plt.show()
    else:
        # test
        agent = PPO(state_dim, action_dim, action_bound, r'./ppo_model')
        for episode in range(TEST_EPISODES):
            # env.modify()
            env.modify()
            td = random.uniform(-30, -150)
            desired_ia = []  # 期望的ia
            actual_ia = []  # 实际的ia
            impact_angle_error = []  # ia的误差
            state = env.get_state(td)
            action = 0
            episode_reward = 0
            done = False
            t = []
            while done is False:
                action = agent.get_action(state, greedy=True)  # use the mean of distribution as action
                # if td - env.Y[0] - env.ia < 0:
                #     action = np.array([0.])
                # else:
                #     action = np.array([30.])
                done = env.step(action=action, h=0.1)
                state = env.get_state(td)
                reward = env.get_reward(td)
                episode_reward += reward

                desired_ia.append(td)
                actual_ia.append(env.ia)
                impact_angle_error.append(td - env.ia)

            env.plot_data(figure_num=0)
            # print the result
            episode_reward = episode_reward / env.Y[0]  # calculate the average episode reward
            print('Testing | Episode: {}/{} | Average Episode Reward: {:.2f} | Running Time: {:.2f} | '
                  'Target Time: {:.2f} | Actual Time: {:.2f} | Error Time: {:.2f}'
                  .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0,
                          td, env.Y[2] * RAD, td - env.Y[2] * RAD))
            plt.figure()
            plt.subplots_adjust(hspace=0.6)
            plt.subplot(2, 1, 1)
            plt.plot(np.array(env.reY)[:, 0], np.array(desired_ia)[:-1], 'k--', label='desired ia')
            plt.plot(np.array(env.reY)[:, 0], np.array(actual_ia)[:-1], 'k-', label='actual ia')
            plt.xlabel('Time (s)')
            plt.ylabel('t_go(s)')
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot()
            plt.plot(np.array(env.reY)[:, 0], np.array(impact_angle_error)[:-1], 'k-')
            plt.xlabel('Time (s)')
            plt.ylabel('impact time error(s)')
            plt.grid()
            plt.show()
