import numpy as np
import random
import matplotlib.pyplot as plt  # clf()清图  # cla()清坐标轴  # close()关窗口
from scipy import interpolate
from scipy.io import savemat
from math import sin, cos, atan2, sqrt, asin

# 常量
pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度
g = 9.81  # 重力加速度
S = 0.0572555  # 特征面积

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

p_flag = False  # 是否考虑推力

Ma2 = np.array([[0.4, 39.056, 0.4604, 39.072],
                [0.6, 40.801, 0.4682, 39.735],
                [0.8, 41.372, 0.4635, 39.242],
                [0.9, 42.468, 0.4776, 40.351]])


def load_atm(path):
    file = open(path)
    atm_str = file.read().split()
    atm = []
    for _ in range(0, len(atm_str), 3):
        atm.append([float(atm_str[_]), float(atm_str[_ + 1]), float(atm_str[_ + 2])])
    return np.array(atm)


class MISSILE:
    def __init__(self, missile=None, target=None):
        if missile is None:
            missile = [300.,
                       0. / RAD,
                       0. / RAD,
                       -20000.,
                       20000.,
                       0.]  # 1.速度,2.弹道倾角,3.弹道偏角 4.导弹x,5.导弹y,6.导弹z  y指向天
        if target is None:
            target = [0., 0., 0., -90. / RAD, 0. / RAD]  # 目标x,y,z,theta,psi

        self.V = np.array(missile[:3])  # v,theta,psi
        self.Euler = np.array(missile[3:])  # x,y,z
        self.Y = np.concatenate([[0.], self.V, self.Euler, [200.]])  # 0.时间, 1.速度，2.位置，3.重量

        self.Rt = target[:3]  # 目标信息
        self.Vt = target[3:]  #

        self.X = 0.  # 阻力drag force
        self.L = 0.  # 升力lift force
        self.B = 0.  # 侧向力
        R = self.Rt - self.Euler
        self.R = np.linalg.norm(R)  # 弹目距离
        self.q_theta = atan2(R[1], sqrt(R[0] ** 2 + R[2] ** 2))  # 弹目视线与水平面夹角
        self.q_psi = atan2(R[2], R[0])  # 弹目视线与北向夹角，北偏西为正
        self.Rdot = 0.
        self.qdot_theta = 0.
        self.qdot_psi = 0.
        self.acy = 0.  # 制导指令
        self.acz = 0.
        self.aby = 0.  # 偏置项
        self.abz = 0.  # 偏置项
        self.alpha = 0.  # 攻角
        self.beta = 0.  # 侧滑角

        # 创建插值函数
        atm = load_atm('atm2.txt')  # 大气参数
        self.f_ma = interpolate.interp1d(atm[:, 0], atm[:, 2], 'linear')
        self.f_rho = interpolate.interp1d(atm[:, 0], atm[:, 1], 'linear')

        k = 1.0
        self.f_clalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 1] * k, 'cubic')
        self.f_cd0 = interpolate.interp1d(Ma2[:, 0], Ma2[:, 2] * k, 'cubic')
        self.f_cdalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 3] * k, 'cubic')

        # 全弹道历史信息
        self.reY, self.reacy, self.reacz = [], [], []

    def modify(self, missile=None):  # 修改导弹初始状态
        if missile is None:
            missile = [random.uniform(200, 300),
                       random.uniform(0, 45) / RAD,
                       random.uniform(-5, 5) / RAD,
                       random.uniform(-30000., -10000),
                       random.uniform(10000, 30000),
                       random.uniform(-2000, 2000)]  # 1.速度,2.弹道倾角,3.弹道偏角 4.导弹x,5.导弹y,6.导弹z

        if missile[2] * missile[5] > 0:
            missile[2] *= -1
        self.V = np.array(missile[:3])  # v,theta,psi
        self.Euler = np.array(missile[3:])  # x,y,z
        self.Y = np.concatenate([[0.], self.V, self.Euler, [200.]])  # 0.时间, 1.速度，2.位置，3.重量

        R_nue = self.Rt - self.Euler
        self.R = np.linalg.norm(R_nue)  # 弹目距离
        self.q_theta = atan2(R_nue[1], sqrt(R_nue[0] ** 2 + R_nue[2] ** 2))  # 弹目视线与水平面夹角
        self.q_psi = atan2(R_nue[2], R_nue[0])  # 弹目视线与北向夹角，北偏西为正

        k = random.uniform(0.8, 1.2)
        # k = 1.0
        self.f_clalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 1] * k, 'cubic')
        self.f_cd0 = interpolate.interp1d(Ma2[:, 0], Ma2[:, 2] * k, 'cubic')
        self.f_cdalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 3] * k, 'cubic')

        self.reY, self.reacy, self.reacz = [], [], []
        return self.collect()

    def get_ma(self, y, v):  # 计算马赫数
        y = max(0, y)
        sonic = self.f_ma(y)
        return v / sonic

    def get_rho(self, y):  # 计算空气密度
        y = max(0, y)
        return self.f_rho(y)

    def get_clalpha(self, ma):
        return self.f_clalpha(max(min(ma, 0.9), 0.4))

    def get_cd0(self, ma):
        return self.f_cd0(max(min(ma, 0.9), 0.4))

    def get_cdalpha(self, ma):
        return self.f_cdalpha(max(min(ma, 0.9), 0.4))

    def dery(self, Y):  # 右端子函数
        v, theta, psi = Y[1:4]  # v,theta,psi
        x, y, z = Y[4:7]  # x,y,z
        m = Y[-1]
        dy = np.array(Y)

        # 速度倾斜角gammav = 0
        dy[0] = 1  # t
        dy[1] = (-self.X / m) - g * sin(theta)  # v
        dy[2] = (self.L - m * g * cos(theta)) / (m * v)  # theta
        dy[3] = -self.B / (m * v * cos(theta))  # psi
        dy[4] = v * cos(theta) * cos(psi)  # x北向
        dy[5] = v * sin(theta)  # y天向
        dy[6] = -v * cos(theta) * sin(psi)  # 东向
        dy[7] = 0
        return dy

    def step(self, action_y=0., action_z=0., h=0.001):
        if self.Y[0] < 400:
            self.V = v, theta, psi = self.Y[1:4]  # v,theta,psi
            self.Euler = x, y, z = self.Y[4:7]  # x,y,z
            m = self.Y[-1]  # 弹重
            RHO = self.get_rho(y)  # 大气密度
            ma = self.get_ma(y, v)  # 马赫数

            Q = 0.5 * RHO * v ** 2  # 动压

            R_nue = self.Rt - self.Euler
            R = self.R  # 上一周期的弹目距离
            self.R = np.linalg.norm(R_nue)  # 弹目距离

            self.q_theta = q_theta = atan2(R_nue[1], sqrt(R_nue[0] ** 2 + R_nue[2] ** 2))  # 弹目视线与水平面夹角
            self.q_psi = q_psi = atan2(R_nue[2], R_nue[0])  # 弹目视线与北向夹角，北偏西为正

            v_nue = [v * cos(theta) * cos(psi), v * sin(theta), -v * cos(theta) * sin(psi)]
            nav_theta = atan2(v_nue[1], sqrt(v_nue[0] ** 2 + v_nue[2] ** 2))  # 根据卫星解算出的速度倾角(速度与水平面夹角)
            nav_psiv = atan2(v_nue[2], v_nue[0])  # 根据卫星解算出的速度与北向夹角

            # if R < 2:
            #     return True, nav_theta, nav_psiv
            if y < 0:
                print("弹已落地, 弹目距离={:.1f}".format(self.R))
                return True, nav_theta, nav_psiv
            elif self.R > R:
                print("远离目标, 弹目距离={:.1f}".format(self.R))
                return True, nav_theta, nav_psiv

            R = self.R  # 当前周期的弹目距离

            # 计算前置角
            eta1 = q_theta - nav_theta
            eta2 = q_psi - nav_psiv

            # 计算视线角变化率
            self.qdot_theta = qdot_theta = v * sin(eta1) / R
            self.qdot_psi = qdot_psi = v * cos(nav_theta) * sin(eta2) / R * cos(q_theta)

            # 实际项目中，飞行系数插值的三个维度分别为马赫数mach，舵偏角delta，攻角alpha
            cl_alpha = self.get_clalpha(ma)

            # self.aby = action_y
            # self.abz = action_z
            # self.acy = acy = 3 * v * qdot_theta + cos(nav_theta) * g + float(action_y)  # 制导指令
            # self.acz = acz = 3 * v * cos(nav_theta) * qdot_psi + float(action_z)  # 制导指令

            # 弹道成型
            tsg_n = 0.5
            Nv = 2 * (tsg_n + 2)
            Nq = (tsg_n + 1) * (tsg_n + 2)
            tgo = R / v
            acy = Nv * v * qdot_theta + cos(nav_theta) * g + \
                  Nq * v * (q_theta - action_y) / tgo
            acz = Nv * v * cos(nav_theta) * qdot_psi + \
                  Nq * v * cos(nav_theta) * (q_psi + action_z) / tgo

            m_max = 3 * g
            self.acy = acy = np.clip(acy, -m_max, m_max)
            self.acz = acz = np.clip(acz, -m_max, m_max)

            alpha = (m * acy) / (Q * S * cl_alpha)  # 使用了sin(x)=x的近似，在10°以内满足这一关系
            beta = (m * acz) / (Q * S * cl_alpha)  # 使用了sin(x)=x的近似，在10°以内满足这一关系

            a_max = 20. / RAD
            self.alpha = alpha = np.clip(alpha, -a_max, a_max)
            self.beta = beta = np.clip(beta, -a_max, a_max)

            cd = self.get_cd0(ma) + self.get_cdalpha(ma) * (alpha ** 2 + beta ** 2)  # 阻力系数
            cl = cl_alpha * alpha  # 升力系数
            cb = cl_alpha * beta  # 侧向力系数

            self.X = cd * Q * S  # 阻力
            self.L = cl * Q * S  # 升力
            self.B = cb * Q * S  # 侧向力

            def rk4(func, Y, h=0.1):
                k1 = h * func(Y)
                k2 = h * func(Y + 0.5 * k1)
                k3 = h * func(Y + 0.5 * k2)
                k4 = h * func(Y + k3)
                output = Y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                return output

            self.Y = rk4(self.dery, self.Y, h)
            if self.Y[2] > 2 * pi:
                self.Y[2] = self.Y[2] - 2 * pi
            if self.Y[2] < -2 * pi:
                self.Y[2] = self.Y[2] + 2 * pi

            if self.Y[3] > 2 * pi:
                self.Y[3] = self.Y[3] - 2 * pi
            if self.Y[3] < -2 * pi:
                self.Y[3] = self.Y[3] + 2 * pi

            self.reY.append(self.Y)
            self.reacy.append(self.acy)
            self.reacz.append(self.acz)
            return False, nav_theta, nav_psiv
        else:
            print("超时！未击中目标！")
            # self.plot_data()
            return True, 0., 0.

    def plot_data(self):
        reY = np.array(self.reY)
        # plt.ion()
        plt.clf()
        fig = plt.axes(projection='3d')
        fig.plot3D(reY[:, 4] / 1000, reY[:, 6] / 1000, reY[:, 5] / 1000, "k-")
        fig.set_xlabel("x")
        fig.set_ylabel("y")
        fig.set_zlabel("z")

        fig = plt.figure(2)
        fig.clf()
        ax1 = fig.add_subplot(211)
        ax1.plot(reY[:, 4] / 1000, reY[:, 5] / 1000, 'k-')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        plt.grid()

        ax2 = fig.add_subplot(212)
        ax2.plot(reY[:, 4] / 1000, reY[:, 6] / 1000, 'k-')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        plt.grid()

        # plt.pause(0.1)
        plt.show()

    def collect(self):
        t = self.Y[0]  # 时间
        v = self.Y[1]  # 速度
        theta = self.Y[2]  # 弹道倾角
        psi = self.Y[3]  # 弹道偏角
        r = self.R  # 弹目距离
        q_theta = self.q_theta  # 弹目视线角
        q_psi = self.q_psi  # 弹目视线角
        x = self.Y[4]  # 弹轴向位置
        y = self.Y[5]  # 弹纵向位置
        z = self.Y[6]  # 弹横向位置
        return v, theta, psi, r, q_theta, q_psi, x, y, z, t

    def save_data(self):
        data = {'Y': np.array(self.reY),
                'R': np.array(self.R),
                'acy': np.array(self.reacy),
                'acz': np.array(self.reacz)}
        return data


if __name__ == '__main__':
    env = MISSILE()
    flight_data = {}
    for i in range(100):
        env.modify([200., 0. / RAD, 0. / RAD, -20000., 10000., 5000.])
        tdy = random.uniform(-60., -20.) / RAD
        tdz = random.uniform(-20., 20.) / RAD
        done = False
        while done is False:
            done, a, b = env.step(tdy, tdz, h=0.01)  # TSG
        env.plot_data()
        flight_data['sim_{}'.format(i)] = env.save_data()
        flight_data['time{}'.format(i)] = {'error_y': [(tdy - env.Y[2]) * RAD, 0],
                                           'error_z': [(tdz - env.Y[3]) * RAD, 0]}
        print('Testing | Episode: {}/{} | Target: {:.2f}/{:.2f} | Actual: {:.2f}/{:.2f} | Error: {:.2f}/{:.2f}'
              .format(i + 1, 100,
                      tdy * RAD, tdz * RAD,
                      env.Y[2] * RAD, env.Y[3] * RAD,
                      (tdy - env.Y[2]) * RAD, (tdz - env.Y[3]) * RAD))
    savemat('./mats/flight_sim_tsg_monte_3d.mat', flight_data)
