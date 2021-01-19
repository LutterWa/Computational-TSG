import math
import numpy as np
import random
import matplotlib.pyplot as plt  # clf()清图  # cla()清坐标轴  # close()关窗口
from scipy import interpolate

# 常量
pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度
S = 0.0572555  # 特征面积

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

p_flag = False  # 是否考虑推力

# Ma2 = np.array([[0.4, 39.056, 0.4604, 39.072],
#                 [0.6, 40.801, 0.4682, 39.735],
#                 [0.8, 41.372, 0.4635, 39.242],
#                 [0.9, 42.468, 0.4776, 40.351]])
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


class OAgentSim:
    def __init__(self, no_=0, missile=None, target=None):
        self.no_ = no_  # 导弹编号

        if missile is None:
            missile = [0.,
                       300.,
                       45. / RAD,
                       -20000.,
                       10000,
                       200]  # 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量
        if target is None:
            target = [0., 0., -60.]  # 目标x,y,落角

        self.Y = np.array(missile)
        self.xt, self.yt, self.qt = target[0], target[1], target[2] / RAD  # 目标信息

        self.X = 0.  # 阻力drag force
        self.L = 0.  # 升力lift force
        Rx = self.xt - self.Y[3]
        Ry = self.yt - self.Y[4]
        self.R = np.linalg.norm([Rx, Ry], ord=2)  # 弹目距离
        self.q = math.atan2(Ry, Rx)  # 弹目视线角
        self.Rdot = 0.
        self.qdot = 0.
        self.ac = 0.  # 制导指令
        self.ab = 0.  # 偏置项
        self.alpha = 0.  # 攻角
        if p_flag:
            self.P = 1000.  # 推力push force
            self.mc = 0.4  # 每秒损耗质量
        else:
            self.P = 0.  # 推力push force
            self.mc = 0.  # 每秒损耗质量
        # self.tgo = (1 + (self.Y[2] - self.q) ** 2 / 10) * self.R / self.Y[1]  # T_go = (1-(theta-lambda)^2/10)*R_go/V

        # 创建插值函数
        atm = load_atm('atm2.txt')  # 大气参数
        self.f_ma = interpolate.interp1d(atm[:, 0], atm[:, 2], 'linear')
        self.f_rho = interpolate.interp1d(atm[:, 0], atm[:, 1], 'linear')
        self.f_clalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 1], 'cubic')
        self.f_cd0 = interpolate.interp1d(Ma2[:, 0], Ma2[:, 2], 'cubic')
        self.f_cdalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 3], 'cubic')

        # 全弹道历史信息
        self.reY, self.reac, self.reCX, self.reP, self.realpha, self.reR = [], [], [], [], [], []

    def modify(self, missile=None):  # 修改导弹初始状态
        if missile is None:
            missile = [0.,
                       random.uniform(200, 300),
                       random.uniform(0, 45) / RAD,
                       random.uniform(-30000., -10000),
                       random.uniform(10000, 30000),
                       200]  # 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量
        self.Y = np.array(missile)
        Rx = self.xt - self.Y[3]
        Ry = self.yt - self.Y[4]
        self.R = np.linalg.norm([Rx, Ry], ord=2)  # 弹目距离
        self.q = math.atan2(Ry, Rx)  # 弹目视线角
        # self.tgo = (1 + (self.Y[2] - self.q) ** 2 / 10) * self.R / self.Y[1]  # T_go = (1-(theta-lambda)^2/10)*R_go/V
        self.reY, self.reac, self.reCX, self.reP, self.realpha, self.reR = [], [], [], [], [], []
        return self.collect()

    def terminate(self):
        self.X = 0.  # 阻力drag force
        self.L = 0.  # 升力lift force
        self.R = 0.  # 弹目距离range
        self.q = 0.  # 弹目视线角
        self.Rdot = 0.
        self.qdot = 0.
        self.ac = 0.  # 制导指令
        self.alpha = 0.  # 攻角
        if p_flag:
            self.P = 1000.  # 推力push force
            self.mc = 0.4  # 每秒损耗质量
        else:
            self.P = 0.  # 推力push force
            self.mc = 0.  # 每秒损耗质量

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
        g = 9.81
        v = Y[1]
        theta = Y[2]
        m = Y[5]
        dy = np.array(Y)
        dy[0] = 1
        dy[1] = (self.P * math.cos(self.alpha) - self.X) / m - g * math.sin(theta)
        dy[2] = (self.P * math.sin(self.alpha) + self.L - m * g * math.cos(theta)) / (v * m)
        dy[3] = v * math.cos(theta)
        dy[4] = v * math.sin(theta)
        dy[5] = -self.mc
        return dy

    def step(self, action=0, h=0.001):
        if self.Y[0] < 400:
            x = self.Y[3]  # 横向位置
            y = self.Y[4]  # 纵向位置
            v = self.Y[1]  # 速度
            theta = self.Y[2]  # 弹道倾角
            m = self.Y[5]  # 弹重
            RHO = self.get_rho(y)  # 大气密度
            ma = self.get_ma(y, v)  # 马赫数

            Q = 0.5 * RHO * v ** 2  # 动压

            Rx = self.xt - x
            Ry = self.yt - y
            vx = -v * math.cos(theta)  # x向速度
            vy = -v * math.sin(theta)  # y向速度
            self.R = R = np.linalg.norm([Rx, Ry], ord=2)  # 弹目距离
            self.qdot = qdot = (Rx * vy - Ry * vx) / R ** 2
            self.q = q = math.atan2(Ry, Rx)  # 弹目视线角
            self.Rdot = Rdot = (Rx * vx + Ry * vy) / R
            # self.tgo = tgo = (1 + (theta - q) ** 2 / 10) * R / v

            if R < 20:
                return True
            elif y < 0:
                print("弹已落地, 弹目距离={:.1f}".format(R))
                return True
            # elif Rdot >= 0:
            #     print("逐渐远离目标...")

            # 工作时序
            t = self.Y[0]
            if p_flag:
                if t <= 86:  # 第86s燃料耗尽
                    self.mc = 0.4  # 秒流量
                    self.P = 1000  # 推力
                else:
                    self.mc = 0.
                    self.P = 0.

            # 实际项目中，飞行系数插值的三个维度分别为马赫数mach，舵偏角delta，攻角alpha
            cl_alpha = self.get_clalpha(ma)

            self.ab = action
            self.ac = ac = 3 * v * qdot + math.cos(theta) * 9.81 + action  # 制导指令  # + 2 * v * (q - self.qt) / tgo
            # self.ac = ac = 4 * v * qdot + math.cos(theta) * 9.81 + 2 * v * (q - self.qt) / (R / v)  # 弹道成型
            self.alpha = alpha = (m * ac) / (Q * S * cl_alpha + self.P)  # 使用了sin(x)=x的近似，在10°以内满足这一关系
            if alpha > 15 / RAD:
                self.alpha = alpha = 15 / RAD
            elif alpha < -15 / RAD:
                self.alpha = alpha = -15 / RAD

            cd = self.get_cd0(ma) + self.get_cdalpha(ma) * alpha ** 2  # 阻力系数
            cl = cl_alpha * alpha  # 升力系数

            self.X = cd * Q * S  # 阻力
            self.L = cl * Q * S  # 升力

            def rk4(func, Y, h=0.1):
                k1 = h * func(Y)
                k2 = h * func(Y + 0.5 * k1)
                k3 = h * func(Y + 0.5 * k2)
                k4 = h * func(Y + k3)
                output = Y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                return output

            self.Y = rk4(self.dery, self.Y, h)
            if self.Y[2] > 2*pi:
                self.Y[2] = self.Y[2] - 2*pi
            if self.Y[2] < -2*pi:
                self.Y[2] = self.Y[2] + 2*pi

            self.reY.append(self.Y)
            self.reac.append(self.ac)
            self.reCX.append(self.X)
            self.reP.append(action)
            self.realpha.append(self.alpha)
            self.reR.append(self.R)
            return False
        else:
            print("超时！未击中目标！")
            # self.plot_data()
            return True

    def collect(self):
        t = self.Y[0]  # 时间
        v = self.Y[1]  # 速度
        theta = self.Y[2]  # 弹道倾角
        r = self.R  # 弹目距离
        q = self.q  # 弹目视线角
        x = self.Y[3]  # 弹横向位置
        y = self.Y[4]  # 弹纵向位置
        return v, theta, r, q, x, y, t

    def plot_data(self, figure_num=0):
        reY = np.array(self.reY)
        reac = np.array(self.reac)
        reCX = np.array(self.reCX)
        reP = np.array(self.reP)
        realpha = np.array(self.realpha)
        reR = np.array(self.reR)
        plt.figure(figure_num)
        plt.clf()
        # 弹道曲线
        plt.subplots_adjust(hspace=0.6)
        plt.subplot(3, 3, 1)
        plt.plot(reY[:, 3] / 1000, reY[:, 4] / 1000, 'k-')
        plt.xlabel('Firing Range (km)')
        plt.ylabel('altitude (km)')
        plt.title('弹道曲线')
        plt.grid()

        # 速度曲线
        plt.subplot(3, 3, 2)
        plt.plot(reY[:, 0], reY[:, 1], 'k-')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('速度')
        plt.grid()

        # 弹道倾角
        plt.subplot(3, 3, 3)
        plt.plot(reY[:, 0], reY[:, 2] * RAD, 'k-')
        plt.xlabel('Time (s)')
        plt.ylabel('Theta (°)')
        plt.title('弹道倾角')
        plt.grid()

        # 过载指令
        plt.subplot(3, 3, 4)
        plt.plot(reY[:, 0], reac, 'k-')
        plt.xlabel('time (s)')
        plt.ylabel('ac')
        plt.title('过载指令')
        plt.grid()

        # 推力
        plt.subplot(3, 3, 5)
        plt.plot(reY[:, 0], reP, 'k-')
        plt.xlabel('Time (s)')
        plt.ylabel('action')
        plt.title('偏置项')
        plt.grid()

        # 高度曲线
        plt.subplot(3, 3, 6)
        plt.plot(reY[:, 0], reY[:, 4] / 1000, 'k-')
        plt.xlabel('time (s)')
        plt.ylabel('altitude (km)')
        plt.title('高度')
        plt.grid()

        # 阻力变化曲线
        plt.subplot(3, 3, 7)
        plt.plot(reY[:, 0], reCX, 'k-')
        plt.title('阻力')
        plt.grid()

        # 平衡攻角曲线
        plt.subplot(3, 3, 8)
        plt.plot(reY[:, 0], realpha * RAD, 'k-')
        plt.title('平衡攻角')
        plt.grid()

        # 弹目距离
        plt.subplot(3, 3, 9)
        plt.plot(reY[:, 0], reR, 'k-')
        plt.title('弹目距离')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    oAgent = OAgentSim()
    for i in range(10):
        oAgent.modify()
        done = False
        while done is False:
            done = oAgent.step(0, 0.01)  # 单步运行
        oAgent.plot_data()
        print(oAgent.Y[2] * RAD)
