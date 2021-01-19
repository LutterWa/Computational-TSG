import numpy as np
import scipy.io
from OAgent_sim import OAgentSim

oAgent = OAgentSim()
dict_launch = {}

pi = 3.141592653589793
RAD = 180 / pi

for launch_time in range(int(1e3)):
    oAgent.modify()
    print("========", launch_time + 1, "========")
    step = []
    done = False
    while done is False:
        done = oAgent.step(action=0, h=0.01)
        v, theta, r, q, x, y, t = oAgent.collect()
        step.append([v, theta, r, q, x, y])
    s = np.array(step)
    t = theta * RAD * np.ones([s.shape[0], 1])  # 换算为角度
    dict_launch['dnn{0}'.format(launch_time)] = np.concatenate([s, t], axis=1)
flight_data = {'flight_data': dict_launch}
scipy.io.savemat('./flight_data.mat', flight_data)
