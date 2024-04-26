from collections import namedtuple
import numpy as np
from numpy.random import randn
from math import sqrt
import matplotlib.pyplot as plt


gaussian = namedtuple("Gaussian", ["mean", "var"])
gaussian.__repr__ = lambda s: f"N=(mean {s[0]:.3f}, var= {s[1]:.3f})"


def gaussian_add(g1, g2):
    return gaussian(g1.mean + g2.mean, g1.var + g2.var)


def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)


def predict(posterior, movement):
    x, P = posterior
    dx, Q = movement

    x = x + dx
    p = P + Q
    return gaussian(x, P)


def update(measurement, prior):
    x, P = prior
    z, R = measurement

    y = z - x
    K = P / (P + R)

    x = x + K * y
    P = (1 - K) * P
    return gaussian(x, P)


def print_result(predict, update, z, epoch):
    predict_template = '{:3.0f} {: 7.3f} {: 8.3f}'
    update_template = '\t{: .3f}\t {: 7.3f} {: 7.3f}'
    print(predict_template.format(epoch, predict[0], predict[1]), end='\t')
    print(update_template.format(z, update[0], update[1]))


def plot_result(epoch, prior_list, x_list, z_list):
    epoch_list = np.array(epoch)

    plt.plot(epoch_list, prior_list, linestyle=':', color='r', label="prior/predicted_pos", lw=2)
    plt.plot(epoch_list, x_list, linestyle='-', color='g', label="posterior/updated_pos", lw=2)
    plt.plot(epoch_list, z_list, linestyle=':', color='b', label="likelihood/measurement", lw=2)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


"""
人估算的运动方差 1.0
GPS的传感器方差 2.0

人的初始位置 N(0, 20*20)
人的初始速度v 1m/s

运动模型(过程模型) N(v, 1.0)
"""

# 1. 初始数据
motion_var = 1.0  # 人的运动方差
sensor_var = 2.0  # gps的运动方差
x = gaussian(0, 20 ** 2)
velocity = 1.0
dt = 1  # 时间单位的最小刻度
motion_model = gaussian(velocity, motion_var)

# 2 生成数据
zs = []
current_x = x.mean
for _ in range(10):
    # 2.1 生成运动数据
    v = velocity + randn() * motion_var
    current_x += v * dt
    # 2.2 生成观测数据
    measurement = current_x + randn() * sensor_var
    zs.append(measurement)


prior_list, x_list, z_list = [], [], []
print('epoch\tpredict\t\t\tupdate')
print('     \tx       var\t\t     z\t        x        var')

for epoch, z in enumerate(zs):
    prior = predict(x, motion_model)     #运动预测，两个高斯之和
    likelihood = gaussian(z, sensor_var)

    x = update(likelihood, prior)        #结合观测 两个高斯的交集
    print_result(prior, x, z, epoch)
    prior_list.append(prior.mean)
    x_list.append(x.mean)
