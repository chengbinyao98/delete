import numpy as np
import math
from pylab import *
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm


class Env2(object):
    def __init__(self):
        # 固定量
        # 帧结构
        self.frame_slot = 0.01          # 帧时隙时间长度
        self.beam_slot = 100            # 波束选择时隙数
        self.right = 5                  # 正确传输最低的SNR
        # 车辆和道路
        self.road_length = 200          # 道路长度
        self.straight = 100             # 基站和道路的直线距离
        self.car_length = 5
        self.max_speed = 105 * 0.277777778

        # 存储单元
        self.cars_posit = []  # 车辆的位置（连续）
        self.cars_speed = []  # 车辆的速度（连续

        # 变化量
        # 算法变化量
        self.road_section = 2  # 每几米划分成一个路段
        self.action_section = 1  # 每几米划分成一个路段
        self.road_range = 35  # 动作可以选择的范围

        # 通信变化量
        self.ann_num = 32  # 天线数目
        self.no_interference = 30

        # 道路变化量
        self.s_mu, self.s_sigma = 0.75, 0.25  # 车速分布
        self.d_sigma = 2  # 车辆间距分布

        # 同一个时段不用变化
        self.s_point1 = 0 * 0.277777778  # 车速范围
        self.s_point2 = 20 * 0.277777778
        self.d_point1 = 4  # 车间距范围
        self.d_point2 = 10
        safe_dis = 4  # 安全距离
        km = self.road_length / (self.car_length + safe_dis)  # 由安全距离计算最大车辆密度
        mean = math.exp(self.s_mu + self.s_sigma * self.s_sigma / 2)  # 由交通流理论计算车辆间距
        k = km / math.exp(mean / self.max_speed)
        distance = self.road_length / k - self.car_length
        self.d_mu = math.log(distance) - self.d_sigma * self.d_sigma / 2

        # self.v_min = 8  # 车辆的最小速度
        # self.v_max = 16  # 车辆的最大速度
        # self.accelerate = 16            # 车辆的加速度
        # self.min_dis = 22  # 车辆之间的最小反应距离
        # self.max_dis = 28

    # region 【功能函数】生成截断对数正态分布，要求对数正态在[log_lower,log_upper]
    def log_zhengtai(self, mu, sigma, log_lower, log_upper, data_num = 1):
        norm_lower = np.log(log_lower)
        norm_upper = np.log(log_upper)
        X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc = mu, scale = sigma)
        norm_data = X.rvs(data_num)
        log_data = np.exp(norm_data)
        return log_data

    def exp(self, scale, low, high, data_num = 1):  # scale是均值不是lamda，是1/lamda
        rnd_cdf = np.random.uniform(stats.expon.cdf(x = low, scale = scale),
                                    stats.expon.cdf(x = high, scale = scale),
                                    size = data_num)
        return stats.expon.ppf(q = rnd_cdf, scale = scale)

    # 由道路上的所有车辆得到所有车辆的路段
    def get_section(self, list):
        section = []
        for i in range(len(list)):
            section.append(math.ceil(list[i] / self.road_section))
        return section

    # 道路路面现有车辆更新
    # def road_step(self):
    #     mark = 0  # 标记当前车辆是否之前被操作过，保证一个时隙车只跑一个时隙的量
    #     for i in range(len(self.cars_posit) - 1):
    #         if mark == 0:
    #             if self.cars_posit[i + 1] - self.cars_posit[i] < self.sa                                                                       :
    #                 if np.random.rand() < 0.5:
    #                     cars_speed_next = self.cars_speed[i] - self.accelerate * self.frame_slot
    #                     # 减速到最小速度即可
    #                     if cars_speed_next <= self.v_min:
    #                         cars_speed_next = self.v_min
    #                     ti = (self.cars_speed[i] - cars_speed_next) / self.accelerate
    #                     self.cars_posit[i] = self.cars_speed[i] * ti - ti * ti * self.accelerate / 2 + (
    #                             self.frame_slot - ti) * cars_speed_next + self.cars_posit[i]
    #                     self.cars_speed[i] = cars_speed_next
    #                     mark = 0
    #                 else:
    #                     cars_speed_next = self.cars_speed[i + 1] + self.accelerate * self.frame_slot
    #                     # 减速到最小速度即可
    #                     if cars_speed_next >= self.v_max:
    #                         cars_speed_next = self.v_max
    #                     ti1 = (cars_speed_next - self.cars_speed[i + 1]) / self.accelerate
    #                     self.cars_posit[i + 1] = self.cars_speed[i + 1] * ti1 + ti1 * ti1 * self.accelerate / 2 + (
    #                             self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i + 1]
    #                     self.cars_speed[i + 1] = cars_speed_next
    #                     self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
    #                     mark = 1
    #             if self.cars_posit[i + 1] - self.cars_posit[i] > self.max_dis:
    #                 if np.random.rand() < 0.5:
    #                     cars_speed_next = self.cars_speed[i+1] - self.accelerate * self.frame_slot
    #                     # 减速到最小速度即可
    #                     if cars_speed_next <= self.v_min:
    #                         cars_speed_next = self.v_min
    #                     ti1 = (self.cars_speed[i+1] - cars_speed_next) / self.accelerate
    #                     self.cars_posit[i+1] = self.cars_speed[i+1] * ti1 - ti1 * ti1 * self.accelerate / 2 + (
    #                             self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i+1]
    #                     self.cars_speed[i+1] = cars_speed_next
    #                     self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
    #                     mark = 2
    #                 else:
    #                     cars_speed_next = self.cars_speed[i] + self.accelerate * self.frame_slot
    #                     # 减速到最小速度即可
    #                     if cars_speed_next >= self.v_max:
    #                         cars_speed_next = self.v_max
    #                     ti = (cars_speed_next - self.cars_speed[i]) / self.accelerate
    #                     self.cars_posit[i] = self.cars_speed[i] * ti + ti * ti * self.accelerate / 2 + (
    #                             self.frame_slot - ti) * cars_speed_next + self.cars_posit[i]
    #                     self.cars_speed[i] = cars_speed_next
    #                     mark = 0
    #             if self.min_dis < self.cars_posit[i + 1] - self.cars_posit[i] < self.max_dis:
    #                 self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
    #                 mark = 0
    #         else:
    #             if self.cars_posit[i + 1] - self.cars_posit[i] < self.min_dis:
    #                 cars_speed_next = self.cars_speed[i + 1] + self.accelerate * self.frame_slot
    #                 # 减速到最小速度即可
    #                 if cars_speed_next >= self.v_max:
    #                     cars_speed_next = self.v_max
    #                 ti1 = (cars_speed_next - self.cars_speed[i + 1]) / self.accelerate
    #                 self.cars_posit[i + 1] = self.cars_speed[i + 1] * ti1 + ti1 * ti1 * self.accelerate / 2 + (
    #                         self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i + 1]
    #                 self.cars_speed[i + 1] = cars_speed_next
    #                 mark = 1
    #             if self.cars_posit[i + 1] - self.cars_posit[i] > self.max_dis:
    #                 cars_speed_next = self.cars_speed[i+1] - self.accelerate * self.frame_slot
    #                 # 减速到最小速度即可
    #                 if cars_speed_next <= self.v_min:
    #                     cars_speed_next = self.v_min
    #                 ti1 = (self.cars_speed[i+1] - cars_speed_next) / self.accelerate
    #                 self.cars_posit[i+1] = self.cars_speed[i+1] * ti1 - ti1 * ti1 * self.accelerate / 2 + (
    #                         self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i+1]
    #                 self.cars_speed[i+1] = cars_speed_next
    #                 mark = 2
    #             if self.min_dis < self.cars_posit[i + 1] - self.cars_posit[i] < self.max_dis:
    #                 mark = 0
    #     if mark == 0:
    #         self.cars_posit[len(self.cars_posit) - 1] = self.cars_speed[len(self.cars_posit) - 1] * self.frame_slot + \
    #                                                     self.cars_posit[len(self.cars_posit) - 1]

    def road_step(self):
        for i in range(len(self.cars_posit)):
            self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]

    def get_reward(self, act,reward,n):
        if self.cars_posit[1] - self.cars_posit[0] <= self.no_interference:
            for i in range(n):
                SNR_noise = 0
                SNR = 0
                for j in range(n):
                    # 直角边
                    a = abs(self.road_length / 2 - self.cars_posit[i])
                    # 斜边
                    b = np.sqrt(np.square(a) + np.square(self.straight))
                    if self.cars_posit[i] > self.road_length / 2:
                        th1 = math.pi - math.acos(a / b)
                    else:
                        th1 = math.acos(a / b)

                    channel = []
                    for t in range(self.ann_num):
                        m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
                        channel.append(m.conjugate())

                    # 直角边
                    c = abs(self.road_length / 2 - act[j])
                    # 斜边
                    d = np.sqrt(np.square(c) + np.square(self.straight))
                    if act[j] > self.road_length / 2:
                        th2 = math.pi - math.acos(c / d)
                    else:
                        th2 = math.acos(c / d)

                    signal = []
                    for t in range(self.ann_num):
                        p = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
                        signal.append(p)

                    if j != i:
                        SNR_noise += np.square(np.linalg.norm(np.dot(channel, signal)))
                    else:
                        SNR = np.square(np.linalg.norm(np.dot(channel, signal)))
                if SNR_noise == 0:
                    if SNR >= self.right:
                        reward[i] += 1
                else:
                    if SNR / SNR_noise >= self.right:
                        reward[i] += 1

        else:
            for i in range(n):
                a = abs(self.road_length / 2 - self.cars_posit[i])
                # 斜边
                b = np.sqrt(np.square(a) + np.square(self.straight))
                if self.cars_posit[i] > self.road_length / 2:
                    th1 = math.pi - math.acos(a / b)
                else:
                    th1 = math.acos(a / b)

                channel = []
                for t in range(self.ann_num):
                    m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
                    channel.append(m.conjugate())

                # 直角边
                c = abs(self.road_length / 2 - act[i])
                # 斜边
                d = np.sqrt(np.square(c) + np.square(self.straight))
                if act[i] > self.road_length / 2:
                    th2 = math.pi - math.acos(c / d)
                else:
                    th2 = math.acos(c / d)

                signal = []
                for t in range(self.ann_num):
                    n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
                    signal.append(n)

                SNR = np.square(np.linalg.norm(np.dot(channel, signal)))

                if SNR >= self.right:
                    reward[i] += 1
        return reward

    def reset(self,n):
        # 道路环境初始化
        self.cars_posit = []  # 车辆的位置（连续）
        self.cars_speed = []  # 车辆的速度（连续)
        speed = self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)[0]
        for i in range(n):
            dis = self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]
            if i == 0:
                self.cars_posit.append(dis)
                self.cars_speed.append(speed)
            else:
                self.cars_posit.append(self.cars_posit[i-1]+dis)
                self.cars_speed.append(speed)

        a = self.get_section(self.cars_posit)

        state = []
        for i in range(n):
            state.append([a[i],a[i]])
        return state

    def step(self, action, state, n):
        reward = [0 for p in range(n)]
        for i in range(self.beam_slot):
            self.road_step()
            reward = self.get_reward(action, reward, n)

        add_reward = 0
        for i in range(n):
            add_reward += reward[i]

        now = self.get_section(self.cars_posit)
        state_ = []
        for i in range(n):
            state_.append([now[i],state[i][0]])

        if self.cars_posit[n-1] > self.road_length:
            done = 1
        else:
            done = 0

        return state_,add_reward,done

