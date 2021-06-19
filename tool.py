# from environment import Env
import numpy as np

class Tools(object):
    #
    # def get_info(self, pos, n):  # 看看不用info的内部存储可以么
    #     num = 0
    #     cars_info = [[0 for m in range(3)] for p in range(len(pos))]  # 注意以后再用的时候能不能行
    #     for i in range(int(len(pos) / n)):
    #         for j in range(n):
    #             cars_info[num][0] = n
    #             cars_info[num][1] = i
    #             cars_info[num][2] = j
    #             num += 1
    #     return cars_info

    def get_list(self, a):
        temp_state = []
        for dim in range(len(a)):
            temp_state.append(a[dim][0])
            temp_state.append(a[dim][1])
        temp_state = np.array(temp_state)
        return temp_state