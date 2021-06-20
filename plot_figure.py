import matplotlib.pylab as plt
import pandas as pd

# 读取csv结果文件
results = pd.read_csv("nine_periods.csv")

def print_figure(period_start_index, x_range, x_name, title):
    # 初始化
    x_axis = []
    x = []
    x_text = []
    main1 = []
    main2 = []
    y_main = []
    y_one_agent = []
    y_direct = []

    # 取时段内参数组数
    for i in range(x_range):
        x_axis.append(i)
        index = period_start_index + i
        x.append(results[x_name][index])
        main1.append(results['main1'][index])
        main2.append(results['main2'][index])
        y_main.append(results['main'][index])
        y_one_agent.append(results['one_agent'][index])
        y_direct.append(results['direct'][index])
        x_text.append("{}: {}\nmain1: {}\nmain2: {}".format(x_name,x[i],main1[i],main2[i]))

    # 绘图
    plt.plot(x_axis, y_main, 'xr-')
    plt.plot(x_axis, y_one_agent, 'oy-')
    plt.plot(x_axis, y_direct, '^b-')
    # x刻度归一化处理，并加上main1，main2的数据
    plt.xticks(x_axis, x_text)
    plt.ylabel("Success rate", fontsize = 12)
    plt.legend(['main', 'one_agent', 'direct'])
    # 添加标题
    plt.title(title)
    plt.show()

print_figure(0, 5, "mu", "Period 1, v:[0-20], d:[4-10]")
print_figure(5, 5, "mu", "Period 2, v:[20-40], d:[10-30]")
print_figure(10, 5, "mu", "Period 3, v:[40-60], d:[30-60]")

print_figure(15, 4, "s_sigma", "Period 1, v:[0-20], d:[4-10]")
print_figure(19, 4, "s_sigma", "Period 2, v:[20-40], d:[10-30]")
print_figure(23, 4, "s_sigma", "Period 3, v:[40-60], d:[30-60]")

print_figure(27, 4, "d_sigma", "Period 1, v:[0-20], d:[4-10]")
print_figure(31, 4, "d_sigma", "Period 2, v:[20-40], d:[10-30]")
print_figure(35, 4, "d_sigma", "Period 3, v:[40-60], d:[30-60]")