import numpy as np
from util import log_util
import torch

windows3_x_step = [-1, 0, 1, -1, 1, -1, 0, 1]
windows3_y_step = [-1, -1, -1, 0, 0, 1, 1, 1]
windows5_x_step = [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
windows5_y_step = [-2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
min_const = 1e-4
max_const = 1e4


# 对应窗口的大小规模，获得所应相关的邻近点
def get_windows_step(scale=3):
    windows_x_step = []
    windows_y_step = []
    if scale == 3:
        windows_x_step = windows3_x_step
        windows_y_step = windows3_y_step
    elif scale == 5:
        windows_x_step = windows5_x_step
        windows_y_step = windows5_y_step
    return windows_x_step, windows_y_step


# 获得当前规模窗口所有的像素值的数组
def get_window_values(noise_img, i, j, scale=5):
    noise_image_row, noise_image_col = noise_img.shape
    windows_x_step, windows_y_step = get_windows_step(scale)
    current_pixel = float(noise_img[i, j])
    pixel_list = [current_pixel]
    for index in range(scale ** 2 - 1):
        neighbor_x = i + windows_x_step[index]
        neighbor_y = j + windows_y_step[index]
        if 0 <= neighbor_x < noise_image_row and 0 <= neighbor_y < noise_image_col:
            pixel_list.append(float(noise_img[neighbor_x, neighbor_y]))
    return pixel_list


# 累加前m个值
def accumulation(values, m=0):
    value = 0
    if m == 0:
        m = len(values)
    for index in range(m):
        if index < len(values):
            value += values[index]
        else:
            break
    return value


# 获得Rank-Ordered Absolute Difference Statistics
# 周围与居中像素之间的绝对差，该统计信息的较高的应为噪声而其较低值应对应于无噪声像素
def get_ROAD(noise_image, i, j, scale=3, m=4):
    windows_x_step, windows_y_step = get_windows_step(scale)
    noise_image_row, noise_image_col = noise_image.shape
    current_pixel = float(noise_image[i, j])
    road = []
    for index in range(scale ** 2 - 1):
        neighbor_x = i + windows_x_step[index]
        neighbor_y = j + windows_y_step[index]
        if 0 <= neighbor_x < noise_image_row and 0 <= neighbor_y < noise_image_col:
            road.append(np.abs(float(current_pixel) - float(noise_image[neighbor_x, neighbor_y])))
    road.sort()
    road_value = accumulation(road, m)
    return road_value


# 优化版计算效率：周围与居中像素之间的绝对差，该统计信息的较高的应为噪声而其较低值应对应于无噪声像素
def get_ROAD_optimization(noise_image, i, j, pixels_list, m=4):
    current_pixel = float(noise_image[i, j])
    road = np.abs(np.subtract(current_pixel, pixels_list))
    road.sort()
    road_value = accumulation(road, m)
    return road_value


# Rank-Ordered Logarithmic Difference Statistics
# 在均匀脉冲噪声中，周围像素与中心像素之差不是很明显，利用log对数扩大像素之间差异性问题
# ROLD利用对数时，当差异值为零时，会得到无穷小，这会导致累加产生问题都为无穷小；推导见公式
# D(x) = 1 + max{log₂x, -5} / 5
def get_ROLD(noise_image, i, j, scale=3, m=4):
    windows_x_step, windows_y_step = get_windows_step(scale)
    noise_image_row, noise_image_col = noise_image.shape
    current_pixel = float(noise_image[i, j])
    rold = []
    for index in range(scale ** 2 - 1):
        neighbor_x = i + windows_x_step[index]
        neighbor_y = j + windows_y_step[index]
        if 0 <= neighbor_x < noise_image_row and 0 <= neighbor_y < noise_image_col:
            road_value = np.abs(float(current_pixel) - float(noise_image[neighbor_x, neighbor_y]))
            # 保护log不会divide by zero encountered in log2
            if road_value == 0:
                value = -5
            else:
                value = max(np.log2(road_value), -5)
            current_rold = 1 + value / 5
            rold.append(current_rold)
    rold.sort()
    rold_value = accumulation(rold, m)
    return rold_value


# 优化版的ROLD
def get_ROLD_optimization(noise_image, i, j, pixels_list, m=4):
    current_pixel = float(noise_image[i, j])
    road_values = np.abs(np.subtract(current_pixel, pixels_list))
    rold_values = np.add(1, np.divide(np.maximum(np.log2(road_values), -5), 5))
    rold_values.sort()
    rold_value = accumulation(rold_values, m)
    return rold_value


# 开始计算噪声极值标准距离
min_pixel_value = 0
max_pixel_value = 255


# 计算极值与当前坐标像素之间的距离
def get_distance(current_pixel, min_pixel=min_pixel_value, max_pixel=max_pixel_value):
    return np.sqrt(np.square((max_pixel + min_pixel) / 2 - current_pixel) + np.square(max_pixel - min_pixel) / 12)


xita = 0.1
dext = get_distance(255)
dmed = get_distance((min_pixel_value + max_pixel_value) / 2)


# 获得像素为信号的概率
def get_signal_probability(noise_img, i, j):
    current_pixel = float(noise_img[i, j])
    dc = get_distance(current_pixel)
    return 1 - np.abs(dc - dmed) / np.abs(dext - dmed)


# Robust Outlyingness Ratio Feature(ROR)
# 鲁棒的离群率当前像素到其中值的距离与中值绝对差归一化（MADN）之比
# ROR值大意味着给定像素更有可能被脉冲噪声破坏
def get_ROR(current_pixel, values):
    med = np.median(values)
    madn, mad = get_MADN(values)
    ror = np.abs(current_pixel - med) / (madn + min_const)
    # if madn == 0:
    #     ror = 0
    return ror, madn, mad


# 绝对离差中位数
def get_MAD(values):
    return np.median(np.abs(values - np.median(values)))


# 归一化绝对偏差中位数
def get_MADN(values):
    mad = get_MAD(values)
    return mad / 0.6745, mad


# 绝对偏差
def get_MEDD(current_pixel, values):
    return np.abs(current_pixel - np.median(values))


def check_windows_enough(noise_image, i, j, m=4, scale=3):
    if (scale ** 2 - 1) < m:
        print("窗口的尺寸: %d * %d 不满足 m = %d" % (scale, scale, m))
        return False
    noise_image_row, noise_image_col = noise_image.shape
    padding = (scale - 1) / 2
    if (i - padding < 0 or i + padding >= noise_image_row) and (j - padding < 0 or j + padding >= noise_image_col):
        padding_x = min(i + padding, noise_image_row - 1) - max(i - padding, 0) + 1
        padding_y = min(j + padding, noise_image_col - 1) - max(j - padding, 0) + 1
        if (padding_x * padding_y - 1) < m:
            print("在窗口的尺寸: %d * %d的（%d, %d）不满足 m = %d" % (scale, scale, i, j, m))
            return False
    return True


# 提取图像的特征
def get_feature(noise_image):
    noise_image_row, noise_image_col = noise_image.shape
    feature_result = []
    log_util.info("开始统计图片的信息")
    for i in range(noise_image_row):
        for j in range(noise_image_col):
            current_pixel = float(noise_image[i, j])
            # check_windows_enough(noise_image,i,j)
            pixel3_list = get_window_values(noise_image, i, j, 3)
            pixel5_list = get_window_values(noise_image, i, j, 5)
            # road_value = get_ROAD(noise_image, i, j)
            road_value = get_ROAD_optimization(noise_image, i, j, pixel3_list[1:])
            # rold_value = get_ROLD(noise_image, i, j)
            rold_value = get_ROLD_optimization(noise_image, i, j, pixel3_list[1:])
            ror5_value, madn5_value, mad5_value = get_ROR(current_pixel, pixel5_list)
            mdd5_value = get_MEDD(current_pixel, pixel5_list)
            mad3_value = get_MAD(pixel3_list)
            mdd3_value = get_MEDD(current_pixel, pixel3_list)
            feature = [road_value, rold_value, ror5_value, madn5_value, mad5_value, mdd5_value, mad3_value, mdd3_value]
            feature_result.append(feature)
    log_util.info("完成统计特征信息")
    return feature_result


# 根据基因训练的结果转化为类别机制
# m表示为类别数目，g表示为基因的计算结果
def classify_result(g, m=2):
    # if g == float('inf'):
    #     g = max_const
    # elif g == float('-inf'):
    #     g = -max_const
    return int(round((m - 1) * (1 / (1 + np.exp(- g)))))


# 投票机制
def MV(gene_list):
    classifyList = {}
    for gene in gene_list:
        gene_result = classify_result(gene)
        if gene_result not in classifyList:
            classifyList[gene_result] = 1
        else:
            classifyList[gene_result] += 1
    key, value = sorted(classifyList.items(), key=lambda item: item[1], reverse=True)[0]
    # 备用方案
    #  np.argmax(np.bincount(b))
    return key


def classify_result_tensor_signal(g, m=2):
    length = len(g)
    result = torch.round((m - 1) * (1 / (1 + torch.exp(- g)))).int()
    vote_result = torch.ones(length, dtype=torch.int)
    for i in range(length):
        vote_result[i] = torch.argmax(torch.bincount(result[i]))
    return vote_result


def classify_result_tensor(g, m=2):
    return torch.round((m - 1) * (1 / (1 + torch.exp(- g)))).int()


def MV_Tensor(tensor_list):
    length = len(tensor_list)
    vote_list = classify_result_tensor(tensor_list)
    vote_result = torch.ones(length, dtype=torch.int)
    for i in range(length):
        vote_result[i] = torch.argmax(torch.bincount(vote_list[i]))
    return vote_result


def classify_result_numpy(g, m=2):
    return np.round((m - 1) * (1 / (1 + np.exp(- g))))


def MV_Numpy(numpy_array):
    length = len(numpy_array)
    vote_list = classify_result_numpy(numpy_array)
    vote_result = np.ones(length)
    for i in range(length):
        vote_result[i] = np.argmax(np.bincount(vote_list[i].astype(np.int)))
    return vote_result


# 多基因之间 “或” 机制限制
def mechanism_Numpy(numpy_array):
    length = numpy_array.shape[0]
    size = numpy_array.shape[1]
    vote_list = classify_result_numpy(numpy_array)
    vote_result = np.zeros(length)
    for i in range(size):
        vote_result = np.logical_or(vote_result, vote_list[:, i]).astype(np.int)
    return vote_result


