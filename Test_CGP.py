from util import cgp_util, setting_util, math_util, img_util, log_util, filter_util
from set.function_set_old import Log, Exp, Sin, Iflte, ProtectRoot, ProtectDiv, ConstantRandFloat, Cos, Square
import os
import glob
import cv2
import cgp
import warnings
import numpy as np
import random

warnings.filterwarnings('ignore')
min_const = 1e-4
max_const = 1e4


# 根据基因训练的结果转化为类别机制
# m表示为类别数目，g表示为基因的计算结果
def classify_result(g, m=2):
    if g == float('inf'):
        g = max_const
    elif g == float('-inf'):
        g = -max_const
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
    return key


# 加载图像的特征
def get_img_feature(noise_dir, start=0, end=0):
    if not os.path.exists(noise_dir):
        log_util.info(noise_dir + 'is not existed.')
        print(noise_dir, 'is not existed.')
        return -1
    noise_files = glob.glob(noise_dir + '/*')
    noise_result = []
    index = 0
    if end == 0:
        end = len(noise_files)
    if start != 0:
        index = start - 1
    while index < end:
        noise_image = cv2.imread(noise_files[index])
        noise_image = cv2.cvtColor(noise_image, cv2.COLOR_RGB2GRAY)
        noise_image = img_util.image_normalization(noise_image)
        log_util.info("开始第%s张图像统计特征" % (index + 1))
        print("开始第%s张图像统计特征" % (index + 1))
        feature_info = math_util.get_feature(noise_image)
        noise_result += feature_info
        index += 1
    return noise_result


def get_img_index(in_dir, index):
    noise_files = glob.glob(in_dir + '/*')
    noise_image = cv2.imread(noise_files[index - 1], cv2.IMREAD_GRAYSCALE)
    return noise_image


def estimate(f, noise_data_numpy):
    numpy_result = f(noise_data_numpy)
    mv_result = math_util.mechanism_Numpy(numpy_result)
    return mv_result


if __name__ == "__main__":
    model = cgp_util.load_model('model/model_champion_491.pkl')
    # 规定检测文件夹下第几张图片
    img_index = 8
    noise_data = get_img_feature(setting_util.TEST_DIR, img_index, img_index)
    noise_data_numpy = np.array(noise_data)
    noise_img = get_img_index(setting_util.TEST_DIR, img_index)
    noise_row, noise_col = noise_img.shape
    noise_repair_map = np.zeros((noise_row, noise_col))
    # 每个方程进行预估噪声结果
    result = estimate(model.to_numpy(),noise_data_numpy)
    for i, vote in enumerate(result):
        if vote == 1:
            current_x = int(i / noise_col)
            current_y = i % noise_col
            noise_repair_map[current_x, current_y] = 1

    # 估计污染浓度且获取最大过滤窗口
    proportion = filter_util.get_property(noise_repair_map)
    max_size = filter_util.choose_widows_size(proportion)
    print("开始自适应中值滤波")
    img = filter_util.median_filter(noise_img, noise_repair_map, proportion, filter_util.init_windows_size, max_size)
    print("开始收缩窗口边缘")
    shrinkage_img = filter_util.shrinkage_window(img, max_size)
    print("开始边缘保护滤波")
    edge_img = filter_util.edge_preserving_filter(shrinkage_img, noise_repair_map)
    cv2.imshow("noise", noise_img)
    cv2.imshow("denoise image", edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
