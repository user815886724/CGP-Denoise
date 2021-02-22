import numpy as np
from util import img_util
import cv2

min_pixel = 0
max_pixel = 255
max_num = 512

init_windows_size = 3
beta = 1 / 4


# 自适应中值滤波器
def AdaptProcess(src, i, j, minSize, maxSize):
    filter_size = minSize

    kernelSize = filter_size // 2
    rio = src[i - kernelSize:i + kernelSize + 1, j - kernelSize:j + kernelSize + 1]
    minPix = np.min(rio)
    maxPix = np.max(rio)
    medPix = np.median(rio)
    zxy = src[i, j]

    if (medPix > minPix) and (medPix < maxPix):
        if (zxy > minPix) and (zxy < maxPix):
            return zxy
        else:
            return medPix
    else:
        filter_size = filter_size + 2
        if filter_size <= maxSize:
            return AdaptProcess(src, i, j, filter_size, maxSize)
        else:
            return medPix


def adapt_median_filter(img, minsize=3, maxsize=7):
    borderSize = maxsize // 2

    src = cv2.copyMakeBorder(img, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_REFLECT)

    for m in range(borderSize, src.shape[0] - borderSize):
        for n in range(borderSize, src.shape[1] - borderSize):
            src[m, n] = AdaptProcess(src, m, n, minsize, maxsize)

    dst = src[borderSize:borderSize + img.shape[0], borderSize:borderSize + img.shape[1]]
    return dst


# 选择窗口的最大规模
def choose_widows_size(noise_proportion):
    if 0 < noise_proportion <= 0.3:
        return 3
    elif 0.3 < noise_proportion <= 0.5:
        return 5
    elif 0.5 < noise_proportion <= 0.7:
        return 7
    else:
        return 9


# 获得噪声像素的数量
def get_noise_count(noise_map):
    noise = 0
    for n_maps in noise_map:
        for n_map in n_maps:
            if n_map == 1:
                noise += 1
    return noise


# 获得噪声在图像中的比例
def get_property(noise_map):
    total = len(noise_map) * len(noise_map[0])
    noise_count = get_noise_count(noise_map)
    return noise_count / total


# 寻找邻居
def search_neighbor(noise_image, noise_map, i, j, padding):
    """
    :param noise_image:
    :param noise_map:
    :param i: current pixel position x
    :param j: current pixel position y
    :param padding: windows size padding
    :return: window's pixels and window's signal count
    """
    noise_image_row, noise_image_col = noise_image.shape
    signal_count = 0
    windows_pixels = []
    for neighbor_x in range(-padding, padding + 1):
        for neighbor_y in range(-padding, padding + 1):
            # 若为当前位置像素则不操作
            if neighbor_x == 0 and neighbor_y == 0:
                continue
            current_x = i + neighbor_x
            current_y = j + neighbor_y
            if 0 <= current_x < noise_image_row and 0 <= current_y < noise_image_col:
                if noise_map[current_x, current_y] == 1:
                    continue
                else:
                    signal_count += 1
                windows_pixels.append(noise_image[current_x, current_y])
    return windows_pixels, signal_count


# 查找扩张的window的边缘计算
def search_window_edge(noise_image, noise_map, windows_pixels, signal_count, i, j, padding):
    noise_image_row, noise_image_col = noise_image.shape
    for neighbor_x in range(-padding, padding + 1):
        for neighbor_y in range(-padding, padding + 1):
            if neighbor_x != -padding and neighbor_x != padding:
                if neighbor_y != -padding and neighbor_y != padding:
                    continue
            current_x = i + neighbor_x
            current_y = j + neighbor_y
            if 0 <= current_x < noise_image_row and 0 <= current_y < noise_image_col:
                if noise_map[current_x, current_y] != 1:
                    signal_count += 1
                    windows_pixels.append(noise_image[current_x, current_y])
    return windows_pixels, signal_count


# 修复图像像素
def repair_image_pixel(noise_image, noise_map, image_noise_proportion, i, j, windows_size, max_windows_size):
    padding = int((windows_size - 1) / 2)
    # 扫描像素周围的邻居获得信号像素集合和数量
    windows_pixels, signal_count = search_neighbor(noise_image, noise_map, i, j, padding)
    # 计算window的信号比率是否满足要求
    signal_proportion = signal_count / (windows_size ** 2 - 1)
    while signal_proportion <= (1 - image_noise_proportion) * beta:
        if windows_size < max_windows_size:
            windows_size += 2
            padding += 1
            windows_pixels, signal_count = search_window_edge(noise_image, noise_map, windows_pixels, signal_count, i,
                                                              j, padding)
        else:
            break
    if len(windows_pixels) == 0:
        repair_pixel = noise_image[i, j]
    else:
        repair_pixel = np.median(windows_pixels)
    return repair_pixel


# 自适应中值滤波
def median_filter(noise_image, noise_map, noise_proportion, widows_size, max_windows_size):
    mean_image = noise_image.copy()
    for i, n_maps in enumerate(noise_map):
        for j, n_map in enumerate(n_maps):
            if n_map == 1:
                repair_pixel = repair_image_pixel(mean_image, noise_map, noise_proportion, i, j, widows_size,
                                                  max_windows_size)
                if np.isnan(repair_pixel):
                    print("error")
                mean_image[i, j] = repair_pixel
    return mean_image


# 检查边缘归一内部元素
def change_windows_edg(noise_image, i, j, windows_size, extremum_padding=0):
    row, col = noise_image.shape
    padding = int((windows_size - 1) / 2)
    window_x = []
    window_y = []
    window_pixels = []
    edge_pixels = []
    flag = False
    for neighbor_x in range(-padding, padding + 1):
        for neighbor_y in range(-padding, padding + 1):
            current_x = i + neighbor_x
            current_y = j + neighbor_y
            if 0 <= current_x < row and 0 <= current_y < col:
                # 窗口边缘
                if neighbor_x == -padding or neighbor_x == padding or neighbor_y == padding or neighbor_y == -padding:
                    edge_pixels.append(noise_image[current_x, current_y])
                    continue
                window_x.append(current_x)
                window_y.append(current_y)
                window_pixels.append(noise_image[current_x, current_y])

    if len(set(edge_pixels)) == 1:
        edge_pixel = edge_pixels[0]
        if edge_pixel not in window_pixels or (edge_pixel in window_pixels and len(set(window_pixels)) != 1):
            flag = True
            for index, window_pixel in enumerate(window_pixels):
                x = window_x[index]
                y = window_y[index]
                noise_image[x, y] = edge_pixel
        elif edge_pixel in window_pixels and len(set(window_pixels)) == 1:
            flag = True
    return noise_image, flag


# 从大到小收缩窗口，检查归一内部元素
def shrinkage_window(noise_image, max_windows_size, extremum_padding=0):
    shrinkage_image = noise_image.copy()
    row, col = shrinkage_image.shape
    for i in range(row):
        for j in range(col):
            for window_size in range(max_windows_size, 3, -1):
                shrinkage_image, flag = change_windows_edg(shrinkage_image, i, j, window_size, extremum_padding)
                if flag:
                    break
    return shrinkage_image


# 该变量由纹理决定
def get_texture_m(noise_image, i, j, n=4):
    neighbor_w = 2
    far_w = 0.5
    tmp_sum = neighbor_w * noise_image[i, j + 1] + neighbor_w * noise_image[i + 1, j] + 2 * far_w * noise_image[
        i + 1, j + 1]
    if tmp_sum >= n - 1:
        east_south_m = 1
        southeast_m = far_w
    else:
        east_south_m = neighbor_w
        southeast_m = far_w
        if noise_image[i, j + 1] == noise_image[i + 1, j + 1] or noise_image[i + 1, j] == noise_image[i + 1, j + 1]:
            east_south_m = 1
            southeast_m = far_w
    return east_south_m, southeast_m


# Minkowski distance（闵可夫斯基距离）
# 当p=1时则表示为曼哈顿距离
# 当p=2时则表示为欧几里得距离
def get_Minkowski_distance(x, y, i, j, p=1):
    return np.power(np.power(np.abs(x - i), p) + np.power(np.abs(y - j), p), 1 / p)


def weighted_mean_filter(noise_image, noise_map, n=4):
    weighted_mean_image = noise_image.copy()
    row, col = weighted_mean_image.shape
    weighted_mean_image = img_util.image_normalization(weighted_mean_image)
    for i in range(row):
        for j in range(col):
            if noise_map[i, j] == 1 and i + 1 < row and j + 1 < col:
                east_south_m, southeast_m = get_texture_m(weighted_mean_image, i, j, n)
                south_d = get_Minkowski_distance(i, j + 1, i, j)
                east_d = get_Minkowski_distance(i + 1, j, i, j)
                southeast_d = get_Minkowski_distance(i + 1, j + 1, i, j)
                south_w = east_south_m * south_d
                east_w = east_south_m * east_d
                southeast_w = southeast_m * southeast_d
                new_pixel = (south_w * weighted_mean_image[i, j + 1] + east_w * weighted_mean_image[
                    i + 1, j] + southeast_w *
                             weighted_mean_image[i + 1, j + 1]) / (n - 1)
                weighted_mean_image[i, j] = new_pixel
    weighted_mean_image = img_util.recover_normalization(weighted_mean_image)
    return weighted_mean_image


# 检查当前是否噪声
def check_noise(noise_map_image, i, j):
    if noise_map_image[i, j] == 1:
        return True
    else:
        return False


def edge_preserving(noise_image, noise_map_image, i, j):
    noise_image_row, noise_image_col = noise_image.shape
    current_pixel = noise_image[i, j]
    if not check_noise(noise_map_image, i,
                       j) or i - 1 < 0 or i + 1 >= noise_image_row or j - 1 < 0 or j + 1 >= noise_image_col:
        return current_pixel
    a = int(noise_image[i - 1, j - 1])
    b = int(noise_image[i - 1, j])
    c = int(noise_image[i - 1, j + 1])
    d = int(noise_image[i, j - 1])
    e = int(noise_image[i, j + 1])
    f = int(noise_image[i + 1, j - 1])
    g = int(noise_image[i + 1, j])
    h = int(noise_image[i + 1, j + 1])
    D1 = np.abs(a - h)
    D2 = np.abs(b - g)
    D3 = np.abs(c - f)
    D4 = np.abs(e - d)
    D5 = (np.abs(d - h) + np.abs(a - e)) / 2
    D6 = (np.abs(a - g) + np.abs(b - h)) / 2
    D7 = (np.abs(b - f) + np.abs(c - g)) / 2
    D8 = (np.abs(c - d) + np.abs(e - f)) / 2
    # 数组规模
    tmp_result = [D1, D2, D3, D4, D5, D6, D7, D8]
    min_D = min(tmp_result)
    if min_D == max_num:
        return current_pixel
    else:
        if min_D == D1:
            return int((a + h) / 2)
        elif min_D == D2:
            return int((b + g) / 2)
        elif min_D == D3:
            return int((c + f) / 2)
        elif min_D == D4:
            return int((e + d) / 2)
        elif min_D == D5:
            return int((a + d + h + e) / 4)
        elif min_D == D6:
            return int((a + g + b + h) / 4)
        elif min_D == D7:
            return int((b + f + c + g) / 4)
        elif min_D == D8:
            return int((c + d + e + f) / 4)


# 边缘保护滤波器
def edge_preserving_filter(noise_image, noise_map_image):
    edge_preserving_image = noise_image.copy()
    row, col = edge_preserving_image.shape
    for i in range(row):
        for j in range(col):
            edge_preserving_image[i, j] = edge_preserving(noise_image, noise_map_image, i, j)
    return edge_preserving_image
