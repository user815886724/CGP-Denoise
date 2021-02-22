import math
from skimage import img_as_float
from skimage.metrics import mean_squared_error as mse
import cv2
from util import log_util, math_util, cgp_util
import numpy as np
import os
import glob
from PIL import Image
import skimage

# 生成噪声地图的背景颜色
image_background = 127


# 将pgm转化为jpg
def batch_image(in_dir, out_dir):
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)

    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    count = 0
    for files in glob.glob(in_dir + '/*'):
        filepath, filename = os.path.split(files)

        out_file = filename.split(".")[0] + '.jpg'
        # print(filepath,',',filename, ',', out_file)
        im = Image.open(files)
        new_path = os.path.join(out_dir, out_file)
        print(count, ',', new_path)
        count = count + 1
        im.save(os.path.join(out_dir, out_file))


# 增加脉冲噪声
def add_impulse_noise(origin_image, proportion, type="random"):
    """
    :param origin_image: 原图像
    :param proportion: 噪声污染比例
    :param type: 如果值等于random则随机生成噪声值，如果值为paper则生成椒盐噪声值
    :return: 噪声污染的图像，噪声图像
    """
    impulse_noise_image = origin_image.copy()
    # 获得图像的row和col
    origin_image_row, origin_image_col = origin_image.shape
    # 随机选取坐标轴
    X = np.random.randint(origin_image_row, size=(int(proportion * origin_image_row * origin_image_col)))
    Y = np.random.randint(origin_image_col, size=(int(proportion * origin_image_row * origin_image_col)))
    # 添加0或255脉冲噪声像素点
    if type == "random":
        impulse_noise_image[Y, X] = np.random.randint(0, high=255,
                                                      size=(int(proportion * origin_image_row * origin_image_col)))
    else:
        impulse_noise_image[Y, X] = np.random.choice([0, 255],
                                                     size=(int(proportion * origin_image_row * origin_image_col)))
    # 噪声容器图像；其中ones_like代表将数组的每个元素都变为1
    impulse_noise = np.ones_like(impulse_noise_image) * image_background
    # 防止生成的噪声与底色重复，既是噪音也是原像素，导致学习效率下降
    for index in range(int(proportion * origin_image_row * origin_image_col)):
        if origin_image[Y[index], X[index]] != impulse_noise_image[Y[index], X[index]]:
            impulse_noise[Y[index], X[index]] = impulse_noise_image[Y[index], X[index]]
    # impulse_noise[Y, X] = impulse_noise_image[Y, X]
    return impulse_noise_image, impulse_noise


# 保存图片
def save_img(image, fileName="out.jpg"):
    # 无损
    cv2.imwrite(fileName, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


# 保存噪声相片
def save_noise_image(origin_image, file_name, file_path="noise_image"):
    log_util.info("开始保存 %s 噪声相片" % file_name)
    if not os.path.exists(file_path):
        print(file_path, 'is not existed.')
        log_util.info("%s 不存在，开始创建该文件夹" % file_path)
        os.mkdir(file_path)
    save_img(origin_image, os.path.join(file_path, file_name))
    log_util.info("%s 保存成功" % file_name)


# 获得检测脉冲噪声Map的信息
def get_noise_map(noise_map_image):
    row, col = noise_map_image.shape
    noise_map = []
    for i in range(row):
        for j in range(col):
            noise_result = 0
            if noise_map_image[i, j] != image_background:
                noise_result = 1
            noise_map.append(noise_result)
    return noise_map


# 创建训练集
def crate_train_data(in_dir, out_dir, out_noise_dir, start=0, end=0, proportion=0.1, type="random"):
    if start == 0 and end == 0:
        log_util.info("开始转化所有的%s文件夹下的所有图片" % in_dir)
        print("开始转化所有的%s文件夹下的所有图片" % in_dir)
    elif start == 0:
        log_util.info("开始转化%s文件夹下%s张图片" % (in_dir, end))
        print("开始转化%s文件夹下%s张图片" % (in_dir, end))
    elif end == 0:
        log_util.info("开始转化%s文件夹下第%s张图片开始" % (in_dir, start))
        print("开始转化%s文件夹下第%s张图片开始" % (in_dir, start))
    else:
        log_util.info("开始转化%s文件夹下第%s到第%s张图片" % (in_dir, start, end))
        print("开始转化%s文件夹下第%s到第%s张图片" % (in_dir, start, end))
    if not os.path.exists(in_dir):
        log_util.info(in_dir + 'is not existed.')
        print(in_dir, 'is not existed.')
        return -1
    count = 0
    for files in glob.glob(in_dir + '/*'):
        if (start == 0 or count >= start - 1) and (end == 0 or count < end):
            filepath, filename = os.path.split(files)
            log_util.info("开始转化第%s张图片——%s" % (count + 1, filename))
            origin_image = cv2.imread(files)
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2GRAY)
            noise_image, noise_plate = add_impulse_noise(origin_image, proportion, type)
            out_file = filename.split(".")[0] + '_noise.jpg'
            save_noise_image(noise_image, out_file, out_dir)
            out_plate_file = filename.split(".")[0] + "_noise_map.jpg"
            save_noise_image(noise_plate, out_plate_file, out_noise_dir)
            log_util.info("成功转化第%s张图片——%s" % (count+1, filename))
            print("成功转化第%s张图片——%s" % (count+1, filename))
        count += 1


# 加载图像的特征
def get_img_feature(noise_dir, noise_map_dir, start=0, end=0):
    if not os.path.exists(noise_dir):
        log_util.info(noise_dir + 'is not existed.')
        print(noise_dir, 'is not existed.')
        return -1
    if not os.path.exists(noise_map_dir):
        log_util.info(noise_map_dir + 'is not existed.')
        print(noise_map_dir, 'is not existed.')
        return -1
    noise_files = glob.glob(noise_dir + '/*')
    noise_map_files = glob.glob(noise_map_dir + '/*')
    if len(noise_files) != len(noise_map_files):
        log_util.info(noise_dir + '和' + noise_map_dir + '图片数量不一样')
        print(noise_dir + '和' + noise_map_dir + '图片数量不一样')
        return -1
    elif len(noise_files) < start and end > len(noise_files):
        log_util.info('数量超出限制')
        print('数量超出限制')
        return -1
    noise_result = []
    noise_map_result = []
    index = 0
    if end == 0:
        end = len(noise_map_files)
    if start != 0:
        index = start - 1
    while index < end:
        noise_image = cv2.imread(noise_files[index])
        noise_map_image = cv2.imread(noise_map_files[index])
        noise_image = cv2.cvtColor(noise_image, cv2.COLOR_RGB2GRAY)
        noise_image = image_normalization(noise_image)
        noise_map_image = cv2.cvtColor(noise_map_image, cv2.COLOR_RGB2GRAY)
        log_util.info("开始第%s张图像统计特征" % (index + 1))
        print("开始第%s张图像统计特征" % (index + 1))
        feature_info = math_util.get_feature(noise_image)
        log_util.info("开始第%s张噪声统计特征" % (index + 1))
        print("开始第%s张噪声统计特征" % (index + 1))
        noise_map = get_noise_map(noise_map_image)
        noise_result += feature_info
        noise_map_result += noise_map
        index += 1
    return noise_result, noise_map_result


# 加载图像特征数据到数据模型中保存下来
def load_data(noise_dir, noise_map_dir, data_path, data_map_path, start=0, end=0):
    noise_result, noise_map_result = get_img_feature(noise_dir, noise_map_dir, start, end)
    cgp_util.save_model(noise_result, data_path)
    cgp_util.save_model(noise_map_result, data_map_path)


# 图像归一化
def image_normalization(image):
    image = np.float32(image)
    dst = np.zeros(image.shape, dtype=np.float32)
    # alpha和beta的意义是，alpha:range normalization模式的最小值
    # beta: range
    # normalization模式的最大值，不用于norm
    # normalization(范数归一化）模式
    cv2.normalize(image, dst=dst, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    return dst


# 图像归一化后恢复
def recover_normalization(dst):
    return np.uint8(dst * 255)


# 图片RMSE计算
def rmse_diff(image1, image2):
    """Calculates the root mean square error (RSME) between two images"""
    return math.sqrt(mse(img_as_float(image1), img_as_float(image2)))


# 将两张图片大小调为一致，大小按照最小对的来调整
def identity(image1, image2):
    image1_row, image1_col = image1.shape[:2]
    image2_row, image2_col = image2.shape[:2]
    min_row = min(image1_row, image2_row)
    min_col = min(image1_col, image2_col)
    if image1_row > min_row or image1_col > min_col:
        image1 = cv2.resize(image1, (min_row, min_col), interpolation=cv2.INTER_CUBIC)
    if image2_row > min_row or image2_col > min_col:
        image2 = cv2.resize(image2, (min_row, min_col), interpolation=cv2.INTER_CUBIC)
    return image1, image2


# 获得图像的（PSNR）峰值信噪比
def get_PSNR(noise_image, origin_image, max_size=255):
    psnr = skimage.metrics.peak_signal_noise_ratio(origin_image, noise_image, data_range=max_size)
    return psnr


# 获得图像的（SSIM）结构相似性
def get_SSIM(noise_image, origin_image, max_size=255):
    ssim = skimage.metrics.structural_similarity(origin_image, noise_image, data_range=max_size)
    return ssim

