import pickle
from util import log_util
import os


# 删除某个文件
def remove_file(path):
    if os.path.exists(path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(path)


# 保存模型
def save_model(model, path):
    log_util.info("正在保存 %s 模型" % path)
    remove_file(path)
    pickle.dump(model, open(path, 'wb'))
    log_util.info("%s 模型保存成功" % path)


# 加载模型
def load_model(path):
    log_util.info("正在加载 %s 模型中" % path)
    model = pickle.load(open(path, 'rb'))
    log_util.info("加载 %s 模型成功" % path)
    return model


# 使用种群population获得最优个体的表达式表达
def get_best_display(pop):
    return pop.champion.to_sympy()


# 返回获得最佳函数
def get_best_function(pop):
    return pop.champion.to_func()


# 使用最佳函数来运算结果
def get_best_function_result(pop, args):
    return pop.champion.to_func()(args)


def get_detection_information(results, noise_data, noise_map):
    detection_information = {}
    detection_result = []
    noise_information = []
    map_information = []
    for index, result in enumerate(results):
        if result == 0:
            detection_result.append(result)
            noise_information.append(noise_data[index])
            map_information.append(noise_map[index])
    detection_information["detection_result"] = detection_result
    detection_information["noise_information"] = noise_information
    detection_information["map_information"] = map_information
    return detection_information


def get_detection_information1(results, noise_data, noise_map):
    detection_information = {}
    detection_result = []
    noise_information = []
    map_information = []
    for index, result in enumerate(results):
        if result == 1:
            detection_result.append(result)
            noise_information.append(noise_data[index])
            map_information.append(noise_map[index])
    detection_information["detection_result"] = detection_result
    detection_information["noise_information"] = noise_information
    detection_information["map_information"] = map_information
    return detection_information
