from util import img_util, cgp_util, setting_util
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    img_util.load_data(setting_util.NOISE_PATH, setting_util.NOISE_MAP_PATH, setting_util.DATA_PATH,
                       setting_util.DATA_MAP_PATH, start=0, end=0)

    feature_info = cgp_util.load_model(setting_util.DATA_PATH)
    # print(feature_info)
    noise_map = cgp_util.load_model(setting_util.DATA_MAP_PATH)
    # print(noise_map)
    print("完成提取操作")
