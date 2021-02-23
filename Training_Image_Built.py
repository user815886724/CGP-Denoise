from util import img_util,setting_util

if __name__ == "__main__":
    # 生成训练集
	# create paper noise
    img_util.crate_train_data(setting_util.IN_DIR, setting_util.OUT_DIR, setting_util.OUT_MAP_DIR,
                              start=1, end=2, proportion=0.4, type="paper")
	# create random noise
	img_util.crate_train_data(setting_util.IN_DIR, setting_util.OUT_DIR, setting_util.OUT_MAP_DIR,
                              start=2, end=3, proportion=0.4, type="random")