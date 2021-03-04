import logging

# 写日志配置，注意还修改原logging的__init__方法
logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='new.log',
                    filemode='w',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    # a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    # 日志格式
                    )


def info(message):
    return logging.info(message)


def error(message):
    return logging.error(message)


def debug(message):
    return logging.debug(message)
