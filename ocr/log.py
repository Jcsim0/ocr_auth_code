import logging

"""
usage
    第一次使用时需到项目根目录下创建log文件夹
    仅作用于Django内实例化的app
    在需要到的py文件导入
    from ocr.log import logger
    logger.error    记录错误信息
    logger.info     记录数据信息
    logger.warn     记录提示信息
"""
# 创建基础的log(记录状态信息以及数据信息)
logger = logging.getLogger(__name__)
