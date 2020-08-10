# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/7/1 18:04
@Desc   : 读取配置文件
"""
import yaml
import traceback
from src.utils.logfactory import LogFactory

class Config:
    def __init__(self):
        """
        初始化
        """
        self.logging = LogFactory()
        self.config_data = self.load_yaml('conf/conf.yaml')

    def load_yaml(self, yaml_file):
        """
        读取配置文件
        :param yaml_file: 配置文件路径
        :return:
        """
        yaml_stream = open(yaml_file, 'rb')
        try:
            config_data = yaml.load(yaml_stream, Loader=yaml.FullLoader)
        except:
            self.logging.error(traceback.format_exc())
            raise
        return config_data
