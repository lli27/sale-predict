# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/6/30 13:57
@Desc   : odps
"""

import traceback
from odps import ODPS
from src.utils.logfactory import LogFactory
from src.conf.config import Config

class odps:
    """
    配置数据库
    """
    def __init__(self):
        """
        初始化
        """
        self.logging = LogFactory()
        self.config_data = Config().config_data
        self.conn = self.get_odps_conn()

    def get_odps_conn(self):
        """
        连接ODPS
        :return:
        """
        odps_config = self.config_data.get('ODPS')
        try:
            conn = ODPS(access_id=odps_config['USER'],
                     secret_access_key=odps_config['PASSWD'],
                     project=odps_config['DBNAME'],
                     endpoint=odps_config['URL'])
        except:
            self.logging.error(traceback.format_exc())
            raise
        return conn
