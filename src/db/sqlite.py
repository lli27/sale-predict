# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/7/1 18:44
@Desc   : sqlite
"""
import sqlite3
import traceback
import os
from src.utils.logfactory import LogFactory

class sqlite:
    def __init__(self, file_path='sale.db'):
        """
        :param file_path: db文件路径。默认存储在内存中。
        """
        file_path = os.path.join('data', file_path)
        self.logging = LogFactory()
        self.conn = sqlite3.connect(file_path)

    def get_sqlite_records(self, query, param=None):
        """
        从sqlite数据库获取数据
        :param query: 查询语句
        :param param: 查询参数
        :return: 查询结果（list）
        """
        cursor = self.conn.cursor()
        try:
            if param is None:
                cursor.execute(query)
            else:
                cursor.execute(query, param)
            records = cursor.fetchall()
        except:
            records = []
            self.logging.error(traceback.format_exc())
        cursor.close()
        return records

    def update_sqlite(self, query, param=None):
        """
        执行增删改类查询sql语句
        :param query: 查询语句
        :param param: 查询参数
        :return: 执行结果（1-成功；0-失败）
        """
        cursor = self.conn.cursor()
        try:
            if param is None:
                cursor.execute(query)
            else:
                cursor.execute(query, param)
            cursor.close()
            self.conn.commit()
            return 1
        except:
            self.logging.error(traceback.format_exc())
            cursor.close()
            self.conn.rollback()
            return 0

    def close(self):
        self.conn.close()