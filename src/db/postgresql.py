# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/7/8 18:35
@Desc   :
"""
import psycopg2
import traceback
import pandas as pd
from src.conf.config import Config
from src.utils.logfactory import LogFactory

class postgresql():
    def __init__(self):
        self.logging = LogFactory()
        self.config_data = Config().config_data
        self.conn = self.get_conn()

    def get_conn(self):
        pg_config = self.config_data.get('POSTGRESQL')
        conn = psycopg2.connect(host=pg_config['HOST'], port=pg_config['PORT'],
                                database=pg_config['DBNAME'], user=pg_config['USER'], password=pg_config['PASSWD'])
        return conn

    def get_pg_records(self, query, param=None):
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

    def update_pg(self, query, param=None):
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

    def pandas_readsql(self, sql, columns=None):
        """
        使用pandas读取
        :param sql: 查询语句
        :param columns: 查询列
        :return: dataframe
        """
        res = pd.read_sql(sql, con=self.conn, columns=columns)
        return res

    def close(self):
        self.conn.close()