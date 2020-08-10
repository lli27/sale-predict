# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/7/1 19:34
@Desc   :
"""
from pydblite.pydblite import Base

class pydblite(object):
    pool = None

    """docstring for redis"""

    def __init__(self):
        pass

    def get_db(self, path, fields):
        """
        获取数据库
        :param path: 数据库路径
        :param fields: 数据库不存在时默认创建字段
        """
        py_base = Base(path, save_to_file=True)
        if not py_base.exists():
            py_base.create(*fields)
        else:
            py_base = py_base.open()
        return py_base