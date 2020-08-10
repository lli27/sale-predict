# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/6/30 13:58
@Desc   : 记录日志
"""
import logging as log
import os
import datetime

class LogFactory():
    """
    docstring for logging
    """
    def __init__(self):
        if not os.path.isdir('log'):
            os.makedirs('log')
        self.dw = datetime.date.today().strftime('%Y%m%d')

    def write(self, level, msg):
        # create logger
        logger = log.getLogger()
        if level == 'error':
            logger.setLevel(log.ERROR)
        elif level == 'critical':
            logger.setLevel(log.CRITICAL)
        elif level == 'exception' or level == 'warn' or level == 'warning':
            logger.setLevel(log.WARNING)
        elif level == 'debug':
            logger.setLevel(log.DEBUG)
        else:
            logger.setLevel(log.INFO)

        # create handler
        filename = 'log/SalePredict_{}.{}'.format(self.dw, level)  # 这里将日志统一写入SalePredict.log
        fh = log.FileHandler(filename, mode='a')
        formatter = log.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)

        # add handler
        logger.addHandler(fh)
        getattr(logger, level)(msg)

        logger.removeHandler(fh)

    def critical(self, msg):
        self.write('critical', msg)

    def error(self, msg):
        self.write('error', msg)

    def exception(self, msg):
        self.write('exception', msg)

    def warning(self, msg):
        self.write('warning', msg)

    def warn(self, msg):
        self.write('warn', msg)

    def info(self, msg):
        self.write('info', msg)

    def debug(self, msg):
        self.write('debug', msg)
