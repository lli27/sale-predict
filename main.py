# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/9/3 20:42
@Desc   :
"""
from src.utils.salepredict import salepredict
from src.utils.costpredict import costpredict
from src.utils.logfactory import LogFactory
from src.db import odps
from src.conf import Config
from odps import DataFrame
import pandas as pd
import traceback
import datetime

class main:
    def __init__(self):
        self.logging = LogFactory()
        self.odps = odps()
        self.config = Config().config_data
        self.date = (datetime.date.today()+datetime.timedelta(days=-1)).strftime('%Y%m%d') # 业务日期T-1
        self.days = 28 # 预测四周

    def write_to_db(self, data):
        """
        将最终结果写入数据库
        :param data:
        :return: （1-成功；0-失败）
        """
        if data.empty:
            self.logging.error("write_to_db fail！predict data is empty!")
            return 0
        else:
            try:
                data['ds'] = data['stat_date']
                data['dw_date'] = datetime.datetime.now()
                DataFrame(data).persist(name='ads_fd_sale_pred_distrib_1d', overwrite=True, partitions=['ds'],
                                        odps=self.odps.conn)
            except:
                self.logging.error(traceback.format_exc())
                return 0
        self.logging.info("write_to_db success！")
        return 1

    def main(self):
        """
        得到销售成本预测、销售额预测，并写入数据库
        :return:
        """
        self.logging.info("begin running sale predict！")
        cost_data = costpredict(self.date, self.days).predict()
        sale_data = salepredict(self.date, self.days).predict()
        data = pd.merge(cost_data, sale_data, on=['distrib_code','season_code','new_line_code','stat_date','week_of_year'], how='inner')
        return self.write_to_db(data)

if __name__=="__main__":
    main().main()