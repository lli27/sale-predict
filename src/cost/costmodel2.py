# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/6/30 13:28
@Desc   :
"""
from src.utils.logfactory import LogFactory
from src.db import odps
from src.conf import Config
import pandas as pd
import numpy as np
import traceback
from multiprocessing.dummy import Pool as ThreadPool
from statsmodels.tsa.arima_model import ARMA

class costmodel2:

    def __init__(self, date, days):
        self.logging = LogFactory()
        self.odps = odps()
        self.config = Config().config_data
        self.date = date
        self.days = days

    def get_dsitrib_data(self,data):
        """
        预测-多进程
        :return: list, list的元素为DataFrame
        """
        try:
            data['cost_amt_clr_log'] = data['cost_amt_clr'].apply(lambda x: np.log(x) if x > 1 else 1e-10)
            # 得到具体分区的时间序列
            pool = ThreadPool(4)
            yhat = pool.starmap(self.algo_arma, zip(data.groupby(by=['distrib_code','season_code','new_line_code'])))
            pool.close()
            pool.join()
        except:
            self.logging.error(traceback.format_exc())
            return []
        return yhat

    def algo_arma(self, grouper):
        """
        时间序列算法
        :param grouper:
        :return: DataFrame
        """
        try:
            data = grouper[1].cost_amt_clr_log.sort_index()
            model = ARMA(data,(2,1)).fit()
            yhat = model.predict(start=len(data), end=len(data) + self.days - 1)
            yhat = pd.DataFrame(yhat, columns=['yhat'])
            yhat['distrib_code'] = grouper[0][0]
            yhat['season_code'] = grouper[0][1]
            yhat['new_line_code'] = grouper[0][2]
            yhat['stat_date'] = yhat.index.to_timestamp()
        except:
            # self.logging.error("distrib_code={0},season_code={1},new_line_code={2},arma cost predict fail!".foramt(grouper[0][0],grouper[0][1],grouper[0][2]))
            self.logging.error(traceback.format_exc())
            return pd.DataFrame()
        self.logging.info("algo_time_series success!")
        return yhat

    def salepredict(self, data,fest_weight,week_weight):
        """
        输出最终预测结果
        :return: DataFrame
        """
        yhat = self.get_dsitrib_data(data)
        num = len(yhat)
        if num == 0:
            self.logging.error("salepredict fail！")
            return pd.DataFrame(columns=['distrib_code', 'season_code', 'new_line_code', 'stat_date', 'cost_pred_arma'])
        else:
            try:
                yhat_full = pd.DataFrame(columns=['yhat','distrib_code','season_code','new_line_code','stat_date'])
                for i in range(num):
                    yhat_full = pd.concat([yhat_full, yhat[i]],ignore_index=True)
                yhat_full['day_of_week'] = yhat_full['stat_date'].dt.dayofweek.values
                yhat_full['stat_date'] = yhat_full['stat_date'].dt.strftime('%Y%m%d').values
                yhat_full['yhat'] = yhat_full['yhat'].apply(lambda x: np.exp(x))
                # 乘以权重
                yhat_full = pd.merge(yhat_full, week_weight, on=['distrib_code','day_of_week'], how = 'left')
                yhat_full = pd.merge(yhat_full, fest_weight, on=['distrib_code','stat_date'], how = 'left')
                yhat_full['weight'] = yhat_full.apply(func=lambda x: x.fest_cost_weight if x.fest_cost_weight > 0 else x.cost_week_wgt,
                                            axis=1)
                yhat_full['cost_pred_arma'] = yhat_full['yhat'] * yhat_full['weight']
            except:
                self.logging.error(traceback.format_exc())
                return pd.DataFrame()
        self.logging.info("salepredict success！")
        return yhat_full
