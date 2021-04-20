# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/8/31 17:43
@Desc   : 将多个模型的预测结果加权平均
"""
from src.sale.salemodel1 import salemodel1
from src.sale.salemodel2 import salemodel2
from src.utils.logfactory import LogFactory
from src.db import odps
from src.conf import Config
import pandas as pd
import traceback

class salepredict:
    def __init__(self, date, days):
        self.logging = LogFactory()
        self.odps = odps()
        self.config = Config().config_data
        self.date = date
        self.days = days

    def read_data(self, sql_cmd):
        """
        读取销售数据
        :return: DataFrame
        """
        self.logging.info(sql_cmd)
        try:
            with self.odps.conn.execute_sql(sql_cmd).open_reader() as reader:
                data = reader.to_pandas()
        except:
            self.logging.error(traceback.format_exc())
            return pd.DataFrame()
        self.logging.info("read_data success!")
        return data

    def model(self):
        # ---------------------------------数据准备--------------------------------------
        # 平滑后的时间序列
        sql_cmd = self.config.get('GET_NOR_DATA').format(self.date)
        data = self.read_data(sql_cmd)
        data["stat_date"] = pd.to_datetime(data["stat_date"])
        data.set_index('stat_date', inplace=True)
        data.index = data.index.to_period("D")
        # 权重数据
        sql_cmd1 = self.config.get('GET_FESTIVAL_WEIGHT').format(self.date)
        fest_weight = self.read_data(sql_cmd1)
        sql_cmd2 = self.config.get('GET_WEEK_WEIGHT').format(self.date)
        week_weight = self.read_data(sql_cmd2)
        # --------------------------------模型预测-------------------------------------
        # gbdt
        model1 = salemodel1(date=self.date, days=self.days)
        sal_pred_gbdt = model1.salepredict(data,fest_weight,week_weight)
        sal_pred_gbdt = sal_pred_gbdt[['distrib_code','season_code','new_line_code','stat_date','sal_pred_gbdt']]
        # arma
        model2 = salemodel2(date=self.date, days=self.days)
        sal_pred_arma = model2.salepredict(data,fest_weight,week_weight)
        if sal_pred_arma.empty:
            self.logging.error("sale_pred_arma is empty !")
        else:
            sal_pred_arma = sal_pred_arma[['distrib_code', 'season_code', 'new_line_code', 'stat_date', 'sal_pred_arma']]
        # 同比
        sql_cmd = self.config.get('GET_DATA_Y2Y_CLR').format(self.date,self.days//7+1)
        sal_pred_y2y = self.read_data(sql_cmd)
        # 乘以权重
        sal_pred_y2y = pd.merge(sal_pred_y2y, week_weight, on=['distrib_code', 'day_of_week'],how='left')
        sal_pred_y2y = pd.merge(sal_pred_y2y, fest_weight, on=['distrib_code', 'stat_date'],how='left')
        sal_pred_y2y['weight'] = sal_pred_y2y.apply(func=lambda x: x.fest_sal_weight if x.fest_sal_weight > 0 else x.sal_week_wgt,axis=1)
        sal_pred_y2y['sal_pred_y2y'] = sal_pred_y2y['sal_pred_y2y'] * sal_pred_y2y['weight']
        sal_pred_y2y = sal_pred_y2y[['distrib_code', 'season_code', 'new_line_code', 'stat_date','week_of_year', 'sal_pred_y2y']]
        return sal_pred_gbdt,sal_pred_arma,sal_pred_y2y

    def predict(self):
        """
        模型加权平均
        :return:
        """
        sal_pred_gbdt, sal_pred_arma, sal_pred_y2y = self.model()
        # 8月份夏季、秋季销售。
        data = pd.merge(sal_pred_gbdt, sal_pred_y2y, on=['stat_date', 'distrib_code', 'season_code', 'new_line_code'], how='left')
        data['sal_pred_y2y'] = data['sal_pred_y2y'].fillna(0)
        # 第几周
        first_week = data.week_of_year.min()
        # 若arma序列不稳定没有得到预测结果时，只用gbdt和同比加权平均。
        data = pd.merge(data, sal_pred_arma, on=['stat_date', 'distrib_code', 'season_code', 'new_line_code'], how='left')
        data1 = data[data['sal_pred_arma'].isna()]
        if len(data1) == 0:
            data.loc[data['season_code'].isin(['1','2']),'sal_pred_ensemble'] = data.loc[data['season_code'].isin(['1','2']),'sal_pred_arma']*2/3+data.loc[data['season_code'].isin(['1','2']),'sal_pred_y2y']/3
            data.loc[data['season_code'].isin(['4']),'sal_pred_ensemble'] = data.loc[data['season_code'].isin(['4']),'sal_pred_y2y']
            data.loc[data['season_code'].isin(['3']), 'sal_pred_ensemble'] = data.loc[data['season_code'].isin(['3']),'sal_pred_y2y']/3+data.loc[data['season_code'].isin(['3']),'sal_pred_gbdt']*2/3
        else:
            data2 = data[~data['sal_pred_arma'].isna()]
            data2.loc[data2['season_code'].isin(['1','2']),'sal_pred_ensemble'] = data2.loc[data2['season_code'].isin(['1','2']),'sal_pred_arma']*2/3+data2.loc[data2['season_code'].isin(['1','2']),'sal_pred_y2y']/3
            data2.loc[data2['season_code'].isin(['4']),'sal_pred_ensemble'] = data2.loc[data2['season_code'].isin(['4']),'sal_pred_y2y']
            data2.loc[data2['season_code'].isin(['3']), 'sal_pred_ensemble'] = data2.loc[data2['season_code'].isin(['3']),'sal_pred_y2y']/3+data2.loc[data2['season_code'].isin(['3']),'sal_pred_gbdt']*2/3
            # arma预测为空时，只用gbdt和同比加权平均。
            data1.loc[data1['season_code'].isin(['4']),'sal_pred_ensemble'] = data1.loc[data1['season_code'].isin(['4']),'sal_pred_y2y']
            data1.loc[data1['season_code'].isin(['1','2','3']), 'sal_pred_ensemble'] = data1.loc[data1['season_code'].isin(['1','2','3']),'sal_pred_y2y']/3+data1.loc[data1['season_code'].isin(['1','2','3']),'sal_pred_gbdt']*2/3
            data = data2.append(data1)
        return data

