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
import matplotlib.pyplot as plt
import numpy as np

# 更改执行目录为src
import os
file_path=os.path.realpath(__file__)
parent_dir = os.path.dirname(os.path.dirname(file_path))
os.chdir(parent_dir)

class saletest:
    def __init__(self):
        self.logging = LogFactory()
        self.odps = odps()
        self.config = Config().config_data
        self.date = '20200823'#训练时间 (datetime.date.today()+datetime.timedelta(days=-1)).strftime('%Y%m%d') # 业务日期T-1
        self.days = 28

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

    def main(self):
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
        sal_pred_arma = sal_pred_arma[['distrib_code', 'season_code', 'new_line_code', 'stat_date', 'sal_pred_arma']]
        # 同比
        sql_cmd = self.config.get('GET_DATA_Y2Y_CLR').format(self.date,self.days//7+1)
        sal_pred_y2y = self.read_data(sql_cmd)
        # 乘以权重
        sal_pred_y2y = pd.merge(sal_pred_y2y, week_weight, on=['distrib_code', 'season_code', 'new_line_code', 'day_of_week'],how='left')
        sal_pred_y2y = pd.merge(sal_pred_y2y, fest_weight, on=['distrib_code', 'season_code', 'new_line_code', 'stat_date'],how='left')
        sal_pred_y2y['weight'] = sal_pred_y2y.apply(func=lambda x: x.fest_sal_weight if x.fest_sal_weight > 0 else x.sal_week_wgt,axis=1)
        sal_pred_y2y['sal_pred_y2y'] = sal_pred_y2y['sal_pred_y2y'] * sal_pred_y2y['weight']
        sal_pred_y2y = sal_pred_y2y[['distrib_code', 'season_code', 'new_line_code', 'stat_date','week_of_year', 'sal_pred_y2y']]
        # --------------------------------真实数据-测试用---------------------------------------
        sql_cmd = self.config.get('GET_REAL_DATA').format(self.date)
        real_data = self.read_data(sql_cmd)
        return sal_pred_gbdt,sal_pred_arma,sal_pred_y2y,real_data

if __name__ == "__main__":
    model = saletest()
    sal_pred_gbdt, sal_pred_arma, sal_pred_y2y, real_data = model.main()
    # 8月份夏季、秋季销售。
    # real_data = real_data[real_data['season_code'].isin(['2','3'])]
    data = pd.merge(real_data,sal_pred_gbdt,on=['stat_date','distrib_code','season_code','new_line_code'],how='inner')
    data = pd.merge(data,sal_pred_arma,on=['stat_date','distrib_code', 'season_code', 'new_line_code'],how='inner')
    data = pd.merge(data,sal_pred_y2y,on=['stat_date','distrib_code', 'season_code', 'new_line_code'],how='inner')
    # 第几周
    first_week = data.week_of_year_x.sort_values()[0]
    # 夏季
    # data.loc[(data['season_code'].isin(['1','2']))&(data['week_of_year_x']<=first_week),'model_ensemble1'] = data.loc[(data['season_code'].isin(['1','2']))&(data['week_of_year_x']<=first_week),'sal_pred_gbdt']*2/3+data.loc[(data['season_code'].isin(['1','2']))&(data['week_of_year_x']<=first_week),'sal_pred_y2y']/3
    # data.loc[(data['season_code'].isin(['1','2']))&(data['week_of_year_x']>first_week),'model_ensemble1'] = data.loc[(data['season_code'].isin(['1','2']))&(data['week_of_year_x']>first_week),'sal_pred_y2y']/2+data.loc[(data['season_code'].isin(['1','2']))&(data['week_of_year_x']>first_week),'sal_pred_arma']/3++data.loc[(data['season_code'].isin(['1','2']))&(data['week_of_year_x']>first_week),'sal_pred_gbdt']/6
    # data.loc[data['season_code'].isin(['3','4']),'model_ensemble1'] = data.loc[data['season_code'].isin(['3','4']),'sal_pred_y2y']*2/3+data.loc[data['season_code'].isin(['3','4']),'sal_pred_arma']/3
    # # 春、夏、秋
    # data.loc[(data['season_code'].isin(['1','2','3']))&(data['week_of_year_x']<=first_week),'model_ensemble1'] = data.loc[(data['season_code'].isin(['1','2','3']))&(data['week_of_year_x']<=first_week),'sal_pred_gbdt']*2/3+data.loc[(data['season_code'].isin(['1','2','3']))&(data['week_of_year_x']<=first_week),'sal_pred_y2y']/3
    # data.loc[(data['season_code'].isin(['1','2','3']))&(data['week_of_year_x']>first_week),'model_ensemble1'] = data.loc[(data['season_code'].isin(['1','2','3']))&(data['week_of_year_x']>first_week),'sal_pred_gbdt']/6+data.loc[(data['season_code'].isin(['1','2','3']))&(data['week_of_year_x']>first_week),'sal_pred_arma']/2+data.loc[(data['season_code'].isin(['1','2','3']))&(data['week_of_year_x']>first_week),'sal_pred_y2y']/3
    # data.loc[data['season_code'].isin(['4']),'model_ensemble1'] = data.loc[data['season_code'].isin(['4']),'sal_pred_y2y']*2/3+data.loc[data['season_code'].isin(['4']),'sal_pred_arma']/3
    agg_str = {'model_ensemble1': 'sum','sal_amt_1d':'sum'}
    d=data.groupby(by=['stat_date']).agg(agg_str).reset_index()
    score = []
    for n in range(len(d) // 7):
        dw = d.iloc[7 * n:7 * (n + 1), :]
        if dw['sal_amt_1d'].sum() != 0:
            s1 = round(np.abs((dw['model_ensemble1'].sum()-dw['sal_amt_1d'].sum())/dw['sal_amt_1d'].sum()),4)
            score.append(s1)
    print(score)

    score = pd.DataFrame()
    for i in data.distrib_code.unique():
        for j in data.season_code.unique():
            for k in data.new_line_code.unique():
                d=data[(data['distrib_code']==i)&(data['season_code']==j)&(data['new_line_code']==k)][['stat_date','model_ensemble1','sal_pred_gbdt','sal_pred_arma','sal_amt_1d','sal_pred_y2y','sal_pred_y2y']]
                d.set_index(pd.to_datetime(d['stat_date']), inplace=True)
                # d.plot()
                # plt.title(i+'_'+j+'_'+k)
                # plt.legend()
                # plt.show()
                # plt.close()
                score_w = []
                for n in range(len(d) // 7):
                    dw = d.iloc[7 * n:7 * (n + 1), :]
                    if dw['sal_amt_1d'].sum() != 0:
                        s1 = np.abs((dw['model_ensemble1'].sum()-dw['sal_amt_1d'].sum())/dw['sal_amt_1d'].sum())
                        score_w.append([n+1,i,j,k,s1])
                score = score.append(score_w)
    score.columns = ['week','distrib_code', 'season_code', 'new_line_code', 'score']
    agg_str = {'score': 'mean'}
    score = score.groupby(by=['season_code','week']).agg(agg_str).reset_index()
    print(score)
