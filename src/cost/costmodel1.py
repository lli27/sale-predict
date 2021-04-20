# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/6/30 13:28
@Desc   : gbdt模型
"""
from src.utils.logfactory import LogFactory
from src.db import odps
from src.conf import Config
import pandas as pd
import numpy as np
import traceback
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

class costmodel1:

    def __init__(self, date, days):
        self.logging = LogFactory()
        self.odps = odps()
        self.config = Config().config_data
        self.date = date
        self.days = days

    def get_dsitrib_data(self, data):
        """
        预测-多进程
        :return: list, list的元素为DataFrame
        """
        try:
            data['cost_amt_clr_log'] = data['cost_amt_clr'].apply(lambda x: x if x > 0 else 1e-10)
            # 得到具体分区的时间序列
            pool = ThreadPool(4)
            yhat = pool.starmap(self.algo_gbdt, zip(data.groupby(by=['distrib_code','season_code','new_line_code'])))
            pool.close()
            pool.join()
        except:
            self.logging.error(traceback.format_exc())
            return []
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
            return pd.DataFrame()
        else:
            try:
                yhat_full = pd.DataFrame(columns=['yhat','distrib_code','season_code','new_line_code','stat_date'])
                for i in range(num):
                    yhat_full = pd.concat([yhat_full, yhat[i]],ignore_index=True)
                yhat_full['day_of_week'] = yhat_full['stat_date'].dt.dayofweek.values
                yhat_full['stat_date'] = yhat_full['stat_date'].dt.strftime('%Y%m%d').values
                yhat_full['yhat'] = yhat_full['yhat'].apply(lambda x: x if x > 0 else 1e-10)
                # 乘以权重
                yhat_full = pd.merge(yhat_full, week_weight, on=['distrib_code','day_of_week'], how = 'left')
                yhat_full = pd.merge(yhat_full, fest_weight, on=['distrib_code','stat_date'], how = 'left')
                yhat_full['weight'] = yhat_full.apply(func=lambda x: x.fest_cost_weight if x.fest_cost_weight > 0 else x.cost_week_wgt,
                                            axis=1)
                yhat_full['cost_pred_gbdt'] = yhat_full['yhat'] * yhat_full['weight']
            except:
                self.logging.error(traceback.format_exc())
                return pd.DataFrame()
        self.logging.info("salepredict success！")
        return yhat_full

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        """
        将时间序列重构为监督学习数据集.
        参数:
            data: 观测值序列，类型为列表或Numpy数组。
            n_in: 输入的滞后观测值(X)长度。
            n_out: 输出观测值(y)的长度。
            dropnan: 是否丢弃含有NaN值的行，类型为布尔值。
        返回值:
            经过重组后的Pandas DataFrame序列.
        """
        n_vars = 1
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # 输入序列 (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # 预测序列 (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # 将列名和数据拼接在一起
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # 丢弃含有NaN值的行
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def series_to_test(self, row, yhat=[]):
        """
        构造有监督学习的测试集
        :param row: array
        :return: array
        """
        row = list(row.flatten())
        row.pop(0)
        if len(yhat)>0:
            row.append(yhat[0])
        row = np.reshape(row,(1,len(row)))
        return row

    def algo_gbdt(self, grouper):
        """
        基于决策树的算法--gbdt
        :param grouper:
        :return:
        """
        try:
            data = grouper[1].cost_amt_clr_log.sort_index()
            # 将时间序列数据转换为有监督数据集
            data = self.series_to_supervised(data, n_in=7) # Dataframe
            test_x = self.series_to_test(data.values[-1])
            train_x = data.iloc[:, :-1]
            train_y = data.iloc[:, -1]
            model = GradientBoostingRegressor()
            param_grid = {'n_estimators': [100,300,500],'learning_rate': [0.01,0.03,0.05,0.1,0.3]}
            grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1)
            grid_search.fit(X=train_x, y=train_y)
            best_parameters = grid_search.best_estimator_.get_params()
            for para, val in list(best_parameters.items()):
                self.logging.info("para: {}, val: {}".format(para, val))
            model = GradientBoostingRegressor(n_estimators=best_parameters['n_estimators'], learning_rate=best_parameters['learning_rate'])
            model.fit(X=train_x.values, y=train_y.values)
            yhat = []
            for i in range(self.days):
                yhat_tmp = model.predict(test_x)
                test_x = self.series_to_test(test_x, yhat_tmp)
                yhat.extend(yhat_tmp)
            yhat = pd.DataFrame(yhat, columns=['yhat'])
            yhat['distrib_code'] = grouper[0][0]
            yhat['season_code'] = grouper[0][1]
            yhat['new_line_code'] = grouper[0][2]
            yhat['stat_date'] = pd.date_range(start=self.date,periods=self.days+1,freq='D')[1:]
        except:
            self.logging.error(traceback.format_exc())
            return pd.DataFrame()
        self.logging.info("algo_time_series success!")
        return yhat







