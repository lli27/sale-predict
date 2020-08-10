# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/6/30 13:28
@Desc   :
"""
from src.utils.logfactory import LogFactory
from src.db import odps
from src.conf import Config
from odps.df import DataFrame
import pandas as pd
import datetime
import math
import numpy as np
import traceback
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARMA
import statsmodels.tsa.stattools as st
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

class SalePredict:

    def __init__(self):
        self.logging = LogFactory()
        self.odps = odps()
        self.config = Config().config_data
        self.bdpdate = (datetime.date.today()+datetime.timedelta(days=-1)).strftime('%Y%m%d') # 业务日期T-1

    def read_data(self, sql_cmd):
        """
        读取销售数据
        :return: DataFrame
        """
        print(sql_cmd)
        try:
            with self.odps.conn.execute_sql(sql_cmd).open_reader() as reader:
                data = reader.to_pandas()
        except:
            self.logging.error(traceback.format_exc())
            return pd.DataFrame()
        self.logging.info("read_data success!")
        return data

    def get_dsitrib_data(self):
        """
        预测-多进程
        :return: list, list的元素为DataFrame
        """
        # 处理为时间序列数据
        sql_cmd = self.config.get('GET_NOR_DATA').format(self.bdpdate)
        data = self.read_data(sql_cmd)
        # data = data[data['distrib_code']==self.distrib_code]
        if len(data)==0:
            self.logging.error("get_dsitrib_data fail!")
            return []
        else:
            try:
                data["stat_date"] = pd.to_datetime(data["stat_date"])
                data.set_index('stat_date', inplace=True)
                data.index = data.index.to_period("D")
                data['amt_1d_clr'] = data['amt_1d_clr'].map(lambda x: np.log(x))
                # 得到具体分区的时间序列
                pool = ThreadPool(4)
                yhat = pool.starmap(self.algo_svm, zip(data.groupby(by=['distrib_code'])))
                pool.close()
                pool.join()
            except:
                self.logging.error(traceback.format_exc())
                return []
        self.logging.info("get_dsitrib_data success!")
        return yhat

    def algo_time_series(self, grouper):
        """
        时间序列算法
        :param grouper:
        :return: DataFrame
        """
        try:
            data = grouper[1].amt_1d_clr.sort_index()
            # model = ExponentialSmoothing(data, seasonal='mul', seasonal_periods = 7).fit() # mul 4 0.124
            model = ARMA(data,(2,1)).fit()
            # print(model.summary)
            # order = st.arma_order_select_ic(data,max_ar=3,max_ma=3,ic=['aic','bic','hqic'])
            # print(order.bic_min_order)
            yhat = model.predict(start=len(data), end=len(data) + 6)
            yhat = pd.DataFrame(yhat, columns=['yhat'])
            yhat['distrib_code'] = grouper[0]
            yhat['stat_date'] = yhat.index.to_timestamp()
        except:
            self.logging.error(traceback.format_exc())
            return pd.DataFrame()
        self.logging.info("algo_time_series success!")
        return yhat

    def salepredict(self):
        """
        输出最终预测结果
        :return: DataFrame
        """
        yhat = self.get_dsitrib_data()
        num = len(yhat)
        if num == 0:
            self.logging.error("salepredict fail！")
            return pd.DataFrame()
        else:
            try:
                yhat_full = pd.DataFrame(columns=['yhat','distrib_code','stat_date'])
                for i in range(num):
                    yhat_full = pd.concat([yhat_full, yhat[i]],ignore_index=True)
                yhat_full['day_of_week'] = yhat_full['stat_date'].dt.dayofweek.values
                yhat_full['stat_date'] = yhat_full['stat_date'].dt.strftime('%Y%m%d').values
                yhat_full['yhat'] = yhat_full['yhat'].map(lambda x: np.exp(x))
                # 乘以权重
                sql_cmd1 = self.config.get('GET_FESTIVAL_WEIGHT').format(self.bdpdate)
                fest_weight = self.read_data(sql_cmd1)
                sql_cmd2 = self.config.get('GET_WEEK_WEIGHT').format(self.bdpdate)
                week_weight = self.read_data(sql_cmd2)
                # 根据天气数据对结果做一定程度的降权。[0.85,1]
                sql_cmd3 = self.config.get('GET_WEATHER_DATA').format(self.bdpdate)
                weather = self.read_data(sql_cmd3)
                weather_weight = self.processing(weather)
                yhat_full = pd.merge(yhat_full, week_weight, on=['distrib_code', 'day_of_week'], how = 'left')
                yhat_full = pd.merge(yhat_full, fest_weight, on=['distrib_code', 'stat_date'], how = 'left')
                yhat_full = pd.merge(yhat_full, weather_weight, on=['distrib_code', 'stat_date'], how='left')
                yhat_full['weight'] = yhat_full.apply(func=lambda x: x.fest_weight if x.fest_weight > 0 else x.week_weight,
                                            axis=1)
                # 只用15天的天气预报
                t15 = (pd.to_datetime(self.bdpdate)+pd.to_timedelta(15,unit='D')).strftime('%Y%m%d')
                yhat_full.loc[yhat_full['stat_date']>t15,'icon'] = 1.0
                yhat_full['sale_pred'] = yhat_full['yhat'] * yhat_full['weight'] * yhat_full['icon']
                # yhat_full['min_sale_pred'] = yhat_full['sale_pred'] * 0.9
                # yhat_full['max_sale_pred'] = yhat_full['sale_pred'] * 1.05
            except:
                self.logging.error(traceback.format_exc())
                return pd.DataFrame()
        self.logging.info("salepredict success！")
        return yhat_full

    def processing(self, X):
        """
        天气数据处理
        :param X:
        :return:
        """
        # 根据是否下雨/下雪进行处理 不算阵雨
        X['icon'] = X['icon'].apply(func=lambda x: '0' if x in ('18','22','25','26','19','29','15','12') else '1')
        # 天气状况、最高温度、最低温度、风速都处理为数值型
        X['icon'] = X['icon'].astype('float')
        agg_str = {'icon': 'mean'}
        data = X.groupby(by=['distrib_code', 'stat_date']).agg(agg_str).reset_index()
        scaler = MinMaxScaler(copy=True, feature_range=(0.85, 1))
        data['icon'] = scaler.fit_transform(np.reshape(data['icon'].values,(-1,1)))
        return data


    def write_to_db(self):
        """
        将最终结果写入数据库
        :param data:
        :return: （1-成功；0-失败）
        """
        yhat = self.salepredict()
        if yhat.empty:
            self.logging.error("write_to_db fail！")
            return 0
        else:
            try:
                DataFrame(yhat).persist(name='sale_pred', overwrite=True, partition='ds={}'.format(self.bdpdate), odps=self.odps.conn)
            except:
                self.logging.error(traceback.format_exc())
                return 0
        self.logging.info("write_to_db success！")
        return 1

    def plot(self,data):
        """
        可视化
        :param data:
        :return:
        """
        data = data.sort_index()
        for code in self.config.get('DISTRIB_CODE_LIST'):
            data[data['distrib_code']==code]['amt_1d_clr'].plot(kind='line')
            plt.show()
            plt.close()
        return

    def score(self):
        """
        验证准确率
        :param yhat: 预测值
        :return: score: 得分
        """
        yhat = self.salepredict()
        sql_cmd = self.config.get('SELECT_ODPS_DATA').format(self.bdpdate)
        data = self.read_data(sql_cmd)
        yhat = pd.merge(yhat, data, on=['distrib_code', 'stat_date'], how = 'left')
        score_w = []
        for i in yhat.distrib_code.unique():
            y = yhat[yhat['distrib_code']==i]
            y['stat_date'] = pd.to_datetime(y['stat_date'])
            y.set_index('stat_date', inplace=True)
            for j in range(len(y) // 7): # 按周计算预测误差
                yw = y.iloc[7 * j:7 * (j + 1), :]
                try:
                    score_w.append(
                        [i, round(abs((yw['sale_pred'].sum() - yw['cost_amt_1d'].sum()) / yw['cost_amt_1d'].sum()), 4)])
                except:
                    pass
            plt.figure()
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.plot(y['sale_pred'],label='销售成本预测值')
            plt.plot(y['cost_amt_1d'],'ro',label='销售成本实际值')
            # plt.plot(y['sale_pred'] * 1.05,label='最大销售成本预测值')
            # plt.plot(y['sale_pred'] * 0.92, label='最小销售成本预测值')
            plt.legend()
            plt.title(i+'销售成本预测'+self.bdpdate)
            plt.show()
            plt.close()
        return score_w

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

    def algo_svm(self, grouper):
        """
        svm算法
        :param grouper:
        :return:
        """
        try:
            data = grouper[1].amt_1d_clr.sort_index()
            # 将时间序列数据转换为有监督数据集，设n=28
            data = self.series_to_supervised(data, n_in=7) # Dataframe
            test_x = self.series_to_test(data.values[-1])
            train_x = data.iloc[:, :-1]
            train_y = data.iloc[:, -1]
            # model = svm.SVR()
            model = RandomForestRegressor()
            # param_grid = {'C': [1,10,100,1000], 'gamma': [1/10,1/28,1/100]}
            param_grid = {'n_estimators': [100,300,500], 'min_samples_leaf': [10,20,30]}
            grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1)
            grid_search.fit(X=train_x, y=train_y)
            best_parameters = grid_search.best_estimator_.get_params()
            for para, val in list(best_parameters.items()):
                print(para, val)
            # model = svm.SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
            model = RandomForestRegressor(n_estimators=best_parameters['n_estimators'],min_samples_leaf=best_parameters['min_samples_leaf'])
            model.fit(X=train_x, y=train_y)
            yhat = []
            for i in range(28):
                yhat_tmp = model.predict(X=test_x)
                test_x = self.series_to_test(test_x, yhat_tmp)
                yhat.append(yhat_tmp[0])
            yhat = pd.DataFrame(yhat, columns=['yhat'])
            yhat['distrib_code'] = grouper[0]
            yhat['stat_date'] = pd.date_range(start=self.bdpdate,periods=29,freq='D')[1:]
        except:
            self.logging.error(traceback.format_exc())
            return pd.DataFrame()
        self.logging.info("algo_time_series success!")
        return yhat
if __name__=="__main__":
    salepredict = SalePredict()
    score=salepredict.score()







