# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/9/11 15:08
@Desc   : 告警：查询odps分区，若为成功生成后28天的分区，则发送告警邮件。每天8点执行一次。
"""
from src.db import odps
import datetime

odps = odps()
table = odps.conn.get_table('ads_fd_sale_pred_distrib_1d')
bdpdate = (datetime.date.today()+datetime.timedelta(days=-1)).strftime('%Y%m%d')
ds = []
for partition in table.partitions:
    ds.append(partition.name.replace("'","").split("=")[1])
value = sum([1 if d > bdpdate else 0 for d in ds])
print(value)