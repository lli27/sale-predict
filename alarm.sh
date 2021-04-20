#!/bin/bash
#author:lilijuan
var1=$(python alarm.py)
var2=$(date +%Y%m%d)
email="lilijuan@fastfish.com"
logname="SalePredict_${var2}.error"
if [ ${var1} -lt 28 ]; then sudo echo "ads_fd_sale_pred_distrib_1d partitions fail! produce partitions number ${var1}" >> "/root/log/${logname}"; fi
mail -s "sale predict alarm!!!" ${email} < "/root/log/${logname}"

