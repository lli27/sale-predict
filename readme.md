模型目标：
预测未来14天的库存成本金额 库存成本（T）= 库存成本（T-1）- 销售额（T-1）+ 进货成本（T）
库存成本（T+1）= 库存成本（T）- 销售额（T）+ 进货成本（T+1） 预测销售额（T）
这里有两个指标是未知的。进货成本（T），销售额（T）。进货成本用目标值替代，销售额需预测。
预测销售：
通过销量预估销售额  或者直接预估销售额