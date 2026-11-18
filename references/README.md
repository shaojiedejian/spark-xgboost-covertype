# 论文与官方代码来源

原论文为 Tianqi Chen 和 Carlos Guestrin 的 XGBoost A Scalable Tree Boosting System，KDD 2016，DOI 10.1145/2939672.2939785。可从 [arXiv 论文页面](https://arxiv.org/abs/1603.02754) 下载全文。

spark_estimator_examples.py 来自 XGBoost 官方 v2.1.4 标签，展示 SparkXGBClassifier 和 SparkXGBRegressor 的 Python 用法。本项目参考该接口编写 Covertype 实验，并固定使用 XGBoost 2.1.4。spark_estimator.rst 保留接口文档，XGBOOST_LICENSE 保留 Apache 2.0 许可证。[官方示例](https://github.com/dmlc/xgboost/blob/v2.1.4/demo/guide-python/spark_estimator_examples.py)

当前应用实验使用 Covertype 自选数据和现代 Spark 接口。原论文的原始 benchmark、历史版本及集群性能数值没有在本机重建。
