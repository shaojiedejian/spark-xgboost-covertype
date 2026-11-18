# 基于 Spark 与 XGBoost 的森林覆盖类型分类

这是一个可以在本机 CPU 上完整运行的 Python 项目。使用 581012 条 Covertype 真实数据，通过 Spark 完成读取、特征组装和缓存，调用官方 SparkXGBClassifier 训练七分类模型，并与 Spark 决策树和随机森林比较。

已完成的实验报告、图表和指标通过 [实验结果入口](results/README.md) 查看。首次运行会自动从 UCI 下载原始数据，并在项目目录生成数据划分、模型和完整实验记录。

## 项目研究的问题

- 相同训练与测试数据上，三个模型的预测效果和训练成本如何。
- XGBoost 的训练样本从 1 万增加到 30 万时，效果与耗时如何变化。
- 固定总 CPU 配额时，一工作进程和两工作进程的训练表现如何。

项目基于 XGBoost 原论文的方法及官方 Python/Spark 实现开展自选数据实验。原论文的集群 benchmark 和 2016 年版本数值结果属于不同实验设置。论文链接、官方示例及许可证见 [参考资料](references/README.md)。

## 目录

| 位置 | 内容 |
|---|---|
| src/forest_project/train.py | 三种模型训练、评价、保存及重新加载校验 |
| src/forest_project/predict.py | 独立加载模型并预测新数据 |
| scripts/prepare_data.py | 下载真实数据、固定划分及校验 |
| scripts/run_experiments.py | 正式模型、规模及并行对比，共 14 次运行 |
| scripts/make_report.py | 中文报告和六张结果图 |
| tests | 数据隔离、完整预测及指标审计 |
| configs/experiment.json | 固定种子、模型参数与实验规模 |
| data/raw | UCI 原始 ZIP、gzip 和数据说明 |
| data/processed | Parquet、固定样本 ID 与校验清单 |
| results | 日志、模型、预测、汇总、图表和报告 |
| references | 原论文链接与官方 v2.1.4 示例 |
| .runtime/xgb-venv | 本项目独立 Python 环境 |

## 本机完整复跑

安装依赖后，在项目目录运行以下命令。

```bash
bash scripts/reproduce.sh
```

该入口执行数据校验、14 次正式实验、逐条预测审计和报告生成。每次运行使用新的结果目录，不覆盖已有实验。结果位置记录在 results/latest_experiment.txt，报告入口为 results/README.md。

本项目实测使用 Python 3.12、Java 17、PySpark 3.5.5、XGBoost 2.1.4。Spark 以 local[4] 运行，driver 内存为 4 GB。它在一台电脑上启动真实 Spark 任务及 XGBoost 工作进程。当前本机实验使用本地 Parquet，未部署 HDFS 或 YARN。

## 首次在新环境运行

需要 Python 3.12 和 Java 17。macOS 上 `scripts/env.sh` 自动查找 Java 17；如需手动指定，可设置 `FOREST_JAVA_HOME`。

```bash
bash scripts/setup.sh
bash scripts/reproduce.sh
```

XGBoost 的 macOS CPU 动态库需要 OpenMP。当前电脑已安装 libomp。其他 macOS 电脑如导入 XGBoost 提示 libomp.dylib 缺失，可在其 Homebrew 环境安装 libomp。所有 Python 依赖安装在 .runtime/xgb-venv，项目使用的具体版本另存 requirements-lock.txt。PySpark 3.5 在 Python 3.12 上还依赖 setuptools 提供 distutils，本项目已显式固定该依赖。

首次安装与下载数据需要网络。数据保存在本地后，完整复跑复用已经过 SHA-256 校验的文件。

## 单独运行一个模型

训练 10 万样本的单工作进程 XGBoost，输出目录必须尚不存在。

```bash
bash scripts/train.sh \
  --model xgboost \
  --train-size 100000 \
  --workers 1 \
  --run-dir results/my_xgboost
```

训练随机森林或决策树。

```bash
bash scripts/train.sh --model random_forest --train-size 100000 --run-dir results/my_random_forest
bash scripts/train.sh --model decision_tree --train-size 100000 --run-dir results/my_decision_tree
```

同一份 10 万样本上运行两个 XGBoost 工作进程。

```bash
bash scripts/train.sh \
  --model xgboost \
  --train-size 100000 \
  --workers 2 \
  --run-dir results/my_xgboost_two_workers
```

单进程分配四个 CPU，两进程分别分配两个 CPU，总预算均为四核。训练日志会显示 Training on CPUs、实际 num_workers、nthread 和工作进程 rank。额外进程会增加通信和协调成本，单机小规模实验中可能更慢。

每个训练目录保存 metrics.json、classification_report.json、confusion_matrix.csv、feature_importance.csv、predictions.parquet 和 Spark event log。默认保存 model 下的完整 PipelineModel，XGBoost 另有 booster.json。正式批量实验仅在每组的第一次运行保存模型，第二次保留逐条预测和计时。

## 加载模型并预测

以下示例使用前述独立训练产生的模型。

```bash
bash scripts/predict.sh \
  --model-dir results/my_xgboost/model \
  --input data/processed/test.parquet \
  --output results/my_predictions
```

输入可以是 Parquet，或带特征名表头的 CSV。54 个特征名见 data/processed/manifest.json。新数据可以没有 label。输出为 Spark 写出的 CSV 目录和 metadata.json，包含 prediction、原数据集 1 至 7 的 predicted_cover_type 以及类别名称。

新数据包含 label 时，预测输出也保留它以便评价。这个字段不会进入 VectorAssembler 特征列表。预处理和训练阶段均使用固定的 54 个特征。

## 数据与实验配置

Covertype 包含 581012 条数据、54 个特征和七类标签。标签从原始 1 至 7 转换为 0 至 6。20% 分层留作固定测试集，共 116203 条，其余数据形成训练池。1 万、5 万、10 万和 30 万训练子集按类别比例构建并逐级嵌套。随机种子为 2026。

XGBoost 使用 100 轮提升、深度 6、学习率 0.1、hist 和 CPU。决策树深度 8，随机森林 80 棵树、深度 8。每个配置重复两次，种子相同，用于观察计时波动。测试集未用于超参数调优或早停。

修改 configs/experiment.json 后重新运行 prepare_data 会根据种子、测试比例、规模和校验值决定是否重建数据。正式实验参数与输入哈希保存在各次 manifest.json，旧结果对应其原始设置。

## 检查与结果解释

```bash
bash -c 'source scripts/env.sh; "$PYSPARK_PYTHON" tests/verify_data.py'
bash -c 'source scripts/env.sh; "$PYSPARK_PYTHON" tests/verify_results.py'
```

数据检查覆盖官方数据哈希、行列数、有限值、七种标签、one-hot 指示变量、训练测试隔离和嵌套关系。结果检查读取所有正式实验的逐条预测，重算准确率、宏平均 F1、加权 F1 和混淆矩阵，并检查 CPU 及工作进程日志。

主要报告宏平均 F1，帮助观察类别不均衡下的效果。模型对比使用固定参数，结果代表这些配置。各模型参数和计算量并未优化为等预算，不应将此实验推广为算法在所有条件下的排名。

training_seconds 只计 fit 调用，但包含该调用内部的重分区、数据转换和通信。preprocessing_seconds 包括 Spark 输入读取、特征组装、分区、缓存和校验。prediction_seconds 包括真实推断和收集预测结果。wall_seconds 包括完整进程生命周期及结果写入，第一次运行还包含模型保存与加载验证。

## 与大数据和原论文的关系

项目使用 Spark 数据分区与缓存，并通过官方 SparkXGBClassifier 组织分布式训练任务，研究数据规模和资源配置的影响。Covertype 属于适合本地教学的中等规模数据集。本项目的多工作进程均运行在本机，跨机器扩展性需要独立集群实验。

论文方法来自 Chen 和 Guestrin 的 XGBoost A Scalable Tree Boosting System，KDD 2016。当前实现采用现代官方 Python 接口及自选数据。它对应作业允许的“使用现有代码及自定义数据进行测试分析”。

## 原始来源

- [XGBoost 原论文](https://arxiv.org/abs/1603.02754)
- [官方 v2.1.4 Spark 示例](https://github.com/dmlc/xgboost/blob/v2.1.4/demo/guide-python/spark_estimator_examples.py)
- [UCI Covertype](https://doi.org/10.24432/C50K5N)

references 中保留官方文件和 Apache 2.0 许可证，便于溯源。个人姓名、班级、成员贡献和自评分由实际提交者填写。
