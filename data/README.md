# Covertype 数据

原始数据来自 UCI Covertype，包含 581012 条记录、54 个特征和七种森林覆盖类型。[UCI 来源](https://doi.org/10.24432/C50K5N)

raw 保留官方 ZIP、covtype.data.gz 和 covtype.info。原始 gzip 的 SHA-256 为 614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771，已与 scikit-learn 发布的 Covertype 校验值核对。

processed 保留完整 Parquet、固定测试集、四种训练规模及其 row_id。manifest.json 保存特征名、标签转换、各类样本数、随机种子、划分比例和文件校验值。文件名中的 train_100000 表示 10 万条训练样本，所有规模使用同一测试集。

特征包括十个数值属性、四个荒野区和四十个土壤类别指示变量。没有对树模型输入进行数值标准化。标签转换为 0 至 6，row_id 仅用于样本身份与划分审计，两者均不会作为输入特征。
