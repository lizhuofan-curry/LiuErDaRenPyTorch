# 糖尿病二分类

本项目是第 08 章的 `Dataset` 与 `DataLoader` 示例。在第 07 章全量训练的基础上，使用自定义 `Dataset`、`random_split` 和 `DataLoader` 建立标准的训练/验证流程。

## 数据与流程

数据来自仓库共享目录 [`datasets/diabetes.csv.gz`](../../datasets/diabetes.csv.gz)。前 8 列是输入特征，最后一列是 0/1 二分类标签。

```text
diabetes.csv.gz
  → DiabetesDataset
  → random_split（80% 训练 / 20% 验证）
  → DataLoader（batch_size=32）
  → DNN：8 → 6 → 4 → 1
  → BCELoss + SGD
  → 验证集 loss 与准确率
```

## 运行方式

从仓库根目录执行：

```bash
python Chapter08_DatasetAndDataloader/DiabetesBinaryClassification/dataset_and_dataloader.py
```

脚本使用固定随机种子 `42`，训练 1000 个 epoch，并将训练集与验证集的 loss 曲线保存到 `images/training_validation_loss.png`。

## 运行结果

本次运行的最终验证准确率为 **77.63%**。固定随机种子可以使数据划分和批次顺序可复现，但不同 PyTorch 版本或硬件环境下的末位数值仍可能略有差异。

![训练集与验证集 loss 曲线](./images/training_validation_loss.png)

## 训练参数

| 参数 | 设置 |
|---|---:|
| 随机种子 | 42 |
| 训练/验证比例 | 80% / 20% |
| Batch size | 32 |
| Epochs | 1000 |
| 优化器 | SGD |
| 学习率 | 0.01 |
| 损失函数 | BCELoss |

验证集只用于观察训练过程，不参与参数更新。若要得到更严格的泛化性能，还应额外保留独立测试集。

