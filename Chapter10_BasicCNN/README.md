# 第 10 章：基础卷积神经网络（CNN）

本章在第 09 章 MNIST 全连接分类器的基础上，引入卷积、ReLU 与最大池化，并完成 Exercise 10-1：构建更复杂的 CNN，尝试不同通道数和全连接层宽度，比较准确率、参数量与训练时间。

![Exercise 10-1 作业要求](./images/exercise_10_1.png)

## 内容

- [`Lecture_10_Basic_CNN.pdf`](./Lecture_10_Basic_CNN.pdf)：第 10 章课程课件。
- [`MNIST_CNN_Configuration_Comparison.ipynb`](./MNIST_CNN_Configuration_Comparison.ipynb)：CNN-S、CNN-M、CNN-L 三种宽度配置的训练与对比实验。
- [`MNIST_Three_CNN_Architectures.ipynb`](./MNIST_Three_CNN_Architectures.ipynb)：普通 CNN、InceptionNet、简化 ResNet 三种架构的 MNIST 实测对比。
- `images/exercise_10_1.png`：Exercise 10-1 作业要求示意图。
- `images/cnn_configuration_comparison.png`：运行 Notebook 后自动生成的损失与准确率对比图。
- `images/mnist_three_cnn_comparison.png`：三种 CNN 架构的已执行训练曲线。

## 作业目标

```text
输入 (batch, 1, 28, 28)
→ [Conv2d → ReLU → MaxPool2d] × 3
→ Linear × 3
→ 输出 (batch, 10)
```

三个 `3×3` 卷积均使用 `padding=1`，最大池化核为 `2×2`。特征图尺寸依次变化：

```text
28×28 → 14×14 → 7×7 → 3×3
```

最后一组卷积的输出经过 Flatten，再由三层全连接层映射为 10 个类别 logits。模型输出前不添加 Softmax，因为 `CrossEntropyLoss` 已在内部完成相应计算。

## 对比配置

| 模型 | 卷积通道 | 全连接隐藏层 | 侧重点 |
|---|---|---|---|
| CNN-S | `8 → 16 → 32` | `64 → 32` | 参数少、训练快 |
| CNN-M | `16 → 32 → 64` | `128 → 64` | 容量与成本均衡 |
| CNN-L | `32 → 64 → 128` | `256 → 128` | 容量更大、训练成本更高 |

为了公平比较，三组实验固定以下条件：

- MNIST 训练集与测试集相同；
- batch size 为 64；
- SGD 学习率为 0.01，momentum 为 0.5；
- 随机种子为 42；
- 默认训练 10 个 epoch。

Notebook 会输出每个模型的参数量、最佳测试准确率、最终测试准确率和训练时间，并生成三组训练曲线。第 09 章原 MLP 的一次已记录测试准确率为 97.73%，图中仅将其作为参考线；CNN 的结果以实际运行输出为准。

## 三种 CNN 架构实测

第二个 Notebook 在相同 MNIST 数据、batch size、SGD 配置、随机种子和 10 个 epoch 下，对比普通 CNN、InceptionNet 与简化 ResNet。

![普通 CNN、InceptionNet 与 ResNetMNIST 对比](./images/mnist_three_cnn_comparison.png)

| 模型 | 参数量 | 最佳测试准确率 | 最终测试准确率 | 训练时间 |
|---|---:|---:|---:|---:|
| Normal_CNN | 106,058 | 98.71% | 98.48% | 285.4 s |
| InceptionNet | 90,074 | **98.91%** | **98.58%** | 331.7 s |
| ResNetMNIST | 77,418 | 98.06% | 98.06% | 291.9 s |

这次运行中，InceptionNet 参数量少于普通 CNN，并取得最高测试准确率，但训练时间最长；简化 ResNet 参数最少，但前两个 epoch 收敛较慢。以上结果来自 Notebook 中已保存的一次实际运行，硬件、PyTorch 版本和随机性变化会使结果略有不同。

## 运行方式

从仓库根目录启动 Jupyter：

```bash
python -m pip install jupyter torch torchvision matplotlib
jupyter lab
```

配置宽度对比 Notebook：

```text
Chapter10_BasicCNN/MNIST_CNN_Configuration_Comparison.ipynb
```

三种架构实测 Notebook：

```text
Chapter10_BasicCNN/MNIST_Three_CNN_Architectures.ipynb
```

配置宽度对比 Notebook 默认复用第 09 章目录中的 MNIST 数据；三种架构实测 Notebook 使用 `../dataset/mnist/`，并可通过 `download=True` 自动下载。首次调试时可将训练轮数临时改为 2，确认模型尺寸、数据加载、训练与绘图均正常后再进行完整实验。

## 验证点

- Shape check 输出应为 `torch.Size([4, 10])`。
- 训练集和测试集规模应分别为 60,000 与 10,000。
- 三个模型均应完整训练，且损失整体下降。
- 对比结论应综合准确率、参数量和训练时间，不能只按模型大小判断。
- 运行结束后应生成 `images/cnn_configuration_comparison.png`。
- 三种架构实测 Notebook 应保留 Normal_CNN、InceptionNet、ResNetMNIST 的结果表与曲线。

## 实验结论模板

> 在相同数据、优化器与训练轮数下，CNN-S、CNN-M、CNN-L 的最佳测试准确率分别为 **待填写**、**待填写**、**待填写**。综合参数量和训练时间，如果 CNN-M 与 CNN-L 的准确率接近，而 CNN-M 成本更低，则 CNN-M 是更均衡的配置。
