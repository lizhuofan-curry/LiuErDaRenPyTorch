# MNIST 手写数字识别

第 09 章 Softmax 多分类器实践：使用全连接神经网络识别 0–9 手写数字。模型最后输出 10 个 logits，`CrossEntropyLoss` 在内部完成 softmax 与交叉熵计算。

![MNIST 样本预览](./images/mnist_samples.png)

## 内容

- `识别手写数字.ipynb`：原实验整理后的 Notebook；从项目目录或仓库根目录启动均可定位数据。
- `train_mnist.py`：可在终端复现训练并生成指标图。
- `data/MNIST/raw/*.gz`：官方 MNIST IDX 压缩文件；首次运行会自动解压为 `torchvision` 可读取的原始文件。
- `images/mnist_samples.png`：训练集 0–9 的样本预览。

## 数据集

MNIST 包含 60,000 张训练图像和 10,000 张测试图像；每张为 28×28 灰度图。仓库保留了官方四个压缩 IDX 文件，以便离线运行，同时避免提交重复的解压副本。

## 环境与运行

```bash
python -m pip install torch torchvision matplotlib
python Chapter09_SoftmaxClassifier/MNISTHandwrittenDigitRecognition/train_mnist.py --epochs 10
```

第一次运行会将项目内 `data/MNIST/raw/*.gz` 解压为 IDX 文件，并在 `images/training_metrics.png` 写出训练损失和测试准确率图。该图由本地运行生成，未提交，以免把不同硬件或随机种子的结果混在一起。

## 模型与训练设置

```text
28×28 image → Flatten(784) → 512 → 256 → 128 → 64 → 10 logits
```

- 隐藏层激活：ReLU
- 损失函数：CrossEntropyLoss
- 优化器：SGD（学习率 0.01，动量 0.5）
- 归一化：MNIST 均值 0.1307、标准差 0.3081
- 默认批大小：64；默认训练轮数：10

## 验证点

训练时应确认数据集规模为 60,000 / 10,000，并观察训练损失总体下降、测试准确率持续提升。最终指标会随 PyTorch 版本、硬件和随机性略有差异。
