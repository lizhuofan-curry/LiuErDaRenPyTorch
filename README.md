# 刘二大人《PyTorch 深度学习实践》学习资料

本仓库按照课程章节顺序，整理刘二大人《PyTorch 深度学习实践》的课程课件、个人学习笔记与实验代码，方便学习、复习和继续补充实践内容。

- [课程官方页面](https://liuii.github.io/post/pytorch-tutorials/)
- [Bilibili 完整课程（13 讲）](https://www.bilibili.com/video/BV1Y7411d7Ys)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/)

> 本仓库是个人学习归档，并非课程官方仓库。课程内容与资料版本请以课程官方页面为准。

## 快速跳转

**页面导航：** [学习进度](#学习进度) · [章节导航](#章节导航) · [运行环境](#运行环境) · [运行示例](#运行示例) · [数据集](#数据集) · [目录结构](#目录结构) · [版权说明](#版权说明)

**章节目录：**
[01](./Chapter01_Overview/) ·
[02](./Chapter02_LinearModel/) ·
[03](./Chapter03_GradientDescent/) ·
[04](./Chapter04_BackPropagation/) ·
[05](./Chapter05_LinearRegressionWithPyTorch/) ·
[06](./Chapter06_LogisticRegression/) ·
[07](./Chapter07_MultipleDimensionInput/) ·
[08](./Chapter08_DatasetAndDataloader/) ·
[09](./Chapter09_SoftmaxClassifier/) ·
[10](./Chapter10_BasicCNN/) ·
[11](./Chapter11_AdvancedCNN/) ·
[12](./Chapter12_BasicRNN/) ·
[13](./Chapter13_RNNClassifier/) ·
[datasets](./datasets/)

## 学习进度

- [x] 13 / 13 章课程课件已归档。
- [x] 第 01–06 章已有章内学习笔记。
- [x] 第 02–09 章已有可运行实验脚本。
- [x] 第 09 章已有 MNIST 手写数字多分类实践。
- [x] 第 10 章已有基础 CNN 配置对比 Notebook 与实验说明。
- [x] 第 11 章已有普通 CNN、Inception 与 ResNet 的 MNIST 架构对比，以及 CIFAR-10 三种 CNN 的实测曲线。
- [x] 第 12 章已有章内学习笔记，以及 RNNCell + One-hot、RNN + One-hot、RNN + Embedding 练习。
- [ ] 第 12 章后续实验、第 13 章学习笔记与实验代码待继续补充。

## 章节导航

| 章节 | 课程主题 | 视频 | 本仓库资料 |
|---|---|---|---|
| [第 01 章](./Chapter01_Overview/) | 深度学习概论（Overview） | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=1) | [课件](./Chapter01_Overview/Lecture_01_Overview.pdf) · [笔记](./Chapter01_Overview/README.md) · 暂无代码 |
| [第 02 章](./Chapter02_LinearModel/) | 线性模型 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=2) | [课件](./Chapter02_LinearModel/Lecture_02_Linear_Model.pdf) · [笔记](./Chapter02_LinearModel/README.md) · [2D Loss](./Chapter02_LinearModel/plot_2d_loss_effect.py) · [3D Cost](./Chapter02_LinearModel/plot_3d_cost_surface_effect.py) |
| [第 03 章](./Chapter03_GradientDescent/) | 梯度下降算法 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=3) | [课件](./Chapter03_GradientDescent/Lecture_03_Gradient_Descent.pdf) · [笔记](./Chapter03_GradientDescent/README.md) · [Batch GD](./Chapter03_GradientDescent/gradient_descent.py) · [SGD](./Chapter03_GradientDescent/stochastic_gradient_descent.py) · [Mini-batch](./Chapter03_GradientDescent/mini_batch_gradient_descent.py) |
| [第 04 章](./Chapter04_BackPropagation/) | 反向传播 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=4) | [课件](./Chapter04_BackPropagation/Lecture_04_Back_Propagation.pdf) · [笔记](./Chapter04_BackPropagation/README.md) · [`y = wx`](./Chapter04_BackPropagation/backpropagation_y_wx.py) · [二次模型](./Chapter04_BackPropagation/backpropagation_quadratic.py) |
| [第 05 章](./Chapter05_LinearRegressionWithPyTorch/) | 使用 PyTorch 实现线性回归 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=5) | [课件](./Chapter05_LinearRegressionWithPyTorch/Lecture_05_Linear_Regression_with_PyTorch.pdf) · [笔记与 7 种优化器示例](./Chapter05_LinearRegressionWithPyTorch/README.md) |
| [第 06 章](./Chapter06_LogisticRegression/) | 逻辑回归 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=6) | [课件](./Chapter06_LogisticRegression/Lecture_06_Logistic_Regression.pdf) · [笔记](./Chapter06_LogisticRegression/README.md) · [Sigmoid 概率回归](./Chapter06_LogisticRegression/logistic_regression.py) |
| [第 07 章](./Chapter07_MultipleDimensionInput/) | 多维特征输入 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=7) | [课件](./Chapter07_MultipleDimensionInput/Lecture_07_Multiple_Dimension_Input.pdf) · [笔记](./Chapter07_MultipleDimensionInput/README.md) · [糖尿病二分类](./Chapter07_MultipleDimensionInput/diabetes_binary_classification.py) |
| [第 08 章](./Chapter08_DatasetAndDataloader/) | Dataset 与 DataLoader | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=8) | [课件](./Chapter08_DatasetAndDataloader/Lecture_08_Dataset_and_Dataloader.pdf) · [笔记](./Chapter08_DatasetAndDataloader/README.md) · [Dataset 与 DataLoader](./Chapter08_DatasetAndDataloader/dataset_and_dataloader.py) |
| [第 09 章](./Chapter09_SoftmaxClassifier/) | Softmax 多分类器 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=9) | [课件](./Chapter09_SoftmaxClassifier/Lecture_09_Softmax_Classifier.pdf) · [章节笔记](./Chapter09_SoftmaxClassifier/README.md) · [MNIST 手写数字识别](./Chapter09_SoftmaxClassifier/MNISTHandwrittenDigitRecognition/README.md) · [Notebook](./Chapter09_SoftmaxClassifier/MNISTHandwrittenDigitRecognition/识别手写数字.ipynb) |
| [第 10 章](./Chapter10_BasicCNN/) | 基础卷积神经网络（CNN） | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=10) | [课件](./Chapter10_BasicCNN/Lecture_10_Basic_CNN.pdf) · [章节笔记](./Chapter10_BasicCNN/README.md) · [MNIST CNN 配置对比](./Chapter10_BasicCNN/MNIST_CNN_Configuration_Comparison.ipynb) |
| [第 11 章](./Chapter11_AdvancedCNN/) | 高级卷积神经网络（CNN） | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=11) | [课件](./Chapter11_AdvancedCNN/Lecture_11_Advanced_CNN.pdf) · [章节笔记](./Chapter11_AdvancedCNN/README.md) · [MNIST 三种 CNN 架构实测](./Chapter11_AdvancedCNN/MNIST_Three_CNN_Architectures.ipynb) · [CIFAR-10 三种 CNN 对比与实测曲线](./Chapter11_AdvancedCNN/CIFAR10_Three_CNN_Architectures/README.md) |
| [第 12 章](./Chapter12_BasicRNN/) | 基础循环神经网络（RNN） | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=12) | [课件](./Chapter12_BasicRNN/Lecture_12_Basic_RNN.pdf) · [章节笔记](./Chapter12_BasicRNN/README.md) · [RNNCell + One-hot](./Chapter12_BasicRNN/RNNCell%20%2B%20Onehot.py) · [RNN + One-hot Notebook](./Chapter12_BasicRNN/RNN%20%EF%BC%8B%20Onehot%20%E7%BB%83%E4%B9%A0.ipynb) · [RNN + Embedding Notebook](./Chapter12_BasicRNN/RNN%20%2B%20Embedding%20%E7%BB%83%E4%B9%A0.ipynb) |
| [第 13 章](./Chapter13_RNNClassifier/) | 循环神经网络分类器 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=13) | [课件](./Chapter13_RNNClassifier/Lecture_13_RNN_Classifier.pdf) · 暂无笔记与代码 |

## 运行环境

仓库当前未锁定 Python 或依赖版本。运行现有示例需要 Python 3，并根据章节安装以下依赖：

```bash
python -m pip install numpy matplotlib torch torchvision jupyter
```

- 第 02 章使用 NumPy 和 Matplotlib。
- 第 03 章仅使用 Python 标准库。
- 第 04–08 章使用 PyTorch 自动求导和优化器。
- 第 09–11 章额外使用 `torchvision` 读取 MNIST 数据。

## 运行示例

克隆仓库并进入根目录：

```bash
git clone https://github.com/lizhuofan-curry/LiuErDaRenPyTorch.git
cd LiuErDaRenPyTorch
```

运行线性模型可视化：

```bash
python Chapter02_LinearModel/plot_2d_loss_effect.py
python Chapter02_LinearModel/plot_3d_cost_surface_effect.py
```

这两个脚本会显示图形，并在 `Chapter02_LinearModel/images/` 下保存 PNG 文件。

运行梯度下降示例：

```bash
python Chapter03_GradientDescent/gradient_descent.py
python Chapter03_GradientDescent/stochastic_gradient_descent.py
python Chapter03_GradientDescent/mini_batch_gradient_descent.py
```

运行 PyTorch 反向传播示例：

```bash
python Chapter04_BackPropagation/backpropagation_y_wx.py
python Chapter04_BackPropagation/backpropagation_quadratic.py
```

运行第 05 章优化器示例：

```bash
python "Chapter05_LinearRegressionWithPyTorch/optim.SGD版.py"
```

其余六种优化器的运行命令与对比说明见[第 05 章笔记](./Chapter05_LinearRegressionWithPyTorch/README.md)。

运行第 06 章 Sigmoid 概率回归示例：

```bash
python Chapter06_LogisticRegression/logistic_regression.py
```

模型原理、训练设置与运行结果图见[第 06 章笔记](./Chapter06_LogisticRegression/README.md)。

运行第 07 章多维输入二分类示例：

```bash
python Chapter07_MultipleDimensionInput/diabetes_binary_classification.py
```

运行第 08 章 Dataset 与 DataLoader 示例：

```bash
python Chapter08_DatasetAndDataloader/dataset_and_dataloader.py
```

训练/验证划分、mini-batch 流程与结果图见[第 08 章笔记](./Chapter08_DatasetAndDataloader/README.md)。

运行第 09 章 MNIST 手写数字识别：

```bash
python Chapter09_SoftmaxClassifier/MNISTHandwrittenDigitRecognition/train_mnist.py --epochs 10
```

数据将从项目内的压缩 IDX 文件自动解压，详细模型说明与样本图见[第 09 章项目 README](./Chapter09_SoftmaxClassifier/MNISTHandwrittenDigitRecognition/README.md)。

运行第 10 章基础 CNN 配置对比：

```bash
jupyter lab
```

打开 `Chapter10_BasicCNN/MNIST_CNN_Configuration_Comparison.ipynb`。Notebook 会依次训练 CNN-S、CNN-M、CNN-L，并生成损失与测试准确率对比图；模型结构和实验说明见[第 10 章笔记](./Chapter10_BasicCNN/README.md)。

运行第 11 章高级 CNN 架构对比：

```bash
jupyter lab
```

打开 `Chapter11_AdvancedCNN/MNIST_Three_CNN_Architectures.ipynb`，比较普通 CNN、InceptionNet 与简化 ResNet。模型结构、实测结果和训练曲线见[第 11 章笔记](./Chapter11_AdvancedCNN/README.md)。

运行第 12 章 RNNCell + One-hot 字符序列预测：

```bash
python "Chapter12_BasicRNN/RNNCell + Onehot.py"
```

运行 RNN + One-hot 或 RNN + Embedding Notebook：

```bash
jupyter lab
```

打开以下任一文件：

```text
Chapter12_BasicRNN/RNN ＋ Onehot 练习.ipynb
Chapter12_BasicRNN/RNN + Embedding 练习.ipynb
```

## 数据集

课程相关数据文件大多保存在 [`datasets/`](./datasets/) 目录，并保留原始 `.csv.gz` 压缩格式；第 09 章的 MNIST 数据随实验项目存放，便于离线复现。

| 文件 | 状态 |
|---|---|
| [`diabetes.csv.gz`](./datasets/diabetes.csv.gz) | 已由第 07、08 章糖尿病二分类脚本引用 |
| [`names_train.csv.gz`](./datasets/names_train.csv.gz) | 已归档，当前脚本尚未引用 |
| [`names_test.csv.gz`](./datasets/names_test.csv.gz) | 已归档，当前脚本尚未引用 |
| [MNIST（压缩 IDX）](./Chapter09_SoftmaxClassifier/MNISTHandwrittenDigitRecognition/data/MNIST/raw/) | 第 09 章全连接分类、第 10 章基础 CNN 与第 11 章高级 CNN 架构对比 |

## 目录结构

```text
LiuErDaRenPyTorch/
├── Chapter01_Overview/
├── Chapter02_LinearModel/
├── ...
├── Chapter13_RNNClassifier/
├── datasets/
├── LICENSE
└── README.md
```

每章课件、笔记和实验代码放在对应章节目录中；后续学习内容也沿用这一结构补充。

## 版权说明

本仓库中的原创代码与笔记遵循 [MIT License](./LICENSE)。课程视频、课件及其中内容的版权归刘二大人及相关权利人所有，仅供学习与研究参考；MIT License 不自动适用于这些第三方课程资料与数据文件。

## 致谢

感谢刘二大人公开《PyTorch 深度学习实践》课程及学习资料。
