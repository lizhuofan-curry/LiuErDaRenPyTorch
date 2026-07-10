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
- [x] 第 01–04 章已有章内学习笔记。
- [x] 第 02–04 章已有 7 个可运行实验脚本。
- [ ] 第 05–13 章学习笔记与实验代码待继续补充。

## 章节导航

| 章节 | 课程主题 | 视频 | 本仓库资料 |
|---|---|---|---|
| [第 01 章](./Chapter01_Overview/) | 深度学习概论（Overview） | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=1) | [课件](./Chapter01_Overview/Lecture_01_Overview.pdf) · [笔记](./Chapter01_Overview/README.md) · 暂无代码 |
| [第 02 章](./Chapter02_LinearModel/) | 线性模型 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=2) | [课件](./Chapter02_LinearModel/Lecture_02_Linear_Model.pdf) · [笔记](./Chapter02_LinearModel/README.md) · [2D Loss](./Chapter02_LinearModel/plot_2d_loss_effect.py) · [3D Cost](./Chapter02_LinearModel/plot_3d_cost_surface_effect.py) |
| [第 03 章](./Chapter03_GradientDescent/) | 梯度下降算法 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=3) | [课件](./Chapter03_GradientDescent/Lecture_03_Gradient_Descent.pdf) · [笔记](./Chapter03_GradientDescent/README.md) · [Batch GD](./Chapter03_GradientDescent/gradient_descent.py) · [SGD](./Chapter03_GradientDescent/stochastic_gradient_descent.py) · [Mini-batch](./Chapter03_GradientDescent/mini_batch_gradient_descent.py) |
| [第 04 章](./Chapter04_BackPropagation/) | 反向传播 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=4) | [课件](./Chapter04_BackPropagation/Lecture_04_Back_Propagation.pdf) · [笔记](./Chapter04_BackPropagation/README.md) · [`y = wx`](./Chapter04_BackPropagation/backpropagation_y_wx.py) · [二次模型](./Chapter04_BackPropagation/backpropagation_quadratic.py) |
| [第 05 章](./Chapter05_LinearRegressionWithPyTorch/) | 使用 PyTorch 实现线性回归 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=5) | [课件](./Chapter05_LinearRegressionWithPyTorch/Lecture_05_Linear_Regression_with_PyTorch.pdf) · 暂无笔记与代码 |
| [第 06 章](./Chapter06_LogisticRegression/) | 逻辑回归 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=6) | [课件](./Chapter06_LogisticRegression/Lecture_06_Logistic_Regression.pdf) · 暂无笔记与代码 |
| [第 07 章](./Chapter07_MultipleDimensionInput/) | 多维特征输入 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=7) | [课件](./Chapter07_MultipleDimensionInput/Lecture_07_Multiple_Dimension_Input.pdf) · 暂无笔记与代码 |
| [第 08 章](./Chapter08_DatasetAndDataloader/) | Dataset 与 DataLoader | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=8) | [课件](./Chapter08_DatasetAndDataloader/Lecture_08_Dataset_and_Dataloader.pdf) · 暂无笔记与代码 |
| [第 09 章](./Chapter09_SoftmaxClassifier/) | Softmax 多分类器 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=9) | [课件](./Chapter09_SoftmaxClassifier/Lecture_09_Softmax_Classifier.pdf) · 暂无笔记与代码 |
| [第 10 章](./Chapter10_BasicCNN/) | 基础卷积神经网络（CNN） | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=10) | [课件](./Chapter10_BasicCNN/Lecture_10_Basic_CNN.pdf) · 暂无笔记与代码 |
| [第 11 章](./Chapter11_AdvancedCNN/) | 高级卷积神经网络（CNN） | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=11) | [课件](./Chapter11_AdvancedCNN/Lecture_11_Advanced_CNN.pdf) · 暂无笔记与代码 |
| [第 12 章](./Chapter12_BasicRNN/) | 基础循环神经网络（RNN） | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=12) | [课件](./Chapter12_BasicRNN/Lecture_12_Basic_RNN.pdf) · 暂无笔记与代码 |
| [第 13 章](./Chapter13_RNNClassifier/) | 循环神经网络分类器 | [观看](https://www.bilibili.com/video/BV1Y7411d7Ys?p=13) | [课件](./Chapter13_RNNClassifier/Lecture_13_RNN_Classifier.pdf) · 暂无笔记与代码 |

## 运行环境

仓库当前未锁定 Python 或依赖版本。运行现有示例需要 Python 3，并根据章节安装以下依赖：

```bash
python -m pip install numpy matplotlib torch
```

- 第 02 章使用 NumPy 和 Matplotlib。
- 第 03 章仅使用 Python 标准库。
- 第 04 章使用 PyTorch 自动求导。

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

## 数据集

课程相关数据文件统一保存在 [`datasets/`](./datasets/) 目录，并保留原始 `.csv.gz` 压缩格式：

| 文件 | 状态 |
|---|---|
| [`diabetes.csv.gz`](./datasets/diabetes.csv.gz) | 已归档，当前脚本尚未引用 |
| [`names_train.csv.gz`](./datasets/names_train.csv.gz) | 已归档，当前脚本尚未引用 |
| [`names_test.csv.gz`](./datasets/names_test.csv.gz) | 已归档，当前脚本尚未引用 |

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
