# 第 12 章：基础循环神经网络（RNN）

本章通过字符序列预测练习理解循环神经网络如何沿时间步传递隐藏状态。练习从 RNNCell + One-hot 开始，继续使用 `torch.nn.RNN` 一次处理完整序列，并进一步用可训练的 Embedding 向量替代 One-hot 编码。

## 内容

- [`Lecture_12_Basic_RNN.pdf`](./Lecture_12_Basic_RNN.pdf)：第 12 章课程课件。
- [`RNNCell + Onehot.py`](./RNNCell%20%2B%20Onehot.py)：使用 RNNCell 和 One-hot 编码完成字符序列预测。
- [`RNN ＋ Onehot 练习.ipynb`](./RNN%20%EF%BC%8B%20Onehot%20%E7%BB%83%E4%B9%A0.ipynb)：使用 `nn.RNN`、One-hot 编码和 Linear 层完成字符序列预测的 Notebook。
- [`RNN + Embedding 练习.ipynb`](./RNN%20%2B%20Embedding%20%E7%BB%83%E4%B9%A0.ipynb)：使用 `nn.Embedding`、`nn.RNN` 和 Linear 层完成字符序列预测的 Notebook。

## 练习目标

模型读取输入序列 `hello`，学习输出目标序列 `ohlol`：

```text
输入：h e l l o
目标：o h l o l
```

字符表及 One-hot 编码如下：

| 编号 | 字符 | One-hot |
|---:|:---:|:---:|
| 0 | e | `[1, 0, 0, 0]` |
| 1 | h | `[0, 1, 0, 0]` |
| 2 | l | `[0, 0, 1, 0]` |
| 3 | o | `[0, 0, 0, 1]` |

## 张量形状

示例采用 `seq_len = 5`、`batch_size = 1`、`input_size = 4` 和 `hidden_size = 4`：

```text
inputs: (seq_len, batch_size, input_size) = (5, 1, 4)
labels: (seq_len, batch_size)             = (5, 1)
hidden: (batch_size, hidden_size)         = (1, 4)
```

`RNNCell` 一次只接收一个时间步。外层循环依次取出 `inputs[i]`，把当前输入和上一时刻的 `hidden` 送入模型：

```text
x(t) + hidden(t-1) → RNNCell → hidden(t)
```

初始隐藏状态使用全零张量。五个时间步的交叉熵损失相加后取平均，再进行一次反向传播，从而完成沿整个序列的梯度计算。

## 模型输出说明

本练习中 `hidden_size` 与字符类别数都等于 4，因此直接把每个时间步的隐藏状态作为四个类别的 logits，并使用 `CrossEntropyLoss` 训练。

更通用的写法通常会在 RNN 输出后增加一个全连接层：

```python
self.fc = nn.Linear(hidden_size, num_classes)
```

这样隐藏状态维度就不必与类别数相同。

## RNN + One-hot

Notebook 将完整输入张量一次送入 `nn.RNN`，得到所有时间步的隐藏状态，再通过 Linear 层把 8 维隐藏状态映射为 4 个字符类别分数。训练前会把序列长度和 batch 维度合并，以符合 `CrossEntropyLoss` 常用的输入形状。

| 示例 | 序列处理方式 | 分类输出 |
|---|---|---|
| `RNNCell + Onehot.py` | 外层循环逐个时间步处理 | 直接使用 4 维隐藏状态 |
| `RNN ＋ Onehot 练习.ipynb` | `nn.RNN` 一次处理完整序列 | Linear：8 维隐藏状态 → 4 个类别 |
| `RNN + Embedding 练习.ipynb` | Embedding 后由 `nn.RNN` 处理完整序列 | Linear：8 维隐藏状态 → 4 个类别 |

## RNN + Embedding

Embedding 版本直接输入字符编号，输入张量形状由 One-hot 版本的 `(5, 1, 4)` 变为 `(5, 1)`，数据类型为 `torch.long`。`nn.Embedding` 将每个编号查询为 6 维向量，再依次经过 RNN 和 Linear 层：

```text
字符编号 (5, 1)
→ Embedding (5, 1, 6)
→ RNN (5, 1, 8)
→ Linear (5, 1, 4)
```

Embedding 查询表属于模型参数，会与 RNN 和 Linear 层一起通过反向传播更新。

## 运行方式

在仓库根目录执行：

```bash
python "Chapter12_BasicRNN/RNNCell + Onehot.py"
```

脚本会打印输入与标签张量、模型结构，以及每轮的预测字符串和平均损失。模型参数为随机初始化，因此不同运行的预测过程可能不同；正常情况下损失应整体下降。

运行 RNN + One-hot Notebook：

```bash
jupyter lab
```

打开 `Chapter12_BasicRNN/RNN ＋ Onehot 练习.ipynb`。

运行 RNN + Embedding Notebook：

```bash
jupyter lab
```

打开 `Chapter12_BasicRNN/RNN + Embedding 练习.ipynb`。

## 验证点

- `inputs.shape` 应为 `torch.Size([5, 1, 4])`。
- `labels.shape` 应为 `torch.Size([5, 1])`。
- 每个时间步的输入和隐藏状态形状都满足 `RNNCell` 的要求。
- `CrossEntropyLoss` 接收形状为 `(1, 4)` 的 logits 和形状为 `(1,)` 的标签。
- 平均损失可以正常反向传播，并在训练过程中整体下降。
- Embedding 版本的输入应为 `torch.long`，形状应为 `torch.Size([5, 1])`。
- Embedding 后的张量形状应为 `torch.Size([5, 1, 6])`。
