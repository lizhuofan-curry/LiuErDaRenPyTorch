# Chapter 05：用 PyTorch 实现线性回归

本章使用同一个一元线性回归任务比较七种 `torch.optim` 优化器。所有脚本的模型、数据、损失函数和训练轮数都相同，唯一的核心变量是参数更新方法。

- 训练数据：`(1, 2)`、`(2, 4)`、`(3, 6)`
- 模型：`y_pred = w * x + b`
- 目标：学习到 `w ≈ 2`、`b ≈ 0`
- 损失函数：`MSELoss(reduction="sum")`
- 学习率：`lr=0.01`
- 训练轮数：1000

## 文件导航

| 优化器 | 示例文件 | 所属思路 |
|---|---|---|
| SGD | [`optim.SGD版.py`](./optim.SGD版.py) | 基础梯度下降 |
| ASGD | [`optim.ASGD版.py`](./optim.ASGD版.py) | SGD + 学习率衰减 + 参数平均 |
| Adagrad | [`optim.Adagrad版.py`](./optim.Adagrad版.py) | 累积历史平方梯度，自适应步长 |
| RMSprop | [`optim.RMSprop版.py`](./optim.RMSprop版.py) | 平方梯度的指数移动平均 |
| Adam | [`optim.Adam版.py`](./optim.Adam版.py) | 一阶动量 + 二阶自适应缩放 |
| Adamax | [`optim.Adamax版.py`](./optim.Adamax版.py) | Adam 的无穷范数变体 |
| Rprop | [`optim.Rprop版.py`](./optim.Rprop版.py) | 根据梯度符号独立调整步长 |

课件见 [`Lecture_05_Linear_Regression_with_PyTorch.pdf`](./Lecture_05_Linear_Regression_with_PyTorch.pdf)。

## 共同的训练流程

七个脚本都遵循相同的 PyTorch 训练步骤：

```python
y_pred = model(x_data)
loss = criterion(y_pred, y_data)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

三步更新的含义如下：

1. `optimizer.zero_grad()` 清除上一轮累积的梯度。
2. `loss.backward()` 通过反向传播计算 `w` 和 `b` 的梯度。
3. `optimizer.step()` 根据所选算法更新 `w` 和 `b`。

因此，这些优化器解决的是同一个问题：**已知当前梯度后，应该沿哪个方向走多远。** 它们之间的差异主要在于是否保存历史梯度、如何缩放每个参数的步长，以及是否使用动量或参数平均。

> `torch.optim.SGD` 的名字虽然包含“随机”，但优化器本身不会创建随机小批量。本章每轮把三个样本一次性传给模型，因此实际使用的是全批量梯度。是否为随机、小批量或全批量训练，取决于数据如何送入模型。

## 优化器之间的关系

```text
torch.optim.Optimizer
├── SGD 系列
│   ├── SGD：直接用当前梯度更新
│   └── ASGD：在 SGD 基础上加入学习率衰减和参数平均
├── 自适应学习率系列
│   ├── Adagrad：累加所有历史平方梯度
│   ├── RMSprop：用指数移动平均代替无限累加
│   ├── Adam：在自适应缩放上同时加入一阶动量
│   └── Adamax：用无穷范数改写 Adam 的二阶缩放
└── 符号驱动系列
    └── Rprop：忽略梯度大小，只根据符号变化调整步长
```

可以把几条主要演进关系理解为：

- **SGD → ASGD**：保留 SGD 的基本更新思路，再加入衰减和参数平均，以降低后期波动。
- **Adagrad → RMSprop**：Adagrad 的平方梯度会一直累积，步长可能越来越小；RMSprop 通过指数移动平均逐渐遗忘很久以前的梯度。
- **RMSprop + 动量思想 → Adam**：Adam 同时估计梯度的一阶矩和二阶矩，并进行偏差修正。
- **Adam → Adamax**：Adamax 用无穷范数维护梯度尺度，是 Adam 家族的另一种稳定缩放方式。
- **Rprop 独立成支**：它只关心相邻梯度的符号是否一致，不使用梯度绝对值来决定更新幅度。

## 核心区别

| 优化器 | 如何决定更新量 | 每个参数自适应步长 | 主要额外状态 | 特点与注意事项 |
|---|---|---:|---|---|
| SGD | 当前梯度乘以统一学习率 | 否 | 无；启用 momentum 时增加动量 | 简单、内存少、容易解释；对学习率和特征尺度较敏感。本例没有启用 momentum。 |
| ASGD | SGD 更新、学习率衰减与参数平均 | 部分 | 平均参数及步数状态 | 适合观察长期平均效果；PyTorch 默认 `t0=1_000_000`，本例 1000 轮尚未进入默认的参数平均阶段。 |
| Adagrad | 用累计平方梯度缩小各参数步长 | 是 | 平方梯度累加器 | 对稀疏或不同尺度参数有帮助；历史只增不减，后期有效学习率可能过早变得很小。 |
| RMSprop | 用近期平方梯度的指数移动平均缩放更新 | 是 | 平方梯度移动平均 | 修正了 Adagrad 永久累积的问题，能更快适应近期梯度尺度。 |
| Adam | 一阶矩估计负责方向与动量，二阶矩估计负责缩放 | 是 | 一阶矩和二阶矩 | 通常前期收敛较快、使用广泛；状态和内存开销高于 SGD。 |
| Adamax | 一阶矩配合梯度无穷范数进行缩放 | 是 | 一阶矩和无穷范数状态 | 属于 Adam 变体，对梯度尺度较大或变化明显的情况可作为替代方案。 |
| Rprop | 符号连续一致则增大步长，符号翻转则减小步长 | 是 | 前一轮梯度和参数步长 | 不受梯度绝对值影响，适合本章这种稳定的全批量小问题；面对噪声较大的随机小批量梯度时，符号频繁变化可能使步长不稳定。 |

## 为什么相同的 `lr=0.01` 会得到不同结果

这里的 `lr` 不能理解成七种算法完全相同的实际步长：

- SGD 直接把 `lr` 乘在梯度上。
- Adagrad、RMSprop、Adam 和 Adamax 会再用历史梯度统计量缩放它。
- ASGD 还包含学习率衰减和参数平均相关参数。
- Rprop 把 `lr` 当作初始参数步长，之后根据梯度符号增大或减小。

此外，当前脚本没有固定随机种子，`torch.nn.Linear` 的初始 `w` 和 `b` 每次可能不同；`MSELoss(reduction="sum")` 也会让梯度大小随样本数变化。因此，某一次运行的最终损失或预测值只能说明“这组初值和超参数下的表现”，不能直接作为优化器优劣排名。

## 如何做更公平的比较

1. 使用 `torch.manual_seed(...)` 固定随机性。
2. 让所有模型加载同一份初始 `state_dict`。
3. 保持数据、批次顺序、损失函数和训练轮数一致。
4. 分别为每种优化器调节合适的学习率，而不是强制使用同一个值。
5. 重复多个随机种子，比较平均最终损失、收敛速度和波动，而不是只看一次预测。
6. 记录并绘制每轮 loss 曲线，观察前期速度、后期稳定性和是否停滞。

## 选择建议

- 想学习最基础、最透明的更新过程：从 **SGD** 开始。
- 想观察 SGD 的衰减与长期平均思想：尝试 **ASGD**，并根据训练轮数调整 `t0`。
- 特征稀疏，或不同参数的梯度尺度差异明显：可以尝试 **Adagrad**。
- Adagrad 后期步长衰减过快：可以尝试 **RMSprop**。
- 需要一个常用的自适应优化器作为起点：通常先试 **Adam**。
- 想比较 Adam 的无穷范数变体：使用 **Adamax**。
- 目标函数稳定、使用全批量梯度，并想忽略梯度绝对值：可以实验 **Rprop**。

这些建议不是固定规则。实际项目仍应根据数据、模型、批量大小和验证集指标选择优化器与超参数。

## 运行示例

从仓库根目录执行：

```bash
python "Chapter05_LinearRegressionWithPyTorch/optim.SGD版.py"
python "Chapter05_LinearRegressionWithPyTorch/optim.ASGD版.py"
python "Chapter05_LinearRegressionWithPyTorch/optim.Adagrad版.py"
python "Chapter05_LinearRegressionWithPyTorch/optim.RMSprop版.py"
python "Chapter05_LinearRegressionWithPyTorch/optim.Adam版.py"
python "Chapter05_LinearRegressionWithPyTorch/optim.Adamax版.py"
python "Chapter05_LinearRegressionWithPyTorch/optim.Rprop版.py"
```

## 参考资料

- [PyTorch `torch.optim` 总览](https://docs.pytorch.org/docs/stable/optim.html)
- [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)
- [ASGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.ASGD.html)
- [Adagrad](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adagrad.html)
- [RMSprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
- [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- [Adamax](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adamax.html)
- [Rprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.Rprop.html)
