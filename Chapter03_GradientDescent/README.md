# Chapter 03: Gradient Descent

## Code

- `gradient_descent.py`: batch gradient descent example.
- `stochastic_gradient_descent.py`: stochastic gradient descent example.
- `mini_batch_gradient_descent.py`: mini-batch gradient descent example.

## Batch GD vs SGD vs Mini-batch GD

### Batch Gradient Descent

Batch Gradient Descent 每次更新参数时，会使用整个训练集计算一次平均梯度。

优点：

- 梯度方向比较稳定，loss 下降过程通常更平滑。
- 适合数据量较小、可以一次性放入内存的情况。

缺点：

- 数据量很大时，每次更新都要遍历全部样本，训练速度慢。
- 参数更新频率低，可能很久才更新一次。

### Stochastic Gradient Descent

SGD 每次只使用一个样本计算梯度并更新参数。

优点：

- 更新频率高，训练开始时往往下降很快。
- 不需要一次处理全部数据，适合大数据集。
- 梯度带有随机性，有时可以帮助跳出局部最优附近。

缺点：

- 单个样本的梯度噪声大，loss 曲线抖动明显。
- 参数更新方向不稳定，可能需要更仔细地调整学习率。

### Mini-batch Gradient Descent

Mini-batch Gradient Descent 每次取一小批样本计算平均梯度，是 Batch GD 和 SGD 之间的折中方案。

优点：

- 比 Batch GD 更新更频繁，训练速度更快。
- 比 SGD 梯度更稳定，loss 抖动更小。
- 适合 GPU 并行计算，是深度学习中最常用的训练方式。

缺点：

- 需要选择合适的 batch size。
- batch size 太小会接近 SGD，抖动较大；太大又会接近 Batch GD，更新较慢。
