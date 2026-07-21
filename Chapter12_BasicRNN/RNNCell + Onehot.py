import torch
import torch.nn as nn
import torch.optim as optim


# 1. 字符表
idx2char = ["e", "h", "l", "o"]

# 2. 输入序列：hello
x_data = [1, 0, 2, 2, 3]

# 3. 目标序列：ohlol
y_data = [3, 1, 2, 3, 2]

# One-hot 查询表
one_hot_lookup = [
    [1, 0, 0, 0],  # 0 -> e
    [0, 1, 0, 0],  # 1 -> h
    [0, 0, 1, 0],  # 2 -> l
    [0, 0, 0, 1],  # 3 -> o
]

# 将输入编号转换成 one-hot
x_one_hot = [one_hot_lookup[x] for x in x_data]

# 基本参数
batch_size = 1
seq_len = 5
input_size = 4
hidden_size = 4

# 转换为 PyTorch Tensor
inputs = torch.tensor(x_one_hot, dtype=torch.float32).view(
    seq_len, batch_size, input_size
)
labels = torch.tensor(y_data, dtype=torch.long).view(seq_len, batch_size)

# 查看数据
print("输入字符：", "".join(idx2char[x] for x in x_data))
print("输出字符：", "".join(idx2char[x] for x in y_data))
print("inputs.shape =", inputs.shape)
print("labels.shape =", labels.shape)
print("\ninputs：")
print(inputs)
print("\nlabels：")
print(labels)


class RNNCellModel(nn.Module):
    """使用 RNNCell 逐个处理序列中的时间步。"""

    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()

        # 保存三个基本参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # RNNCell 一次只处理一个时间步，因此内部不需要 seq_len
        self.rnn_cell = nn.RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )

    def forward(self, x, hidden):
        # 根据当前输入 x 和上一时刻 hidden 得到新的 hidden
        hidden = self.rnn_cell(x, hidden)
        return hidden

    def init_hidden(self):
        # 第一个时间步还没有历史信息，因此初始 hidden 全部设为 0
        return torch.zeros(self.batch_size, self.hidden_size)


# 创建模型
net = RNNCellModel(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_size=batch_size,
)
print(net)

# 多分类损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.05)

# 训练 15 轮
for epoch in range(15):
    # 清空上一轮保存的梯度
    optimizer.zero_grad()
    hidden = net.init_hidden()
    total_loss = 0
    predicted_chars = []

    for i in range(seq_len):
        # 取当前时间步的输入和正确标签
        current_input = inputs[i]
        current_label = labels[i]

        # 根据当前输入和上一时刻 hidden 计算当前 hidden
        hidden = net(current_input, hidden)

        # hidden 的四个分量同时作为四个字符类别的 logits
        current_loss = criterion(hidden, current_label)
        total_loss += current_loss

        # 只取最大值的下标，不需要最大值本身
        _, predicted_index = hidden.max(dim=1)
        predicted_chars.append(idx2char[predicted_index.item()])

    # 使用五个时间步的平均损失，使更新幅度不随序列长度线性变化
    average_loss = total_loss / seq_len
    average_loss.backward()
    optimizer.step()

    predicted_string = "".join(predicted_chars)
    print(
        f"Epoch [{epoch + 1}/15] "
        f"预测：{predicted_string} "
        f"平均损失：{average_loss.item():.4f}"
    )
