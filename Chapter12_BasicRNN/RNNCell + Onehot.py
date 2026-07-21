# 动手练习RNNcell
import torch
import torch.nn as nn
import torch.optim as optim

# 1.字符表
idx2char = ['e','h','l','o']

# 2.输入序列： hello
x_data  = [1,0,2,2,3]

# 3.目标序列： ohlol
y_data = [3,1,2,3,2]

# One-hot 查询表
one_hot_lookup = [
    [1,0,0,0], # 0 -> e
    [0,1,0,0], # 1 -> h
    [0,0,1,0], # 2 -> l
    [0,0,0,1]  # 3 -> o
]

# 将输入编号转换成0ne_hot
x_one_hot = [one_hot_lookup[x] for x in x_data]

# 基本参数
batch_size =1
seq_len =5
input_size =4
hidden_size =4

# 7.转化为PyTorch tensor
inputs = torch.tensor(x_one_hot,
                      dtype = torch.float32,
                      ).view(seq_len,batch_size,input_size)
labels = torch.tensor(y_data,
                      dtype = torch.long).view(seq_len,batch_size)

# 8.查看数据
print('输入字符',''.join(idx2char[x] for x in x_data ))
print('输出字符',''.join(idx2char[x] for x in y_data))

print("inputs.shape =", inputs.shape)
print("labels.shape =", labels.shape)

print("\ninputs：")
print(inputs)

print("\nlabels：")
print(labels)

# 先定义RNNCell模型

# RNNCell 一次只处理一个时间步，所以模型内部不需要知道seq_len
# seq_len是由外面的数据形状和循环控制的
class RNNCellModel(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super().__init__()

        # 保存三个基本参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # 创建RNNCell
        self.rnn_cell = torch.nn.RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )

    def forward(self, x, hidden):
        # 当前输入 x 加上一时刻 hidden
        # 得到新的hidden
        hidden = self.rnn_cell(x,hidden)
        return hidden

    def init_hidden(self):
        # 第一个时间步还没有历史信息
        # 所以初始hidden 全部设置为0
        hidden = torch.zeros(self.batch_size,self.hidden_size)
        return hidden

# 创建模型
net = RNNCellModel(input_size=input_size,
                   hidden_size=hidden_size,
                    batch_size=batch_size)
print(net)

# 多分类损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(net.parameters(),lr=0.05)

# 训练15轮
for epoch in range(15):

    # 1.清空上一轮保存的梯度
    optimizer.zero_grad()

    hidden = net.init_hidden()

    total_loss = 0

    predicted_chars = []

    for i in range(seq_len):

        # 导入当前输入和当前正确标签
        current_input = inputs[i]
        current_label = labels[i]
        # 通过当前输入和上一个hidden计算当前hidden
        hidden = net(current_input,hidden)
        # 计算交叉熵损失
        current_loss = criterion(hidden,current_label)
        # 总损失得全部加起来，因为这里的损失属于矩阵
        total_loss += current_loss
        # 我们这里只要取最大值的下标，不用取值为多少
        # 所以第一项用_表示
        _,predicted_index = hidden.max(dim=1)

        # 将当前得出的字母接到预测字符数组后面
        predicted_chars.append(idx2char[predicted_index.item()])

    average_loss = total_loss / seq_len
    # 我们希望模型根据5个时间步的平均错误程度来更新参数
    # 而不是让损失和序列长度绑在一起
    # 它的total_loss.backward()的区别在于调整的方向一样，但是移动幅度不同
    average_loss.backward()

    optimizer.step()

    predicted_string = ''.join(predicted_chars)
    print(
        f"Epoch [{epoch + 1}/15] "
        f"预测：{predicted_string} "
        f"平均损失：{average_loss.item():.4f}"
    )
