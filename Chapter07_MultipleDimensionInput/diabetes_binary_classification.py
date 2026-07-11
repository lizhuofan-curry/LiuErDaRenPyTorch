from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


# 固定随机种子，便于复现实验结果。
torch.manual_seed(42)


# 1. 准备数据：前 8 列是输入特征，最后一列是 0/1 标签。
data_path = Path(__file__).resolve().parents[1] / "datasets" / "diabetes.csv.gz"
xy = np.loadtxt(data_path, delimiter=",", dtype=np.float32)

x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


# 2. 定义模型：隐藏层使用 ReLU，输出层使用 Sigmoid。
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.sigmoid(self.linear3(x))


model = Model()


# 3. 二分类交叉熵衡量预测误差，SGD 根据梯度更新各层 W 和 b。
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# 4. 使用整个训练集训练；每轮建立新的计算图并更新一次参数。
loss_history = []

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d}, Loss: {loss.item():.6f}")


# 5. 将 Sigmoid 概率按 0.5 转成类别，并与真实标签逐个比较。
with torch.no_grad():
    probabilities = model(x_data)
    predictions = (probabilities >= 0.5).float()

    correct = (predictions == y_data).sum().item()
    total = y_data.numel()
    accuracy = correct / total

    print("预测正确数量：", correct)
    print("样本总数：", total)
    print(f"训练准确率：{accuracy * 100:.2f}%")
    print("真实标签中 0 的数量：", (y_data == 0).sum().item())
    print("真实标签中 1 的数量：", (y_data == 1).sum().item())
    print("预测为 0 的数量：", (predictions == 0).sum().item())
    print("预测为 1 的数量：", (predictions == 1).sum().item())

    true_counts = [
        (y_data == 0).sum().item(),
        (y_data == 1).sum().item(),
    ]
    predicted_counts = [
        (predictions == 0).sum().item(),
        (predictions == 1).sum().item(),
    ]


# 6. 保存训练效果图。
image_dir = Path(__file__).resolve().parent / "images"
image_dir.mkdir(exist_ok=True)
image_path = image_dir / "diabetes_training_result.png"

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(range(1, len(loss_history) + 1), loss_history, color="#2563eb")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCELoss")
axes[0].grid(alpha=0.25)

positions = np.arange(2)
width = 0.35
axes[1].bar(positions - width / 2, true_counts, width, label="True", color="#16a34a")
axes[1].bar(
    positions + width / 2,
    predicted_counts,
    width,
    label="Predicted",
    color="#f97316",
)
axes[1].set_title(f"Class Counts (Accuracy: {accuracy * 100:.2f}%)")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Samples")
axes[1].set_xticks(positions, ["0", "1"])
axes[1].legend()
axes[1].grid(axis="y", alpha=0.25)

figure.tight_layout()
figure.savefig(image_path, dpi=160, bbox_inches="tight")
plt.close(figure)

print("效果图已保存：", image_path)
