from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


SEED = 42
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.01


class DiabetesDataset(Dataset):
    """读取糖尿病二分类数据，并返回单个特征张量与标签张量。"""

    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


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


def average_loss(model, data_loader, criterion):
    """在不记录梯度的情况下计算平均 loss 与二分类准确率。"""
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            probabilities = model(inputs)
            loss = criterion(probabilities, labels)
            predictions = (probabilities >= 0.5).float()

            loss_sum += loss.item() * inputs.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    return loss_sum / total, correct / total


def main():
    torch.manual_seed(SEED)

    data_path = Path(__file__).resolve().parents[1] / "datasets" / "diabetes.csv.gz"
    dataset = DiabetesDataset(data_path)

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(SEED)
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = Model()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    valid_losses = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for inputs, labels in train_loader:
            probabilities = model(inputs)
            loss = criterion(probabilities, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * inputs.size(0)

        train_loss = train_loss_sum / len(train_dataset)
        valid_loss, accuracy = average_loss(model, valid_loader, criterion)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Valid Loss: {valid_loss:.6f} | "
                f"Valid Accuracy: {accuracy * 100:.2f}%"
            )

    image_dir = Path(__file__).resolve().parent / "images"
    image_dir.mkdir(exist_ok=True)
    image_path = image_dir / "training_validation_loss.png"

    plt.figure(figsize=(8, 5))
    epochs = range(1, EPOCHS + 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="#2563eb")
    plt.plot(epochs, valid_losses, label="Validation Loss", color="#f97316")
    plt.xlabel("Epoch")
    plt.ylabel("BCELoss")
    plt.title(f"Training and Validation Loss (Accuracy: {accuracy * 100:.2f}%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Result image saved to: {image_path}")


if __name__ == "__main__":
    main()
