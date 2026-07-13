from __future__ import annotations

import argparse
import gzip
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"


def ensure_raw_mnist() -> None:
    raw = DATA_ROOT / "MNIST" / "raw"
    for archive in raw.glob("*.gz"):
        target = raw / archive.stem
        if not target.exists():
            with gzip.open(archive, "rb") as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP digit classifier on the bundled MNIST data.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_raw_mnist()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=transform)
    test_set = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    losses, accuracies = [], []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))
        accuracies.append(evaluate(model, test_loader, device))
        print(f"Epoch {epoch + 1:02d}/{args.epochs} | loss={losses[-1]:.4f} | test_acc={accuracies[-1]:.2f}%")

    output = PROJECT_ROOT / "images" / "training_metrics.png"
    output.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(losses, marker="o"); axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(accuracies, marker="o"); axes[1].set(title="Test accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    fig.tight_layout(); fig.savefig(output, dpi=160)
    print(f"Metrics chart written to {output}")


if __name__ == "__main__":
    main()
