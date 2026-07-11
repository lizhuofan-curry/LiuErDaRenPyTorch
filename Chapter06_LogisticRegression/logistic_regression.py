import matplotlib.pyplot as plt
import numpy as np
import torch


torch.manual_seed(0)

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(epoch, loss.item())

x = np.linspace(0, 10, 200)
x_t = torch.tensor(x, dtype=torch.float32).view(200, 1)

with torch.no_grad():
    y_t = model(x_t)
y = y_t.numpy()

plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], color="red")
plt.xlabel("Hours")
plt.ylabel("Probability of Pass")
plt.grid()
plt.show()
