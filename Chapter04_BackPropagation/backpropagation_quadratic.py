import torch


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.tensor([1.0], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)


def forward(x):
    return w1 * x**2 + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)

        l.backward()

        print(
            "epoch:",
            epoch,
            "x:",
            x,
            "y:",
            y,
            "w1.grad:",
            w1.grad.item(),
            "w2.grad:",
            w2.grad.item(),
            "b.grad:",
            b.grad.item(),
        )

        with torch.no_grad():
            w1 -= 0.01 * w1.grad
            w2 -= 0.01 * w2.grad
            b -= 0.01 * b.grad

        w1.grad.zero_()
        w2.grad.zero_()
        b.grad.zero_()


print("训练后的参数：")
print("w1 =", w1.item())
print("w2 =", w2.item())
print("b =", b.item())

print("当 x = 4 时，预测值 =", forward(4).item())
