# Mini-batch gradient descent
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


batch_size = 2

print("Predict(before training)", 4, forward(4))

for epoch in range(100):
    for i in range(0, len(x_data), batch_size):
        x_batch = x_data[i : i + batch_size]
        y_batch = y_data[i : i + batch_size]

        grad = gradient(x_batch, y_batch)
        w = w - 0.01 * grad

print("Predict(after training)", 4, forward(4))
