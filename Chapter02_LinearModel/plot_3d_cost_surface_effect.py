import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)

W, B = np.meshgrid(w_range, b_range)
Cost = np.zeros(W.shape)

for x_val, y_val in zip(x_data, y_data):
    y_pred_val = x_val * W + B
    loss_val = (y_pred_val - y_val) ** 2
    Cost += loss_val

Cost /= 3.0

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(W, B, Cost, cmap="coolwarm", edgecolor="none")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("Cost Value")
ax.set_title("3D Cost Surface")

fig.colorbar(surf, shrink=0.5, aspect=5)

output_dir = Path(__file__).resolve().parent / "images"
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "linear_model_cost_surface.png", dpi=200, bbox_inches="tight")
plt.show()
