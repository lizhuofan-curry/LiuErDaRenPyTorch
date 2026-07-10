# Chapter 04: Back Propagation

## Code

- `backpropagation_y_wx.py`: uses PyTorch autograd to train the simple model `y = w * x`.
- `backpropagation_quadratic.py`: uses PyTorch autograd to train the quadratic model `y = w1 * x^2 + w2 * x + b`.

## Key Points

- `requires_grad=True` tells PyTorch to track operations on trainable parameters.
- `l.backward()` computes the gradient of loss with respect to each parameter.
- Parameter updates should be placed inside `torch.no_grad()` so the update itself is not tracked by autograd.
- `grad.zero_()` clears the accumulated gradient before the next backward pass.
