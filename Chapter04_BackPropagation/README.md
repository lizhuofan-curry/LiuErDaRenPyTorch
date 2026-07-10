# Chapter 04: Back Propagation

## Code

- `backpropagation_y_wx.py`: uses PyTorch autograd to train the simple model `y = w * x`.

## Key Points

- `requires_grad=True` tells PyTorch to track operations on `w`.
- `l.backward()` computes the gradient of loss with respect to `w`.
- Parameter updates should be placed inside `torch.no_grad()` so the update itself is not tracked by autograd.
- `w.grad.zero_()` clears the accumulated gradient before the next backward pass.
