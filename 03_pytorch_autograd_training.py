"""
Listing 3: Automatic differentiation and an explicit training loop in PyTorch.
Part (a): Differentiate x^2 + sin(x) at x = 2.
Part (b): Fit a linear regression model to simulated data.
"""

import numpy as np
import torch
from torch import nn

# Part (a): Automatic differentiation
x0 = torch.tensor(2.0, requires_grad=True)
(x0 ** 2 + torch.sin(x0)).backward()
print(f"f'(2) = {x0.grad.item():.4f}")  # 2x + cos(x) at x=2

# Part (b): Linear regression using nn.Linear and SGD
rng = np.random.default_rng(seed=0)
x = rng.uniform(-1, 1, size=(200, 1)).astype("float32")
y = (
    3.0 * x
    + 2.0
    + rng.normal(0, 0.3, size=(200, 1)).astype("float32")
)

# Convert NumPy arrays to PyTorch tensors
x = torch.from_numpy(x)
y = torch.from_numpy(y)

# Define model and optimiser
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(500):
    optimizer.zero_grad()                       # Reset gradients
    loss = nn.functional.mse_loss(model(x), y)  # Forward pass
    loss.backward()                              # Backpropagation
    optimizer.step()                             # Update parameters

print(
    f"w = {model.weight.item():.4f}, "
    f"b = {model.bias.item():.4f}"
)
