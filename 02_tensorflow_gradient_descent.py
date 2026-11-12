"""
Listing 2: Gradient descent with tf.GradientTape and tf.function.
Task: Estimate y = wx + b from simulated data.
"""

import numpy as np
import tensorflow as tf

# Generate data: y = 3x + 2 + noise
rng = np.random.default_rng(seed=0)
x = rng.uniform(-1, 1, size=(200, 1)).astype("float32")
y = (
    3.0 * x
    + 2.0
    + rng.normal(0, 0.3, size=(200, 1)).astype("float32")
)

# Trainable parameters
w = tf.Variable(0.0)
b = tf.Variable(0.0)


@tf.function
def train_step():
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(w * x + b - y))

    # Calculate gradients
    dw, db = tape.gradient(loss, [w, b])

    # Update parameters
    w.assign_sub(0.1 * dw)
    b.assign_sub(0.1 * db)

    return loss


# Train for 500 steps
for step in range(500):
    loss = train_step()

print(
    f"w = {w.numpy():.4f}, "
    f"b = {b.numpy():.4f}, "
    f"MSE = {loss.numpy():.4f}"
)
