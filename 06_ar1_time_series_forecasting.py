"""
Listing 6: Simulation and forecasting of an AR(1) process.

The process is X_t = 0.8 X_(t-1) + epsilon_t, where epsilon_t is
standard normal noise. The first 800 observations are used for training;
the remaining observations are used for testing in chronological order.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# Simulate an AR(1) process with phi = 0.8 and sigma = 1
rng = np.random.default_rng(seed=1)
x = np.zeros(1000)

for t in range(1, 1000):
    x[t] = 0.8 * x[t - 1] + rng.normal()

# Use the previous value to predict the next value
X_lag = x[:-1].reshape(-1, 1)
y_next = x[1:]

# Train using the first 800 observations (time-ordered split)
fit = LinearRegression().fit(X_lag[:800], y_next[:800])

# Evaluate on the remaining observations
err = y_next[800:] - fit.predict(X_lag[800:])

print(
    f"phi_hat = {fit.coef_[0]:.4f}, "
    f"forecast RMSE = {np.sqrt(np.mean(err ** 2)):.4f}"
)
