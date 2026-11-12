"""
Listing 4: A Sequential Keras classifier with dropout and early stopping.
Dataset: Breast Cancer Wisconsin (included with scikit-learn).

This is a standalone version of the report's example. It prepares and
standardises the data before training the Keras model.
"""

import keras
from keras import layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reproducibility
keras.utils.set_random_seed(42)

# Load and split the dataset
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardise using training data only
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

# Define a Sequential neural network
model = keras.Sequential([
    keras.Input(shape=(30,)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])

# Configure training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Stop training when validation loss stops improving
stop = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

# Train and evaluate
model.fit(
    X_tr,
    y_tr,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[stop],
    verbose=0
)

print("Parameters:", model.count_params())
print(
    "Test accuracy: %.4f"
    % model.evaluate(X_te, y_te, verbose=0)[1]
)
