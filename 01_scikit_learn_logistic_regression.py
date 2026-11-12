"""
Listing 1: Logistic regression pipeline with cross-validation in Scikit-learn.
Dataset: Breast Cancer Wisconsin (included with scikit-learn).
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
X, y = load_breast_cancer(return_X_y=True)

# Split the data into training and testing sets
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Build a pipeline: standardisation followed by logistic regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Evaluate with 5-fold cross-validation on the training data
cv = cross_val_score(model, X_tr, y_tr, cv=5)

# Train on the full training set and evaluate on the test set
model.fit(X_tr, y_tr)

print(f"CV accuracy:   {cv.mean():.4f} +/- {cv.std():.4f}")
print(f"Test accuracy: {model.score(X_te, y_te):.4f}")
