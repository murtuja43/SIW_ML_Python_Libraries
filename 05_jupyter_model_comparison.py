"""
Listing 5: Data preparation, training, and evaluation in a notebook-style
workflow. Compares Logistic Regression, SVM, and Random Forest.

The report presents this as three Jupyter Notebook cells. It is provided
here as one Python script so it can also be run from a terminal.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Cell 1 and Cell 2: Load and prepare data
X, y = load_breast_cancer(return_X_y=True)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Fit the scaler only on the training set to prevent data leakage
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

# Cell 3: Construct, train, and evaluate models
models = {
    "Logistic regression": LogisticRegression(max_iter=1000),
    "SVM (RBF kernel)": SVC(random_state=42),
    "Random forest": RandomForestClassifier(300, random_state=42),
}

rows = []

for name, model in models.items():
    model.fit(X_tr, y_tr)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_te)
    else:
        scores = model.predict_proba(X_te)[:, 1]

    rows.append([
        name,
        accuracy_score(y_te, model.predict(X_te)),
        roc_auc_score(y_te, scores),
    ])

results = pd.DataFrame(
    rows,
    columns=["Model", "Accuracy", "ROC AUC"]
)

print(results.round(4).to_string(index=False))
