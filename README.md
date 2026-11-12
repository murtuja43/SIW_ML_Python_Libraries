# Python Libraries for Machine Learning — SIW Code

**Author:** Kayes Md Golam Murtuja  
**University:** Al-Farabi Kazakh National University  
**Faculty:** Mechanics and Mathematics  
**Department:** Applied Mathematics  

This repository contains the six Python code listings from the SIW report:

**Python Libraries in the Jupyter Notebook Environment for Applying Machine Learning Methods (TensorFlow, Scikit-learn, PyTorch, Keras).**

## Project structure

| File | Description | Library / topic |
|---|---|---|
| `01_scikit_learn_logistic_regression.py` | Logistic regression pipeline and 5-fold cross-validation | Scikit-learn |
| `02_tensorflow_gradient_descent.py` | Linear regression trained with automatic differentiation | TensorFlow |
| `03_pytorch_autograd_training.py` | Automatic differentiation and an explicit training loop | PyTorch |
| `04_keras_neural_network_classifier.py` | Sequential neural network with dropout and early stopping | Keras |
| `05_jupyter_model_comparison.py` | Data preparation and comparison of three classifiers | Scikit-learn, pandas |
| `06_ar1_time_series_forecasting.py` | AR(1) simulation and time-series forecasting | NumPy, Scikit-learn |

## Requirements

- Python 3.10 or newer is recommended.
- Install the packages listed in `requirements.txt`.

## Installation

Create and activate a virtual environment (recommended), then run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the examples

Run any script from the project directory:

```bash
python 01_scikit_learn_logistic_regression.py
python 02_tensorflow_gradient_descent.py
python 03_pytorch_autograd_training.py
python 04_keras_neural_network_classifier.py
python 05_jupyter_model_comparison.py
python 06_ar1_time_series_forecasting.py
```

The examples use the Breast Cancer Wisconsin dataset bundled with scikit-learn or generate their own synthetic data. No separate dataset download is required.

## Running in Jupyter Notebook

Install Jupyter if needed:

```bash
pip install notebook
jupyter notebook
```

You can copy the contents of each script into a separate notebook cell, or open the scripts in JupyterLab.

## Notes

- Listing 4 is made standalone by including the data-loading, train/test split, and standardisation steps it references from Listing 5.
- Listing 5 is arranged as one executable Python script; the original report describes it as three notebook cells.
- Neural-network training results can vary slightly depending on package versions and hardware.
- For reproducibility, the examples set random seeds where appropriate.

## Academic source

These code examples are based on the practical listings in the SIW report by Kayes Md Golam Murtuja. The explanations and theoretical discussion are in the accompanying report.
