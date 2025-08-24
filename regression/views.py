"""
import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt  # ✅ Added this

from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

# Paths to training and test data
DATA_DIR = os.path.join(settings.BASE_DIR, "regression", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# Load initial dataset
TRAIN_DF = pd.read_csv(TRAIN_PATH)
TEST_DF = pd.read_csv(TEST_PATH)

# Store points and model
POINTS = TRAIN_DF[['x', 'y']].values.tolist()
model_coefficients = None


def train_model():
    global model_coefficients
    if not POINTS:
        model_coefficients = None
        return None

    X = np.array([p[0] for p in POINTS])
    y = np.array([p[1] for p in POINTS])

    X_b = np.c_[np.ones((len(X), 1)), X]  # add bias term
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    model_coefficients = theta_best
    return theta_best


def plot_regression():
    global model_coefficients
    if not POINTS:
        return None

    X = np.array([p[0] for p in POINTS])
    y = np.array([p[1] for p in POINTS])

    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, color="blue", label="Data Points")

    if model_coefficients is not None:
        x_vals = np.linspace(min(X), max(X), 100)
        y_vals = model_coefficients[0] + model_coefficients[1] * x_vals
        plt.plot(x_vals, y_vals, color="red", label="Regression Line", linewidth=2)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression Visualization")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return image_base64


def index(request):
    train_model()  # Train on current dataset
    graph = plot_regression()
    return render(request, "regression/index.html", {"graph": graph, "points": POINTS})


def add_point(request):
    if request.method == "POST":
        try:
            x = float(request.POST.get("x"))
            y = float(request.POST.get("y"))
            POINTS.append([x, y])
            messages.success(request, f"Added point ({x}, {y})")
        except Exception:
            messages.error(request, "Invalid input. Please enter numeric values.")
    return redirect("index")


def reset_points(request):
    global POINTS, model_coefficients
    # Reset to original train.csv dataset
    df = pd.read_csv(TRAIN_PATH)
    POINTS = df[['x', 'y']].values.tolist()
    model_coefficients = None
    messages.success(request, "Dataset reset to original training data.")
    return redirect("index")
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

# Paths to training and test data (for future use)
DATA_DIR = os.path.join(settings.BASE_DIR, "regression", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# Load initial dataset (future use, not default now)
TRAIN_DF = pd.read_csv(TRAIN_PATH)
TEST_DF = pd.read_csv(TEST_PATH)

# Default simple dataset for visualization
POINTS = [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]
model_coefficients = None


def train_model():
    """Train linear regression using numpy (least squares)."""
    global model_coefficients
    X = np.array([p[0] for p in POINTS])
    y = np.array([p[1] for p in POINTS])

    # Add bias term
    X_b = np.c_[np.ones((len(X), 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    model_coefficients = theta_best
    return theta_best


def plot_regression():
    """Generate plot of regression line + points."""
    global model_coefficients
    if not POINTS:
        return None

    X = np.array([p[0] for p in POINTS])
    y = np.array([p[1] for p in POINTS])

    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, color="blue", label="Data Points")

    if model_coefficients is not None:
        x_vals = np.linspace(min(X), max(X), 100)
        y_vals = model_coefficients[0] + model_coefficients[1] * x_vals
        plt.plot(x_vals, y_vals, color="red", label="Regression Line")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return image_base64


def index(request):
    train_model()  # Train on current dataset
    graph = plot_regression()
    return render(request, "regression/index.html", {"graph": graph, "points": POINTS})


def add_point(request):
    if request.method == "POST":
        try:
            x = float(request.POST.get("x"))
            y = float(request.POST.get("y"))
            POINTS.append([x, y])
            messages.success(request, f"Added point ({x}, {y})")
        except:
            messages.error(request, "Invalid input. Please enter numeric values.")
    return redirect("index")


def reset_points(request):
    global POINTS, model_coefficients
    # Reset to original train.csv dataset
    df = pd.read_csv(TRAIN_PATH)
    POINTS = df[['x', 'y']].values.tolist()
    model_coefficients = None
    messages.success(request, "Dataset reset to original training data.")
    return redirect("index")
