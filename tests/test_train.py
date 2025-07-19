import json
import os
import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import train_model  # We'll create this function

CONFIG_PATH = "config/config.json"

def test_config_loading():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    assert "C" in config and isinstance(config["C"], float)
    assert "solver" in config and isinstance(config["solver"], str)
    assert "max_iter" in config and isinstance(config["max_iter"], int)

def test_model_training():
    digits = load_digits()
    X, y = digits.data, digits.target

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model = train_model(X, y, config)

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert hasattr(model, "classes_")

def test_model_accuracy():
    digits = load_digits()
    X, y = digits.data, digits.target

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model = train_model(X, y, config)
    score = model.score(X, y)

    assert score > 0.85  # baseline threshold
