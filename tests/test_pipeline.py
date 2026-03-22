"""Unit tests for the ML pipeline."""
import numpy as np
from sklearn.datasets import load_iris

from src.data import load_data, preprocess_data
from src.models import train_model, evaluate_model


def test_load_data():
    """Test data loading."""
    X_train, X_test, y_train, y_test = load_data()
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_preprocess_data():
    """Test data preprocessing."""
    X_train, X_test, y_train, y_test = load_data()
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.allclose(np.mean(X_train_scaled), 0, atol=1e-6)


def test_train_model():
    """Test model training."""
    X_train, _, y_train, _ = load_data()
    _, _, scaler = preprocess_data(X_train, X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = train_model(X_train_scaled, y_train)
    assert model is not None
    assert hasattr(model, "predict")


def test_evaluate_model():
    """Test model evaluation."""
    X_train, X_test, y_train, y_test = load_data()
    X_train_scaled, X_test_scaled, _ = preprocess_data(X_train, X_test)
    
    model = train_model(X_train_scaled, y_train)
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1
