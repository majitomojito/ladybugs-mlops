"""Model training and evaluation utilities."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def train_model(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model


def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
    }
    
    logger.info(f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def save_metrics(metrics_dir: Path, metrics: Dict[str, float]) -> None:
    """Save metrics to JSON file."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = metrics_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_file}")
