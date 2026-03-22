"""Main ML pipeline orchestration."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import yaml

from src.data import load_data, preprocess_data, save_data
from src.models import train_model, evaluate_model, save_metrics

logger = logging.getLogger(__name__)


def _ensure_logging_config() -> None:
    """Ensure logging is configured in caller environments."""
    root_logger = logging.getLogger()
    
    # Only add handler if none exist
    if not root_logger.handlers:
        import sys
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(config_path: str = "config/config.yaml") -> None:
    """
    Execute the complete ML pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    _ensure_logging_config()
    logger.info("=" * 60)
    logger.info("Starting MLOps Pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")
    
    # Create output directories
    data_dir = Path(config["paths"]["data"])
    models_dir = Path(config["paths"]["models"])
    metrics_dir = Path(config["paths"]["metrics"])
    
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    logger.info("\n[Step 1/4] Loading data...")
    X_train, X_test, y_train, y_test = load_data(
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    save_data(data_dir, X_train, X_test, y_train, y_test)
    
    # 2. Preprocess data
    logger.info("\n[Step 2/4] Preprocessing data...")
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    # Save scaler
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # 3. Train model
    logger.info("\n[Step 3/4] Training model...")
    model = train_model(X_train_scaled, y_train, random_state=config["model"]["random_state"])
    
    # Save model
    model_path = models_dir / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 4. Evaluate model
    logger.info("\n[Step 4/4] Evaluating model...")
    metrics = evaluate_model(model, X_test_scaled, y_test)
    save_metrics(metrics_dir, metrics)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete - Model Performance Summary")
    logger.info("=" * 60)
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name.capitalize()}: {metric_value:.4f}")
    logger.info("=" * 60)
    
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_pipeline()
