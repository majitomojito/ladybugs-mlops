"""Data loading and preprocessing utilities."""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load iris dataset and split into train/test sets.
    
    Args:
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Loading iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


def preprocess_data(
    X_train: np.ndarray, 
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess features using standard scaling.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    logger.info("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Data preprocessing complete")
    return X_train_scaled, X_test_scaled, scaler


def save_data(data_dir: Path, X_train: np.ndarray, X_test: np.ndarray, 
              y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Save dataset splits to disk."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(data_dir / "X_train.npy", X_train)
    np.save(data_dir / "X_test.npy", X_test)
    np.save(data_dir / "y_train.npy", y_train)
    np.save(data_dir / "y_test.npy", y_test)
    
    logger.info(f"Data saved to {data_dir}")
