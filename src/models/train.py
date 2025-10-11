"""Model training pipeline."""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.features.tfidf import TFIDFFeatureExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trainer for fake news detection models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: str = 'balanced'
    ) -> Dict[str, Any]:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression...")
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Train model
        model = LogisticRegression(
            random_state=self.random_state,
            class_weight=class_weight_dict,
            max_iter=1000,
            solver='liblinear'  # Good for small datasets
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='f1_macro'
        )
        
        metrics = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'model_name': 'LogisticRegression'
        }
        
        self.models['logistic_regression'] = model
        self.metrics['logistic_regression'] = metrics
        
        logger.info(f"Logistic Regression - Train: {train_score:.4f}, "
                   f"Val: {val_score:.4f}, CV F1: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        return metrics
    
    def train_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: str = 'balanced'
    ) -> Dict[str, Any]:
        """Train Linear SVM model."""
        logger.info("Training Linear SVM...")
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Train model
        model = LinearSVC(
            random_state=self.random_state,
            class_weight=class_weight_dict,
            max_iter=1000
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='f1_macro'
        )
        
        metrics = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'model_name': 'LinearSVM'
        }
        
        self.models['svm'] = model
        self.metrics['svm'] = metrics
        
        logger.info(f"Linear SVM - Train: {train_score:.4f}, "
                   f"Val: {val_score:.4f}, CV F1: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        return metrics
    
    def select_best_model(self) -> None:
        """Select the best model based on validation F1 score."""
        best_score = -1
        best_name = None
        
        for name, metrics in self.metrics.items():
            score = metrics['cv_f1_mean']
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"Best model: {best_name} (CV F1: {best_score:.4f})")
    
    def save_models(self, output_dir: Path) -> None:
        """Save all trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = output_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = output_dir / "best_model.joblib"
            joblib.dump(self.best_model, best_model_path)
            logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")
    
    def save_metrics(self, output_dir: Path) -> None:
        """Save training metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'models': self.metrics
        }
        
        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
        
        return metadata


def train_models(
    features_dir: Path,
    output_dir: Path,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train multiple models and select the best one."""
    
    logger.info("Loading features...")
    
    # Load features
    X_train = np.load(features_dir / "X_train.npy")
    y_train = np.load(features_dir / "y_train.npy")
    X_val = np.load(features_dir / "X_val.npy")
    y_val = np.load(features_dir / "y_val.npy")
    
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=random_state)
    
    # Train models
    logger.info("Training models...")
    trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
    trainer.train_svm(X_train, y_train, X_val, y_val)
    
    # Select best model
    trainer.select_best_model()
    
    # Save models and metrics
    trainer.save_models(output_dir / "models")
    metadata = trainer.save_metrics(output_dir / "reports")
    
    return metadata


def main() -> None:
    """Main function to train models."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    features_dir = project_root / "artifacts" / "vectorizers"
    output_dir = project_root / "artifacts"
    
    # Check if features exist
    required_files = [
        features_dir / "X_train.npy",
        features_dir / "y_train.npy",
        features_dir / "X_val.npy",
        features_dir / "y_val.npy"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"Feature file not found: {file_path}")
            logger.error("Please run feature extraction first")
            sys.exit(1)
    
    # Train models
    metadata = train_models(
        features_dir=features_dir,
        output_dir=output_dir,
        random_state=42
    )
    
    # Print summary
    logger.info("\nTraining Summary:")
    for model_name, metrics in metadata['models'].items():
        logger.info(f"{model_name}:")
        logger.info(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"  Val Accuracy: {metrics['val_accuracy']:.4f}")
        logger.info(f"  CV F1: {metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}")
    
    logger.info(f"\nBest Model: {metadata['best_model']}")


if __name__ == "__main__":
    main()
