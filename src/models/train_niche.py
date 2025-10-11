"""Per-niche model training pipeline."""

import sys
import yaml
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
from sklearn.preprocessing import LabelEncoder

import argparse

from src.features.tfidf import TFIDFFeatureExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_niche_config(niche: str) -> Dict[str, Any]:
    """Load configuration for a specific niche."""
    config_path = Path(__file__).parent.parent.parent / "configs" / f"{niche}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_minimum_samples(df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """Check if dataset meets minimum sample requirements."""
    min_samples = config.get('min_samples_per_class', 300)
    
    positive_label = config['positive_label']
    negative_label = config['negative_label']
    
    positive_count = len(df[df['label'] == positive_label])
    negative_count = len(df[df['label'] == negative_label])
    
    if positive_count < min_samples or negative_count < min_samples:
        logger.error(f"Insufficient samples for training {config['niche']}:")
        logger.error(f"  {positive_label}: {positive_count} (minimum: {min_samples})")
        logger.error(f"  {negative_label}: {negative_count} (minimum: {min_samples})")
        logger.error("Please collect more data or adjust heuristics")
        return False
    
    logger.info(f"Sample counts for {config['niche']}:")
    logger.info(f"  {positive_label}: {positive_count}")
    logger.info(f"  {negative_label}: {negative_count}")
    
    return True


def preprocess_niche_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess niche data for training."""
    from src.data.clean import clean_text, tokenize_text
    
    logger.info("Cleaning and tokenizing text...")
    
    # Clean text
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Tokenize text
    df['tokens'] = df['text_clean'].apply(
        lambda x: tokenize_text(x, remove_stopwords=True)
    )
    
    # Join tokens back to text for TF-IDF
    df['text_processed'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    # Remove empty texts
    df = df[df['text_processed'].str.len() > 0]
    
    logger.info(f"After preprocessing: {len(df)} samples")
    
    return df


def split_niche_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split niche data into train/validation/test sets."""
    from sklearn.model_selection import train_test_split
    
    # Stratified split to maintain class balance
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )
    
    logger.info(f"Data splits for {df['niche'].iloc[0]}:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Validation: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


class NicheModelTrainer:
    """Trainer for niche-specific models."""
    
    def __init__(self, niche: str, config: Dict[str, Any], random_state: int = 42):
        self.niche = niche
        self.config = config
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.vectorizer = None
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train multiple models for the niche."""
        
        # Create feature extractor (adjusted for small datasets)
        self.vectorizer = TFIDFFeatureExtractor(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Fit vectorizer
        train_texts = pd.Series([f"dummy_{i}" for i in range(X_train.shape[0])])
        train_labels = pd.Series(y_train)
        self.vectorizer.fit(train_texts, train_labels)
        
        # Train Logistic Regression
        self._train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Train Linear SVM
        self._train_svm(X_train, y_train, X_val, y_val)
        
        # Select best model
        self._select_best_model()
    
    def _train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> None:
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
            solver='liblinear'
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
            'train_accuracy': float(train_score),
            'val_accuracy': float(val_score),
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'model_name': 'LogisticRegression'
        }
        
        self.models['logistic_regression'] = model
        self.metrics['logistic_regression'] = metrics
        
        logger.info(f"Logistic Regression - Train: {train_score:.4f}, "
                   f"Val: {val_score:.4f}, CV F1: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    def _train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> None:
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
            'train_accuracy': float(train_score),
            'val_accuracy': float(val_score),
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'model_name': 'LinearSVM'
        }
        
        self.models['svm'] = model
        self.metrics['svm'] = metrics
        
        logger.info(f"Linear SVM - Train: {train_score:.4f}, "
                   f"Val: {val_score:.4f}, CV F1: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    def _select_best_model(self) -> None:
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
        
        logger.info(f"Best model for {self.niche}: {best_name} (CV F1: {best_score:.4f})")
    
    def save_models(self, output_dir: Path) -> None:
        """Save all trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = output_dir / f"model_{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = output_dir / f"model_{self.niche}.joblib"
            joblib.dump(self.best_model, best_model_path)
            logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")
        
        # Save vectorizer
        if self.vectorizer is not None:
            vectorizer_path = output_dir.parent / "vectorizers" / f"vectorizer_{self.niche}.joblib"
            vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
            self.vectorizer.save(vectorizer_path)
    
    def save_metrics(self, output_dir: Path) -> Dict[str, Any]:
        """Save training metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        metadata = {
            'niche': self.niche,
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'config': self.config,
            'models': self.metrics
        }
        
        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
        
        return metadata


def train_niche_model(niche: str) -> Dict[str, Any]:
    """Train models for a specific niche."""
    
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / niche / "processed" / "train.csv"
    output_dir = project_root / "artifacts" / niche
    
    # Load config and data
    config = load_niche_config(niche)
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please run data pipeline first")
        sys.exit(1)
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} samples for niche: {niche}")
    
    # Check minimum samples
    if not check_minimum_samples(df, config):
        sys.exit(1)
    
    # Preprocess data
    df = preprocess_niche_data(df)
    
    # Split data
    train_df, val_df, test_df = split_niche_data(df)
    
    # Create TF-IDF features (adjusted for small datasets)
    vectorizer = TFIDFFeatureExtractor(max_features=1000, ngram_range=(1, 2), min_df=1)
    
    X_train, y_train = vectorizer.fit_transform(train_df['text_processed'], train_df['label'])
    X_val, y_val = vectorizer.transform(val_df['text_processed'], val_df['label'])
    
    logger.info(f"Feature matrix shapes - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Train models
    trainer = NicheModelTrainer(niche, config)
    trainer.vectorizer = vectorizer
    trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Save models and metrics
    trainer.save_models(output_dir / "models")
    metadata = trainer.save_metrics(output_dir / "reports")
    
    return metadata


def main():
    """Main function for niche model training."""
    parser = argparse.ArgumentParser(description="Train models for a specific niche")
    parser.add_argument("--niche", required=True, help="Niche to train models for")
    
    args = parser.parse_args()
    
    try:
        metadata = train_niche_model(args.niche)
        
        # Print summary
        logger.info(f"\nTraining Summary for {args.niche}:")
        for model_name, metrics in metadata['models'].items():
            logger.info(f"{model_name}:")
            logger.info(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
            logger.info(f"  Val Accuracy: {metrics['val_accuracy']:.4f}")
            logger.info(f"  CV F1: {metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}")
        
        logger.info(f"\nBest Model: {metadata['best_model']}")
        
    except Exception as e:
        logger.error(f"Training failed for {args.niche}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
