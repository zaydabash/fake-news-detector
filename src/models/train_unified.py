"""Unified multi-niche model training."""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

import argparse

from src.features.tfidf import TFIDFFeatureExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_all_niche_data(niches: List[str]) -> pd.DataFrame:
    """Load and combine data from all niches."""
    project_root = Path(__file__).parent.parent.parent
    all_data = []
    
    for niche in niches:
        data_file = project_root / "data" / niche / "processed" / "train.csv"
        
        if not data_file.exists():
            logger.warning(f"Data file not found for {niche}: {data_file}")
            continue
        
        df = pd.read_csv(data_file)
        df['niche'] = niche  # Ensure niche column exists
        
        # Add binary labels for unified training
        # Map niche-specific labels to binary: positive=1, negative=0
        niche_config = load_niche_config(niche)
        df['label_binary'] = df['label'].map({
            niche_config['positive_label']: 1,
            niche_config['negative_label']: 0
        })
        
        # Remove unlabeled samples
        df = df.dropna(subset=['label_binary'])
        
        all_data.append(df)
        logger.info(f"Loaded {len(df)} samples from {niche}")
    
    if not all_data:
        raise ValueError("No niche data found")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} samples from {len(niches)} niches")
    
    return combined_df


def load_niche_config(niche: str) -> Dict[str, Any]:
    """Load configuration for a specific niche."""
    config_path = Path(__file__).parent.parent.parent / "configs" / f"{niche}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def preprocess_unified_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess unified dataset."""
    from src.data.clean import clean_text, tokenize_text
    
    logger.info("Preprocessing unified dataset...")
    
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


def split_unified_data(df: pd.DataFrame) -> tuple:
    """Split unified data with stratification by both class and niche."""
    from sklearn.model_selection import train_test_split
    
    # Create stratification labels combining class and niche
    df['stratify_label'] = df['label_binary'].astype(str) + '_' + df['niche']
    
    # Stratified split to maintain balance
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['stratify_label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['stratify_label']
    )
    
    logger.info(f"Unified data splits:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Validation: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    # Show distribution by niche
    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        logger.info(f"  {split_name} by niche:")
        niche_counts = split_df['niche'].value_counts()
        for niche, count in niche_counts.items():
            logger.info(f"    {niche}: {count}")
    
    return train_df, val_df, test_df


class UnifiedModelTrainer:
    """Trainer for unified multi-niche models."""
    
    def __init__(self, niches: List[str], random_state: int = 42):
        self.niches = niches
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.vectorizer = None
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train multiple models for unified classification."""
        
        # Train Logistic Regression
        self._train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Train Linear SVM
        self._train_svm(X_train, y_train, X_val, y_val)
        
        # Select best model
        self._select_best_model()
    
    def _train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train Logistic Regression model."""
        logger.info("Training Unified Logistic Regression...")
        
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
            'model_name': 'UnifiedLogisticRegression'
        }
        
        self.models['logistic_regression'] = model
        self.metrics['logistic_regression'] = metrics
        
        logger.info(f"Unified Logistic Regression - Train: {train_score:.4f}, "
                   f"Val: {val_score:.4f}, CV F1: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    def _train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train Linear SVM model."""
        logger.info("Training Unified Linear SVM...")
        
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
            'model_name': 'UnifiedLinearSVM'
        }
        
        self.models['svm'] = model
        self.metrics['svm'] = metrics
        
        logger.info(f"Unified Linear SVM - Train: {train_score:.4f}, "
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
        
        logger.info(f"Best unified model: {best_name} (CV F1: {best_score:.4f})")
    
    def save_models(self, output_dir: Path) -> None:
        """Save all trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = output_dir / f"model_unified_{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved unified {name} to {model_path}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = output_dir / "model_multiniche.joblib"
            joblib.dump(self.best_model, best_model_path)
            logger.info(f"Saved best unified model ({self.best_model_name}) to {best_model_path}")
        
        # Save vectorizer
        if self.vectorizer is not None:
            vectorizer_path = output_dir / "vectorizer_multiniche.joblib"
            self.vectorizer.save(vectorizer_path)
    
    def save_metrics(self, output_dir: Path, niches: List[str]) -> Dict[str, Any]:
        """Save training metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        metadata = {
            'model_type': 'unified_multiniche',
            'niches': niches,
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'models': self.metrics
        }
        
        metrics_path = output_dir / "unified_training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved unified metrics to {metrics_path}")
        
        return metadata


def train_unified_model(niches: List[str]) -> Dict[str, Any]:
    """Train unified model for multiple niches."""
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "artifacts" / "unified"
    
    logger.info(f"Training unified model for niches: {niches}")
    
    # Load all niche data
    combined_df = load_all_niche_data(niches)
    
    # Preprocess data
    combined_df = preprocess_unified_data(combined_df)
    
    # Split data
    train_df, val_df, test_df = split_unified_data(combined_df)
    
    # Create TF-IDF features
    vectorizer = TFIDFFeatureExtractor(max_features=15000, ngram_range=(1, 2), min_df=3)
    
    X_train, y_train = vectorizer.fit_transform(train_df['text_processed'], train_df['label_binary'])
    X_val, y_val = vectorizer.transform(val_df['text_processed'], val_df['label_binary'])
    
    logger.info(f"Unified feature matrix shapes - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Train models
    trainer = UnifiedModelTrainer(niches)
    trainer.vectorizer = vectorizer
    trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Save models and metrics
    trainer.save_models(output_dir / "models")
    metadata = trainer.save_metrics(output_dir / "reports", niches)
    
    return metadata


def main():
    """Main function for unified model training."""
    parser = argparse.ArgumentParser(description="Train unified model for multiple niches")
    parser.add_argument("--niches", required=True, 
                       help="Comma-separated list of niches to include")
    
    args = parser.parse_args()
    
    niches = [niche.strip() for niche in args.niches.split(',')]
    
    try:
        metadata = train_unified_model(niches)
        
        # Print summary
        logger.info(f"\nUnified Training Summary for {len(niches)} niches:")
        for model_name, metrics in metadata['models'].items():
            logger.info(f"{model_name}:")
            logger.info(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
            logger.info(f"  Val Accuracy: {metrics['val_accuracy']:.4f}")
            logger.info(f"  CV F1: {metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}")
        
        logger.info(f"\nBest Unified Model: {metadata['best_model']}")
        logger.info(f"Trained on niches: {', '.join(metadata['niches'])}")
        
    except Exception as e:
        logger.error(f"Unified training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
