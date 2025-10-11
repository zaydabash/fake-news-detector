"""TF-IDF feature extraction pipeline."""

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TFIDFFeatureExtractor:
    """TF-IDF feature extractor with preprocessing."""
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 3,
        max_df: float = 0.95,
        lowercase: bool = True,
        stop_words: str = 'english'
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase
        self.stop_words = stop_words
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase,
            stop_words=stop_words,
            sublinear_tf=True  # Apply sublinear TF scaling
        )
        
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def fit(self, texts: pd.Series, labels: pd.Series) -> 'TFIDFFeatureExtractor':
        """Fit the vectorizer and label encoder."""
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} samples")
        logger.info(f"Parameters: max_features={self.max_features}, "
                   f"ngram_range={self.ngram_range}, min_df={self.min_df}")
        
        # Fit vectorizer
        self.vectorizer.fit(texts)
        
        # Fit label encoder
        self.label_encoder.fit(labels)
        
        self.is_fitted = True
        
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"Label classes: {list(self.label_encoder.classes_)}")
        
        return self
    
    def transform(self, texts: pd.Series, labels: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """Transform texts to TF-IDF features."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        logger.info(f"Transforming {len(texts)} samples")
        
        # Transform texts
        X = self.vectorizer.transform(texts)
        
        # Transform labels if provided
        y = None
        if labels is not None:
            y = self.label_encoder.transform(labels)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        if y is not None:
            logger.info(f"Label distribution: {np.bincount(y)}")
        
        return X, y
    
    def fit_transform(self, texts: pd.Series, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        return self.fit(texts, labels).transform(texts, labels)
    
    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original labels."""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_feature_names(self) -> np.ndarray:
        """Get feature names from the vectorizer."""
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, model, n_features: int = 20) -> dict:
        """Get top features for each class."""
        if not hasattr(model, 'coef_'):
            raise ValueError("Model must have coef_ attribute")
        
        feature_names = self.get_feature_names()
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        
        # Get top features for each class
        top_indices = np.argsort(coef)[::-1][:n_features]
        bottom_indices = np.argsort(coef)[:n_features]
        
        top_features = {
            'class_0': [(feature_names[i], coef[i]) for i in top_indices],
            'class_1': [(feature_names[i], coef[i]) for i in bottom_indices]
        }
        
        return top_features
    
    def save(self, filepath: Path) -> None:
        """Save the feature extractor."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted extractor")
        
        joblib.dump(self, filepath)
        logger.info(f"Saved feature extractor to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'TFIDFFeatureExtractor':
        """Load a feature extractor."""
        extractor = joblib.load(filepath)
        logger.info(f"Loaded feature extractor from {filepath}")
        return extractor


def create_features(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    output_dir: Path,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3
) -> None:
    """Create TF-IDF features for all datasets."""
    
    # Load datasets
    logger.info("Loading datasets...")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Validation: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")
    
    # Create feature extractor
    extractor = TFIDFFeatureExtractor(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df
    )
    
    # Fit on training data
    logger.info("Fitting TF-IDF vectorizer...")
    X_train, y_train = extractor.fit_transform(
        train_df['text_processed'],
        train_df['label_binary']
    )
    
    # Transform validation and test data
    logger.info("Transforming validation data...")
    X_val, y_val = extractor.transform(
        val_df['text_processed'],
        val_df['label_binary']
    )
    
    logger.info("Transforming test data...")
    X_test, y_test = extractor.transform(
        test_df['text_processed'],
        test_df['label_binary']
    )
    
    # Save feature extractor
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor_path = output_dir / "tfidf_vectorizer.joblib"
    extractor.save(extractor_path)
    
    # Save processed features
    logger.info("Saving processed features...")
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    logger.info(f"Saved features to {output_dir}")
    
    # Print feature statistics
    logger.info(f"Feature matrix shapes:")
    logger.info(f"Train: {X_train.shape}")
    logger.info(f"Validation: {X_val.shape}")
    logger.info(f"Test: {X_test.shape}")
    
    # Print vocabulary statistics
    feature_names = extractor.get_feature_names()
    logger.info(f"Vocabulary size: {len(feature_names)}")
    logger.info(f"Sample features: {feature_names[:10].tolist()}")


def main() -> None:
    """Main function to create features."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "processed"
    output_dir = project_root / "artifacts" / "vectorizers"
    
    train_file = data_dir / "train.csv"
    val_file = data_dir / "val.csv"
    test_file = data_dir / "test.csv"
    
    # Check if input files exist
    for file_path in [train_file, val_file, test_file]:
        if not file_path.exists():
            logger.error(f"Input file not found: {file_path}")
            logger.error("Please run data cleaning pipeline first")
            sys.exit(1)
    
    # Create features with default parameters
    create_features(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        output_dir=output_dir,
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3
    )


if __name__ == "__main__":
    main()
