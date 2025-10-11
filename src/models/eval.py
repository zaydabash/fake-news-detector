"""Model evaluation and reporting pipeline."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import scikitplot as skplt
import joblib

from src.features.tfidf import TFIDFFeatureExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model_path: Path, vectorizer_path: Path):
        self.model = joblib.load(model_path)
        self.vectorizer = TFIDFFeatureExtractor.load(vectorizer_path)
        self.results = {}
    
    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_texts: pd.Series
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info("Evaluating model on test set...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Basic metrics
        accuracy = (y_pred == y_test).mean()
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }
        
        # ROC and PR curves if probabilities available
        if y_pred_proba is not None:
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
            avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
            
            results.update({
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'precision': precision.tolist(),
                'recall': recall.tolist()
            })
        
        self.results = results
        return results
    
    def plot_confusion_matrix(self, output_dir: Path) -> None:
        """Plot and save confusion matrix."""
        cm = np.array(self.results['confusion_matrix'])
        
        # Matplotlib version
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plotly version
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Real', 'Fake'],
            y=['Real', 'Fake'],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=600,
            height=500
        )
        
        fig.write_html(output_dir / 'confusion_matrix.html')
        logger.info(f"Saved confusion matrix plots to {output_dir}")
    
    def plot_roc_curve(self, output_dir: Path) -> None:
        """Plot and save ROC curve."""
        if 'roc_auc' not in self.results:
            logger.warning("ROC curve not available (no probability predictions)")
            return
        
        fpr = np.array(self.results['fpr'])
        tpr = np.array(self.results['tpr'])
        roc_auc = self.results['roc_auc']
        
        # Matplotlib version
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plotly version
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.2f})',
            line=dict(color='darkorange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=600,
            height=500
        )
        
        fig.write_html(output_dir / 'roc_curve.html')
        logger.info(f"Saved ROC curve plots to {output_dir}")
    
    def plot_precision_recall_curve(self, output_dir: Path) -> None:
        """Plot and save Precision-Recall curve."""
        if 'avg_precision' not in self.results:
            logger.warning("PR curve not available (no probability predictions)")
            return
        
        precision = np.array(self.results['precision'])
        recall = np.array(self.results['recall'])
        avg_precision = self.results['avg_precision']
        
        # Matplotlib version
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plotly version
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR curve (AP = {avg_precision:.2f})',
            line=dict(color='darkorange', width=2)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=600,
            height=500
        )
        
        fig.write_html(output_dir / 'precision_recall_curve.html')
        logger.info(f"Saved PR curve plots to {output_dir}")
    
    def analyze_top_features(self, output_dir: Path, n_features: int = 20) -> None:
        """Analyze and plot top features."""
        try:
            top_features = self.vectorizer.get_top_features(self.model, n_features)
            
            # Create feature importance plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Real news features
            real_features = top_features['class_0']
            real_words = [feat[0] for feat in real_features]
            real_scores = [feat[1] for feat in real_features]
            
            ax1.barh(range(len(real_words)), real_scores)
            ax1.set_yticks(range(len(real_words)))
            ax1.set_yticklabels(real_words)
            ax1.set_xlabel('Coefficient')
            ax1.set_title('Top Features for Real News')
            ax1.grid(True, alpha=0.3)
            
            # Fake news features
            fake_features = top_features['class_1']
            fake_words = [feat[0] for feat in fake_features]
            fake_scores = [feat[1] for feat in fake_features]
            
            ax2.barh(range(len(fake_words)), fake_scores)
            ax2.set_yticks(range(len(fake_words)))
            ax2.set_yticklabels(fake_words)
            ax2.set_xlabel('Coefficient')
            ax2.set_title('Top Features for Fake News')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature analysis as JSON
            feature_analysis = {
                'real_news_features': [{'word': w, 'coefficient': c} for w, c in real_features],
                'fake_news_features': [{'word': w, 'coefficient': c} for w, c in fake_features]
            }
            
            with open(output_dir / 'feature_analysis.json', 'w') as f:
                json.dump(feature_analysis, f, indent=2)
            
            logger.info(f"Saved feature analysis to {output_dir}")
            
        except Exception as e:
            logger.warning(f"Could not analyze features: {e}")
    
    def generate_report(self, output_dir: Path) -> None:
        """Generate comprehensive evaluation report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate plots
        self.plot_confusion_matrix(output_dir)
        self.plot_roc_curve(output_dir)
        self.plot_precision_recall_curve(output_dir)
        self.analyze_top_features(output_dir)
        
        # Print summary
        logger.info("\nEvaluation Summary:")
        logger.info(f"Accuracy: {self.results['accuracy']:.4f}")
        
        if 'roc_auc' in self.results:
            logger.info(f"ROC AUC: {self.results['roc_auc']:.4f}")
            logger.info(f"Average Precision: {self.results['avg_precision']:.4f}")
        
        # Print classification report
        class_report = self.results['classification_report']
        logger.info("\nClassification Report:")
        logger.info(f"Real News - Precision: {class_report['0']['precision']:.4f}, "
                   f"Recall: {class_report['0']['recall']:.4f}, "
                   f"F1: {class_report['0']['f1-score']:.4f}")
        logger.info(f"Fake News - Precision: {class_report['1']['precision']:.4f}, "
                   f"Recall: {class_report['1']['recall']:.4f}, "
                   f"F1: {class_report['1']['f1-score']:.4f}")
        logger.info(f"Macro Avg - Precision: {class_report['macro avg']['precision']:.4f}, "
                   f"Recall: {class_report['macro avg']['recall']:.4f}, "
                   f"F1: {class_report['macro avg']['f1-score']:.4f}")


def evaluate_model(
    model_path: Path,
    vectorizer_path: Path,
    features_dir: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Evaluate a trained model."""
    
    # Load test features
    X_test = np.load(features_dir / "X_test.npy")
    y_test = np.load(features_dir / "y_test.npy")
    
    # Load test texts for analysis
    test_df = pd.read_csv(features_dir.parent / "processed" / "test.csv")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, vectorizer_path)
    
    # Evaluate
    results = evaluator.evaluate_model(X_test, y_test, test_df['text_processed'])
    
    # Generate report
    evaluator.generate_report(output_dir)
    
    return results


def main() -> None:
    """Main function to evaluate models."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    artifacts_dir = project_root / "artifacts"
    features_dir = artifacts_dir / "vectorizers"
    
    model_path = artifacts_dir / "models" / "best_model.joblib"
    vectorizer_path = features_dir / "tfidf_vectorizer.joblib"
    output_dir = artifacts_dir / "reports"
    
    # Check if required files exist
    required_files = [model_path, vectorizer_path]
    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"Required file not found: {file_path}")
            logger.error("Please run training pipeline first")
            sys.exit(1)
    
    # Evaluate model
    results = evaluate_model(
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        features_dir=features_dir,
        output_dir=output_dir
    )
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
