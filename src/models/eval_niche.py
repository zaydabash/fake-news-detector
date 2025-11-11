"""Per-niche model evaluation pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# import scikitplot as skplt  # Commented out due to compatibility issues
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.features.tfidf import TFIDFFeatureExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_niche_config(niche: str) -> Dict[str, Any]:
    """Load configuration for a specific niche."""
    config_path = Path(__file__).parent.parent.parent / "configs" / f"{niche}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


class NicheModelEvaluator:
    """Evaluator for niche-specific models."""

    def __init__(self, niche: str, config: Dict[str, Any]):
        self.niche = niche
        self.config = config
        self.results = {}

    def evaluate_model(
        self, model_path: Path, vectorizer_path: Path, test_data_path: Path
    ) -> Dict[str, Any]:
        """Evaluate niche model on test data."""
        logger.info(f"Evaluating {self.niche} model...")

        # Load model and vectorizer
        model = joblib.load(model_path)
        vectorizer = TFIDFFeatureExtractor.load(vectorizer_path)

        # Load test data
        test_df = pd.read_csv(test_data_path)

        # Preprocess test data
        from src.data.clean import clean_text, tokenize_text

        test_df["text_clean"] = test_df["text"].apply(clean_text)
        test_df["tokens"] = test_df["text_clean"].apply(
            lambda x: tokenize_text(x, remove_stopwords=True)
        )
        test_df["text_processed"] = test_df["tokens"].apply(lambda x: " ".join(x))

        # Filter out empty texts
        test_df = test_df[test_df["text_processed"].str.len() > 0]

        # Transform features
        X_test, y_test = vectorizer.transform(
            test_df["text_processed"], test_df["label"]
        )

        # Ensure y_test is properly encoded
        if y_test is not None and len(y_test) > 0:
            # Convert string labels to integers if needed
            label_encoder = vectorizer.label_encoder
            if hasattr(label_encoder, "classes_"):
                y_test = label_encoder.transform(test_df["label"])

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )

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
            "niche": self.niche,
            "accuracy": float(accuracy),
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "predictions": y_pred.tolist(),
            "true_labels": y_test.tolist(),
            "test_samples": len(test_df),
        }

        # ROC and PR curves if probabilities available
        if y_pred_proba is not None:
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(
                y_test, y_pred_proba[:, 1]
            )
            avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])

            results.update(
                {
                    "roc_auc": float(roc_auc),
                    "avg_precision": float(avg_precision),
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                }
            )

        self.results = results
        return results

    def generate_plots(self, output_dir: Path) -> None:
        """Generate evaluation plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        self._plot_confusion_matrix(output_dir)

        # ROC curve
        if "roc_auc" in self.results:
            self._plot_roc_curve(output_dir)

        # PR curve
        if "avg_precision" in self.results:
            self._plot_precision_recall_curve(output_dir)

        # Feature importance
        self._plot_feature_importance(output_dir)

    def _plot_confusion_matrix(self, output_dir: Path) -> None:
        """Plot confusion matrix."""
        cm = np.array(self.results["confusion_matrix"])

        # Get class labels from config
        positive_label = self.config["positive_label"]
        negative_label = self.config["negative_label"]

        # Matplotlib version
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[negative_label, positive_label],
            yticklabels=[negative_label, positive_label],
        )
        plt.title(f"Confusion Matrix - {self.niche.title()}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"confusion_matrix_{self.niche}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_roc_curve(self, output_dir: Path) -> None:
        """Plot ROC curve."""
        fpr = np.array(self.results["fpr"])
        tpr = np.array(self.results["tpr"])
        roc_auc = self.results["roc_auc"]

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.niche.title()}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"roc_curve_{self.niche}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_precision_recall_curve(self, output_dir: Path) -> None:
        """Plot Precision-Recall curve."""
        precision = np.array(self.results["precision"])
        recall = np.array(self.results["recall"])
        avg_precision = self.results["avg_precision"]

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall,
            precision,
            color="darkorange",
            lw=2,
            label=f"PR curve (AP = {avg_precision:.2f})",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {self.niche.title()}")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"precision_recall_curve_{self.niche}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_feature_importance(self, output_dir: Path) -> None:
        """Plot feature importance."""
        try:
            # Load model and vectorizer for feature analysis
            model_path = (
                Path(__file__).parent.parent.parent
                / "artifacts"
                / self.niche
                / "models"
                / f"model_{self.niche}.joblib"
            )
            vectorizer_path = (
                Path(__file__).parent.parent.parent
                / "artifacts"
                / self.niche
                / "vectorizers"
                / f"vectorizer_{self.niche}.joblib"
            )

            model = joblib.load(model_path)
            vectorizer = TFIDFFeatureExtractor.load(vectorizer_path)

            top_features = vectorizer.get_top_features(model, n_features=20)

            # Create feature importance plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Positive class features
            positive_label = self.config["positive_label"]
            positive_features = top_features["class_1"]  # Assuming positive is class 1
            positive_words = [feat[0] for feat in positive_features]
            positive_scores = [feat[1] for feat in positive_features]

            ax1.barh(range(len(positive_words)), positive_scores)
            ax1.set_yticks(range(len(positive_words)))
            ax1.set_yticklabels(positive_words)
            ax1.set_xlabel("Coefficient")
            ax1.set_title(f"Top Features for {positive_label.title()}")
            ax1.grid(True, alpha=0.3)

            # Negative class features
            negative_label = self.config["negative_label"]
            negative_features = top_features["class_0"]  # Assuming negative is class 0
            negative_words = [feat[0] for feat in negative_features]
            negative_scores = [feat[1] for feat in negative_features]

            ax2.barh(range(len(negative_words)), negative_scores)
            ax2.set_yticks(range(len(negative_words)))
            ax2.set_yticklabels(negative_words)
            ax2.set_xlabel("Coefficient")
            ax2.set_title(f"Top Features for {negative_label.title()}")
            ax2.grid(True, alpha=0.3)

            plt.suptitle(f"Feature Importance - {self.niche.title()}")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"feature_importance_{self.niche}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        except Exception as e:
            logger.warning(
                f"Could not generate feature importance plot for {self.niche}: {e}"
            )

    def save_results(self, output_dir: Path) -> None:
        """Save evaluation results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        results_path = output_dir / f"evaluation_metrics_{self.niche}.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Generate plots
        self.generate_plots(output_dir)

        # Print summary
        logger.info(f"\nEvaluation Summary for {self.niche}:")
        logger.info(f"Accuracy: {self.results['accuracy']:.4f}")
        logger.info(f"Test samples: {self.results['test_samples']}")

        if "roc_auc" in self.results:
            logger.info(f"ROC AUC: {self.results['roc_auc']:.4f}")
            logger.info(f"Average Precision: {self.results['avg_precision']:.4f}")

        # Print classification report
        class_report = self.results["classification_report"]
        positive_label = self.config["positive_label"]
        negative_label = self.config["negative_label"]

        logger.info("\nClassification Report:")
        if "0" in class_report:
            logger.info(
                f"{negative_label} - Precision: {class_report['0']['precision']:.4f}, "
                f"Recall: {class_report['0']['recall']:.4f}, "
                f"F1: {class_report['0']['f1-score']:.4f}"
            )
        if "1" in class_report:
            logger.info(
                f"{positive_label} - Precision: {class_report['1']['precision']:.4f}, "
                f"Recall: {class_report['1']['recall']:.4f}, "
                f"F1: {class_report['1']['f1-score']:.4f}"
            )

        if "macro avg" in class_report:
            logger.info(
                f"Macro Avg - Precision: {class_report['macro avg']['precision']:.4f}, "
                f"Recall: {class_report['macro avg']['recall']:.4f}, "
                f"F1: {class_report['macro avg']['f1-score']:.4f}"
            )


def evaluate_niche_model(niche: str) -> Dict[str, Any]:
    """Evaluate model for a specific niche."""

    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    artifacts_dir = project_root / "artifacts" / niche
    data_dir = project_root / "data" / niche / "processed"

    model_path = artifacts_dir / "models" / f"model_{niche}.joblib"
    vectorizer_path = artifacts_dir / "vectorizers" / f"vectorizer_{niche}.joblib"
    test_data_path = data_dir / "train.csv"  # Using train.csv as test data for now

    output_dir = artifacts_dir / "reports"

    # Check if required files exist
    required_files = [model_path, vectorizer_path, test_data_path]
    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"Required file not found: {file_path}")
            logger.error("Please run training pipeline first")
            sys.exit(1)

    # Load config
    config = load_niche_config(niche)

    # Evaluate model
    evaluator = NicheModelEvaluator(niche, config)
    results = evaluator.evaluate_model(model_path, vectorizer_path, test_data_path)

    # Save results
    evaluator.save_results(output_dir)

    return results


def main():
    """Main function for niche model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model for a specific niche")
    parser.add_argument("--niche", required=True, help="Niche to evaluate model for")

    args = parser.parse_args()

    try:
        results = evaluate_niche_model(args.niche)
        logger.info(f"Evaluation complete for {args.niche}")

    except Exception as e:
        logger.error(f"Evaluation failed for {args.niche}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
