"""Command-line interface for fake news prediction."""

import sys
from pathlib import Path
from typing import Optional, List
import json

import pandas as pd
import typer
import joblib
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.features.tfidf import TFIDFFeatureExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()
app = typer.Typer(help="Fake News Detection CLI")


class FakeNewsPredictor:
    """Predictor for fake news detection."""
    
    def __init__(self, model_path: Path, vectorizer_path: Path):
        """Initialize predictor with trained model and vectorizer."""
        self.model = joblib.load(model_path)
        self.vectorizer = TFIDFFeatureExtractor.load(vectorizer_path)
        
        # Get class names
        self.class_names = self.vectorizer.label_encoder.classes_
        logger.info(f"Loaded model with classes: {list(self.class_names)}")
    
    def predict_text(self, text: str) -> dict:
        """Predict fake news for a single text."""
        # Preprocess text
        from src.data.clean import clean_text, tokenize_text
        
        cleaned_text = clean_text(text)
        tokens = tokenize_text(cleaned_text, remove_stopwords=True)
        processed_text = ' '.join(tokens)
        
        if not processed_text.strip():
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'probabilities': {'Real': 0.5, 'Fake': 0.5},
                'error': 'Text is too short or contains no meaningful content'
            }
        
        # Transform text
        X, _ = self.vectorizer.transform(pd.Series([processed_text]))
        
        # Predict
        prediction = self.model.predict(X)[0]
        prediction_class = self.vectorizer.inverse_transform_labels([prediction])[0]
        
        # Get probabilities if available
        probabilities = None
        confidence = 0.0
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            probabilities = {
                self.class_names[0]: float(proba[0]),
                self.class_names[1]: float(proba[1])
            }
            confidence = float(max(proba))
        
        return {
            'prediction': str(prediction_class),
            'confidence': confidence,
            'probabilities': probabilities,
            'processed_text': processed_text
        }
    
    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Predict fake news for a batch of texts."""
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.predict_text(text)
                result['index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting text {i}: {e}")
                results.append({
                    'index': i,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'probabilities': {'Real': 0.5, 'Fake': 0.5},
                    'error': str(e)
                })
        
        return results


def load_predictor() -> FakeNewsPredictor:
    """Load the trained predictor."""
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "artifacts" / "models" / "best_model.joblib"
    vectorizer_path = project_root / "artifacts" / "vectorizers" / "tfidf_vectorizer.joblib"
    
    if not model_path.exists():
        console.print(f"[red]Error: Model not found at {model_path}[/red]")
        console.print("Please run 'make train' first to train the model.")
        raise typer.Exit(1)
    
    if not vectorizer_path.exists():
        console.print(f"[red]Error: Vectorizer not found at {vectorizer_path}[/red]")
        console.print("Please run 'make data' first to prepare the data.")
        raise typer.Exit(1)
    
    return FakeNewsPredictor(model_path, vectorizer_path)


def display_prediction(result: dict, text: str) -> None:
    """Display prediction results in a formatted way."""
    # Create prediction panel
    prediction_text = f"[bold]{result['prediction']}[/bold]"
    confidence_text = f"Confidence: {result['confidence']:.2%}"
    
    if result['prediction'] == 'Real':
        color = "green"
        emoji = "✅"
    elif result['prediction'] == 'Fake':
        color = "red"
        emoji = "❌"
    else:
        color = "yellow"
        emoji = "⚠️"
    
    # Main prediction panel
    panel = Panel(
        f"{emoji} {prediction_text}\n{confidence_text}",
        title="Prediction",
        border_style=color,
        padding=(1, 2)
    )
    console.print(panel)
    
    # Probabilities table
    if result['probabilities']:
        table = Table(title="Prediction Probabilities")
        table.add_column("Class", style="cyan")
        table.add_column("Probability", style="magenta")
        
        for class_name, prob in result['probabilities'].items():
            table.add_row(class_name, f"{prob:.2%}")
        
        console.print(table)
    
    # Error message if any
    if 'error' in result:
        console.print(f"[yellow]Warning: {result['error']}[/yellow]")
    
    # Show processed text
    if 'processed_text' in result and result['processed_text']:
        console.print(f"\n[dim]Processed text: {result['processed_text'][:100]}...[/dim]")


@app.command()
def predict_text(
    text: str = typer.Argument(..., help="Text to classify as fake or real news")
):
    """Predict if a single text is fake news."""
    try:
        predictor = load_predictor()
        result = predictor.predict_text(text)
        
        console.print(f"\n[bold blue]Analyzing text:[/bold blue]")
        console.print(f"[italic]\"{text[:100]}{'...' if len(text) > 100 else ''}\"[/italic]\n")
        
        display_prediction(result, text)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def predict_file(
    file_path: str = typer.Argument(..., help="Path to CSV file with texts to classify"),
    text_column: str = typer.Option("text", help="Name of the column containing texts"),
    output_file: Optional[str] = typer.Option(None, help="Output file path (default: predictions.csv)")
):
    """Predict fake news for texts in a CSV file."""
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns:
            console.print(f"[red]Error: Column '{text_column}' not found in CSV file.[/red]")
            console.print(f"Available columns: {list(df.columns)}")
            raise typer.Exit(1)
        
        console.print(f"Loaded {len(df)} texts from {file_path}")
        
        # Load predictor
        predictor = load_predictor()
        
        # Predict
        console.print("Making predictions...")
        texts = df[text_column].astype(str).tolist()
        results = predictor.predict_batch(texts)
        
        # Add predictions to dataframe
        df['prediction'] = [r['prediction'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        if results[0]['probabilities']:
            for class_name in results[0]['probabilities'].keys():
                df[f'prob_{class_name.lower()}'] = [
                    r['probabilities'][class_name] for r in results
                ]
        
        # Save results
        output_path = output_file or "predictions.csv"
        df.to_csv(output_path, index=False)
        
        console.print(f"[green]Predictions saved to {output_path}[/green]")
        
        # Show summary
        prediction_counts = df['prediction'].value_counts()
        console.print(f"\n[bold]Summary:[/bold]")
        for pred, count in prediction_counts.items():
            console.print(f"  {pred}: {count} ({count/len(df):.1%})")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def interactive():
    """Interactive mode for continuous prediction."""
    console.print("[bold blue]Fake News Detector - Interactive Mode[/bold blue]")
    console.print("Type 'quit' or 'exit' to stop.\n")
    
    try:
        predictor = load_predictor()
        
        while True:
            text = typer.prompt("\nEnter text to classify")
            
            if text.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not text.strip():
                console.print("[yellow]Please enter some text.[/yellow]")
                continue
            
            result = predictor.predict_text(text)
            display_prediction(result, text)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
