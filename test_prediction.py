#!/usr/bin/env python3
"""Quick test of the trained climate_claims model."""

import sys
from pathlib import Path
import joblib
import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.features.tfidf import TFIDFFeatureExtractor

def test_climate_model():
    """Test the trained climate_claims model."""
    
    # Load model and vectorizer
    model_path = project_root / "artifacts" / "climate_claims" / "models" / "model_climate_claims.joblib"
    vectorizer_path = project_root / "artifacts" / "climate_claims" / "vectorizers" / "vectorizer_climate_claims.joblib"
    
    model = joblib.load(model_path)
    vectorizer = TFIDFFeatureExtractor.load(vectorizer_path)
    
    # Test texts
    test_texts = [
        "Climate change is a hoax! Cities will be underwater next year!",
        "The IPCC report shows significant radiative forcing from greenhouse gas emissions.",
        "Climate lockdowns are coming to control everyone!",
        "Our methodology uses peer-reviewed studies with confidence intervals.",
        "Climate scam exposed! Scientists are lying about global warming!"
    ]
    
    print("🧪 Testing Climate Claims Model")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        # Preprocess text
        from src.data.clean import clean_text, tokenize_text
        
        cleaned_text = clean_text(text)
        tokens = tokenize_text(cleaned_text, remove_stopwords=True)
        processed_text = ' '.join(tokens)
        
        # Transform
        X, _ = vectorizer.transform(pd.Series([processed_text]))
        
        # Predict
        prediction = model.predict(X)[0]
        prediction_label = vectorizer.inverse_transform_labels([prediction])[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            confidence = max(proba)
        else:
            confidence = 1.0
        
        print(f"\n{i}. Text: {text[:60]}...")
        print(f"   Prediction: {prediction_label}")
        print(f"   Confidence: {confidence:.2%}")
        if hasattr(model, 'predict_proba'):
            print(f"   Probabilities: {dict(zip(vectorizer.label_encoder.classes_, proba))}")

if __name__ == "__main__":
    test_climate_model()
