#!/bin/bash

# Full pipeline execution script
# This script runs the complete fake news detection pipeline from start to finish

set -e  # Exit on any error

echo "🚀 Starting Fake News Detector Pipeline"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run 'make setup' first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Set random seed for reproducibility
export PYTHONHASHSEED=42
export RANDOM_SEED=42

echo ""
echo "📊 Step 1: Data Pipeline"
echo "------------------------"
echo "Downloading and preprocessing LIAR dataset..."
python -m src.data.download_liar
python -m src.data.clean

echo ""
echo "🔧 Step 2: Feature Engineering"
echo "-----------------------------"
echo "Creating TF-IDF features..."
python -m src.features.tfidf

echo ""
echo "🤖 Step 3: Model Training"
echo "-------------------------"
echo "Training models and selecting best performer..."
python -m src.models.train

echo ""
echo "📈 Step 4: Model Evaluation"
echo "---------------------------"
echo "Generating comprehensive evaluation reports..."
python -m src.models.eval

echo ""
echo "🧪 Step 5: Testing Predictions"
echo "------------------------------"
echo "Testing the trained model on sample texts..."

# Test single prediction
echo "Testing single text prediction:"
python -m src.predict.cli predict-text "The Federal Reserve raised interest rates today."

# Test batch prediction
echo ""
echo "Testing batch prediction:"
python -m src.predict.cli predict-file examples/sample_texts.csv --text-column text

echo ""
echo "✅ Pipeline Complete!"
echo "===================="
echo ""
echo "Results saved to:"
echo "  📁 artifacts/models/     - Trained models"
echo "  📁 artifacts/vectorizers/ - Feature extractors"
echo "  📁 artifacts/reports/    - Evaluation reports and plots"
echo ""
echo "Next steps:"
echo "  🌐 Run Streamlit demo: streamlit run app/streamlit_app.py"
echo "  📊 View results: open artifacts/reports/"
echo "  🔍 Try predictions: make predict TEXT=\"your text here\""
echo ""
echo "🎉 Fake News Detector is ready for use!"
